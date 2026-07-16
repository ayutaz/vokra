//! HTTP + Wyoming listeners on the shared tokio runtime.
//!
//! T03 wires the two listeners and the `/health` endpoint. Later tickets
//! attach the OpenAI / vLLM / piper-plus routers (T06/T09/T11) and the
//! Wyoming event loop (T15/T16) here.
//!
//! M4-19 (T02/T04): the Wyoming accept loop is now service-aware. When an
//! engine registry ([`WyomingBackend`]) is wired it serves the full ASR + TTS
//! connection handler ([`run_wyoming_connection`]) through the M3-15
//! [`Scheduler`] (one permit + stream slot per connection); otherwise it falls
//! back to the discovery-only handler so Home Assistant's `describe` probe
//! still succeeds. The production startup path
//! ([`crate::server::run_with_config`]) has no CLI model wiring yet
//! (M2-09-T04 carry-over), so it passes `None` and stays discovery-only —
//! regression-free.

// ServiceError embeds the intentionally-rich VokraError verbatim (FR-EX-08
// failure-kind preservation); the T04 config→service helpers propagate it by
// value, matching service.rs / openai.rs / wyoming.rs. Boxing it just to
// satisfy the lint would obscure the failure kind for no real gain.
#![allow(clippy::result_large_err)]

use crate::api::wyoming::{
    BargeIn, WyomingBackend, run_describe_only_connection, run_wyoming_connection,
    write_wyoming_error,
};
use crate::config::Config;
use crate::error::spawn_isolated_wyoming_task;
use crate::scheduler::{Scheduler, SchedulerConfig};
use crate::service::{InferenceService, ServiceConfig, ServiceError, TranscribeService};
use crate::session::SessionRegistryConfig;
use crate::shutdown::{ShutdownSignal, ShutdownTrigger, install_shutdown_signal};
use axum::Router;
use axum::http::StatusCode;
use axum::routing::get;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

/// Default multi-session concurrency cap for the production startup path
/// (M4-19-T04). A conservative v0.5 default; a `--max-concurrent-sessions`
/// CLI knob is a follow-up. Must be the shared value for both the session
/// registry and the scheduler permit pool (see [`Scheduler::new`]).
const DEFAULT_MAX_CONCURRENT_SESSIONS: usize = 4;

/// Bound listener addresses returned by `spawn_server`. `run_with_config` uses
/// them for tests (so callers can hit `http_actual`) and for logging.
#[derive(Debug, Clone)]
pub struct ServerHandles {
    /// Actual HTTP bind address (may differ from the requested one if
    /// port `0` was passed — the OS picks a free port).
    pub http_actual: SocketAddr,
    /// Actual Wyoming TCP bind address, same caveat.
    pub wyoming_actual: SocketAddr,
}

/// Blocking entry point used by `main.rs`. Builds a multi-thread tokio
/// runtime, spawns HTTP + Wyoming listeners, waits for shutdown, drains.
pub fn run_with_config(cfg: Config) -> std::io::Result<()> {
    // Enforce LC_NUMERIC=C BEFORE the runtime spawns worker threads (see
    // `enforce_c_numeric_locale`'s SAFETY note).
    crate::enforce_c_numeric_locale();

    // Build the inference registry SYNCHRONOUSLY, before the runtime binds any
    // listener. When no model paths are configured this is `Ok(None)` and the
    // server boots health-only + Wyoming-discovery-only exactly as the M2-09
    // default did. When models ARE configured, a partial config or a broken /
    // missing GGUF is a hard startup error here (FR-EX-08 — never a silently
    // half-wired server that binds a port and then 404s / no-ops every
    // request).
    let service = build_service(&cfg).map_err(startup_io_error)?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let (signal, _trigger) = install_shutdown_signal();
        let handles = spawn_server_wired(cfg, signal.clone(), service).await?;
        eprintln!(
            "vokra-server: HTTP listening on {}, Wyoming listening on {}",
            handles.http_actual, handles.wyoming_actual
        );
        // `spawn_server` returns as soon as the listeners are bound and the
        // per-listener event loops are spawned as background tasks. Those
        // tasks live on the runtime until the shutdown signal fires — if we
        // returned from `block_on` here the runtime would be dropped and
        // every accept loop would stop mid-flight before ever seeing a
        // connection. Wait on the same signal the loops watch so this
        // future co-terminates with them (graceful drain).
        signal.wait().await;
        Ok::<_, std::io::Error>(())
    })
}

/// Bind both listeners and spawn their event loops. Returns the actual
/// addresses (useful when the caller passed port `0`).
///
/// Both listeners share the passed [`ShutdownSignal`] so ctrl_c / SIGTERM
/// terminates them together (graceful).
///
/// This is the discovery-only startup path (no engine registry wired) — it
/// delegates to [`spawn_server_with_service`] with `None`. Callers that build
/// an [`WyomingBackend`] registry use [`spawn_server_with_service`] directly.
pub async fn spawn_server(cfg: Config, signal: ShutdownSignal) -> std::io::Result<ServerHandles> {
    spawn_server_with_service(cfg, signal, None, None).await
}

/// Bind both listeners and spawn their event loops, optionally wiring a
/// Wyoming engine registry + multi-session scheduler (M4-19 T02/T04).
///
/// * `service = Some(_)` → the Wyoming accept loop serves the full ASR + TTS
///   connection handler ([`run_wyoming_connection`]); if `scheduler = Some(_)`
///   each connection first takes a permit + stream slot (overload → explicit
///   `error` event, FR-EX-08).
/// * `service = None` → discovery-only fallback (`describe` → empty `info`), so
///   Home Assistant discovery still succeeds before any model is wired.
///
/// This variant wires the Wyoming side ONLY; the HTTP listener stays
/// health-only. It exists so the M4-19 Wyoming integration tests can drive a
/// mock [`WyomingBackend`] without a real Whisper GGUF. The production path
/// ([`spawn_server_wired`]) wires the HTTP OpenAI / vLLM routers too via
/// [`spawn_server_full`]. Both share the passed [`ShutdownSignal`] so shutdown
/// drains the listeners together.
pub async fn spawn_server_with_service(
    cfg: Config,
    signal: ShutdownSignal,
    service: Option<Arc<dyn WyomingBackend>>,
    scheduler: Option<Arc<Scheduler>>,
) -> std::io::Result<ServerHandles> {
    spawn_server_full(cfg, signal, None, service, scheduler).await
}

/// Bind both listeners and spawn their event loops, wiring BOTH the HTTP API
/// routers (when `http_transcribe = Some`) and the Wyoming engine registry
/// (when `service = Some`) — the full production startup path (M4-19 T04 /
/// M2-09 T06/T09).
///
/// * `http_transcribe = Some(_)` → the HTTP app attaches the OpenAI
///   `/v1/audio/transcriptions` route and the vLLM `/v1/completions` +
///   `/v1/chat/completions` contract routes (see [`build_http_app`]); `None` →
///   health-only.
/// * `service` / `scheduler` behave exactly as in
///   [`spawn_server_with_service`].
///
/// The concrete [`InferenceService`] implements BOTH
/// [`TranscribeService`] (HTTP) and [`WyomingBackend`] (Wyoming), so the
/// production caller passes two trait-object views of the same `Arc`. Kept as
/// separate optionals rather than a bundle so the Wyoming-only test path can
/// leave `http_transcribe = None` without a mock HTTP service.
pub async fn spawn_server_full(
    cfg: Config,
    signal: ShutdownSignal,
    http_transcribe: Option<Arc<dyn TranscribeService>>,
    service: Option<Arc<dyn WyomingBackend>>,
    scheduler: Option<Arc<Scheduler>>,
) -> std::io::Result<ServerHandles> {
    // ---- HTTP ----
    let http_listener = TcpListener::bind(cfg.http_bind).await?;
    let http_actual = http_listener.local_addr()?;
    // FR-EX-08: announce which HTTP surface booted so an operator never
    // wonders why `/v1/audio/*` 404s.
    if http_transcribe.is_some() {
        eprintln!(
            "vokra-server: HTTP serving OpenAI /v1/audio/transcriptions + vLLM \
             /v1/completions (contract-only 501) + /health"
        );
    } else {
        eprintln!(
            "vokra-server: HTTP in health-only mode (no inference registry wired); \
             only /health is answered"
        );
    }
    let app = build_http_app(http_transcribe);
    let http_signal = signal.clone();
    tokio::spawn(async move {
        let shutdown = async move { http_signal.wait().await };
        // axum::serve is the tokio-native builder; with_graceful_shutdown
        // drains in-flight requests before returning.
        let _ = axum::serve(http_listener, app)
            .with_graceful_shutdown(shutdown)
            .await;
    });

    // ---- Wyoming (TCP JSONL) ----
    let wy_listener = TcpListener::bind(cfg.wyoming_bind).await?;
    let wyoming_actual = wy_listener.local_addr()?;
    // FR-EX-08: be explicit about which mode the server booted in so an
    // operator never wonders why ASR "silently" does nothing.
    if service.is_some() {
        eprintln!(
            "vokra-server: wyoming serving full ASR+TTS (multi-session {})",
            if scheduler.is_some() {
                "scheduler wired"
            } else {
                "no scheduler"
            }
        );
    } else {
        eprintln!(
            "vokra-server: wyoming in discovery-only mode (no engine registry wired); \
             only `describe` is answered"
        );
    }
    let wy_signal = signal.clone();
    tokio::spawn(async move {
        wyoming_accept_loop(wy_listener, wy_signal, service, scheduler).await;
    });

    Ok(ServerHandles {
        http_actual,
        wyoming_actual,
    })
}

/// Build the production HTTP router.
///
/// Always serves `/health`. When an [`InferenceService`] transcribe view is
/// supplied it additionally attaches the OpenAI `/v1/audio/transcriptions`
/// route ([`crate::api::openai::attach_routes`]) bound to that service and the
/// vLLM contract router ([`crate::api::vllm::router`]). When it is `None` those
/// routes are OMITTED (they would 404), which is the honest signal that no
/// inference registry was wired — never a route that silently no-ops
/// (FR-EX-08).
///
/// The whole app resolves to `Router<()>`: the OpenAI sub-router is bound to
/// its state with `.with_state(transcribe)` before being merged, exactly like
/// the `attach_routes` unit tests do, and the vLLM router is stateless.
fn build_http_app(http_transcribe: Option<Arc<dyn TranscribeService>>) -> Router {
    let app = Router::new().route("/health", get(health_handler));
    match http_transcribe {
        None => app,
        Some(transcribe) => {
            // `attach_routes::<Arc<dyn TranscribeService>>` picks up the state
            // via axum's blanket `FromRef<S> for S`; `.with_state` then binds
            // it, yielding a `Router<()>` we can merge into the app.
            let openai = crate::api::openai::attach_routes(Router::new()).with_state(transcribe);
            app.merge(openai).merge(crate::api::vllm::router())
        }
    }
}

/// Wire an already-built [`InferenceService`] (or its absence) into the full
/// startup path: derive the HTTP transcribe view, the Wyoming backend view,
/// and a multi-session [`Scheduler`], then bind both listeners.
///
/// * `service = None` → health-only + Wyoming-discovery-only (the M2-09
///   default).
/// * `service = Some(_)` → HTTP OpenAI / vLLM routers + Wyoming ASR / TTS
///   through a [`Scheduler`] capped at [`DEFAULT_MAX_CONCURRENT_SESSIONS`].
async fn spawn_server_wired(
    cfg: Config,
    signal: ShutdownSignal,
    service: Option<Arc<InferenceService>>,
) -> std::io::Result<ServerHandles> {
    let Some(svc) = service else {
        return spawn_server_full(cfg, signal, None, None, None).await;
    };
    // Two trait-object views of the SAME registry `Arc` — the HTTP router
    // needs `TranscribeService`, the Wyoming loop needs `WyomingBackend`, and
    // `InferenceService` implements both.
    let http: Arc<dyn TranscribeService> = svc.clone();
    let wyoming: Arc<dyn WyomingBackend> = svc;
    // `n_stream` must equal `max_concurrent_sessions` (see `Scheduler::new`);
    // both use the same constant here, so the only error path is unreachable
    // in practice — surfaced as a hard startup error regardless (FR-EX-08).
    let scheduler = Scheduler::new(
        SessionRegistryConfig::minimum(DEFAULT_MAX_CONCURRENT_SESSIONS),
        SchedulerConfig::minimum(DEFAULT_MAX_CONCURRENT_SESSIONS),
    )
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;
    spawn_server_full(cfg, signal, Some(http), Some(wyoming), Some(scheduler)).await
}

/// Build the inference registry from the startup [`Config`] model paths, or
/// `Ok(None)` when none are configured (health-only boot).
///
/// Loading is eager: every configured GGUF must open and pass the compliance
/// gate here, so a caller that reaches the listener bind is guaranteed a
/// working registry (FR-EX-08 — no silent partial boot).
fn build_service(cfg: &Config) -> Result<Option<Arc<InferenceService>>, ServiceError> {
    match service_config_from_config(cfg)? {
        None => Ok(None),
        Some(service_cfg) => Ok(Some(InferenceService::build(&service_cfg)?)),
    }
}

/// Map the startup [`Config`] model paths onto a [`ServiceConfig`].
///
/// * No model path set at all → `Ok(None)` (health-only).
/// * At least one set but the required minimum (`whisper_base` AND
///   `piper_plus`) incomplete → `Err(ServiceError::InvalidConfig)`; a
///   half-wired config is a hard error, never a silent partial boot
///   (FR-EX-08).
/// * Required minimum present → `Ok(Some(_))` with every optional slot
///   forwarded. [`ServiceConfig::minimum`] supplies the CPU backend + strict
///   compliance + default watermark settings; the deeper consistency check
///   (large-v3 tokenizer without a large-v3 GGUF) is enforced by
///   [`InferenceService::build`].
fn service_config_from_config(cfg: &Config) -> Result<Option<ServiceConfig>, ServiceError> {
    let any_model = cfg.whisper_base_gguf.is_some()
        || cfg.whisper_base_tokenizer.is_some()
        || cfg.whisper_large_v3_gguf.is_some()
        || cfg.whisper_large_v3_tokenizer.is_some()
        || cfg.piper_plus_gguf.is_some()
        || cfg.kokoro_gguf.is_some()
        || cfg.voxtral_gguf.is_some()
        || cfg.silero_vad_gguf.is_some();
    if !any_model {
        return Ok(None);
    }

    let (Some(whisper_base_gguf), Some(piper_plus_gguf)) =
        (cfg.whisper_base_gguf.clone(), cfg.piper_plus_gguf.clone())
    else {
        return Err(ServiceError::InvalidConfig(
            "model paths were configured but the required minimum is incomplete: \
             both --whisper-base (ASR) and --piper-plus (TTS) are required before \
             the server will serve; refusing to start half-wired (FR-EX-08)"
                .to_string(),
        ));
    };

    let mut service_cfg = ServiceConfig::minimum(whisper_base_gguf, piper_plus_gguf);
    service_cfg.whisper_base_tokenizer = cfg.whisper_base_tokenizer.clone();
    service_cfg.whisper_large_v3_gguf = cfg.whisper_large_v3_gguf.clone();
    service_cfg.whisper_large_v3_tokenizer = cfg.whisper_large_v3_tokenizer.clone();
    service_cfg.kokoro_gguf = cfg.kokoro_gguf.clone();
    service_cfg.voxtral_gguf = cfg.voxtral_gguf.clone();
    service_cfg.silero_vad_gguf = cfg.silero_vad_gguf.clone();
    Ok(Some(service_cfg))
}

/// Convert a startup [`ServiceError`] into the `io::Error` that
/// [`run_with_config`] returns (and `main.rs` maps to exit code 1). The
/// message preserves the failing slot + path + inner cause.
fn startup_io_error(err: ServiceError) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, err.to_string())
}

/// Wyoming Protocol accept loop (M4-19 T02/T04).
///
/// When an engine registry is wired (`service = Some`) each connection is
/// served by the full ASR + TTS handler [`run_wyoming_connection`], optionally
/// through the multi-session [`Scheduler`]. When it is not
/// (`service = None`, the current production startup path — no CLI model
/// wiring yet, M2-09-T04 carry-over) we fall back to the discovery-only
/// handler so Home Assistant's `describe` probe still completes cleanly
/// (without this, the historical accept-and-drop stub made even wire-level
/// discovery fail — see `integrations/vokra-server/tests/wyoming-ha-smoke.md`).
///
/// Each connection runs on its own [`spawn_isolated_wyoming_task`] so a
/// panicked JSONL parser closes ONE connection, never the listener
/// (NFR-RL-07).
async fn wyoming_accept_loop(
    listener: TcpListener,
    signal: ShutdownSignal,
    service: Option<Arc<dyn WyomingBackend>>,
    scheduler: Option<Arc<Scheduler>>,
) {
    loop {
        tokio::select! {
            _ = signal.clone().wait() => break,
            accept = listener.accept() => {
                match accept {
                    Ok((stream, peer)) => {
                        let service = service.clone();
                        let scheduler = scheduler.clone();
                        spawn_isolated_wyoming_task(peer, async move {
                            serve_wyoming_connection(stream, peer, service, scheduler).await;
                        });
                    }
                    Err(err) => {
                        eprintln!("vokra-server: wyoming accept error: {err}");
                    }
                }
            }
        }
    }
}

/// Serve one accepted Wyoming connection (M4-19 T02/T04).
///
/// With a wired `service` we optionally take a multi-session slot from
/// `scheduler` (permit + `StreamSlot`, released by RAII when this function
/// returns), then run the full ASR + TTS handler. Overload is surfaced as an
/// explicit `error` event mapped through
/// [`SchedulerError::to_server_error`](crate::scheduler::SchedulerError::to_server_error)
/// — never a silent connection drop (FR-EX-08). Without a service we run the
/// discovery-only fallback.
async fn serve_wyoming_connection(
    stream: tokio::net::TcpStream,
    peer: SocketAddr,
    service: Option<Arc<dyn WyomingBackend>>,
    scheduler: Option<Arc<Scheduler>>,
) {
    let (reader, mut writer) = stream.into_split();
    let Some(service) = service else {
        // Discovery-only fallback.
        if let Err(err) = run_describe_only_connection(reader, &mut writer).await {
            eprintln!("vokra-server: wyoming session with {peer} ended: {err}");
        }
        return;
    };

    // T04: hold a scheduler session (permit + stream slot) for the lifetime of
    // this connection. `_session` releases both via RAII on drop.
    let _session = match scheduler {
        Some(sched) => match sched.acquire_or_503().await {
            Ok(sess) => Some(sess),
            Err(e) => {
                let msg = format!("wyoming session refused: {}", e.to_server_error().message());
                let _ = write_wyoming_error(&mut writer, &msg).await;
                return;
            }
        },
        None => None,
    };

    if let Err(err) = run_wyoming_connection(reader, &mut writer, service, BargeIn::new()).await {
        eprintln!("vokra-server: wyoming session with {peer} ended: {err}");
    }
}

/// `GET /health` — returns `200 OK` with a fixed body. No dependencies on
/// the engine registry (added T04); a bare health probe must succeed as
/// soon as the listener is up so container / systemd readiness probes work.
async fn health_handler() -> (StatusCode, &'static str) {
    (StatusCode::OK, "ok")
}

/// Compose a shutdown-driven variant that returns handles immediately, so
/// tests can trigger shutdown after probing `/health`. Not used by `main`.
/// Discovery-only (no engine registry) — the M2-09 default.
pub async fn spawn_server_for_test(
    cfg: Config,
) -> std::io::Result<(ServerHandles, ShutdownTrigger)> {
    let (signal, trigger) = install_shutdown_signal();
    let handles = spawn_server(cfg, signal).await?;
    Ok((handles, trigger))
}

/// Like [`spawn_server_for_test`] but wires a Wyoming engine registry +
/// optional scheduler (M4-19 T04). Integration tests use this to drive the
/// full ASR + TTS + barge-in protocol path over TCP loopback.
pub async fn spawn_server_for_test_with_service(
    cfg: Config,
    service: Option<Arc<dyn WyomingBackend>>,
    scheduler: Option<Arc<Scheduler>>,
) -> std::io::Result<(ServerHandles, ShutdownTrigger)> {
    let (signal, trigger) = install_shutdown_signal();
    let handles = spawn_server_with_service(cfg, signal, service, scheduler).await?;
    Ok((handles, trigger))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Binding to port 0 must succeed and both listeners must expose a
    /// port. This is the T03 acceptance test: two listeners on one runtime,
    /// no silent failure, `/health` reachable.
    #[tokio::test]
    async fn startup_binds_http_and_wyoming_on_random_ports() {
        let cfg = Config {
            http_bind: "127.0.0.1:0".parse().unwrap(),
            wyoming_bind: "127.0.0.1:0".parse().unwrap(),
            config_file: None,
            ..Config::default()
        };
        let (handles, trigger) = spawn_server_for_test(cfg).await.expect("bind");
        assert_ne!(
            handles.http_actual.port(),
            0,
            "OS must have assigned an HTTP port"
        );
        assert_ne!(
            handles.wyoming_actual.port(),
            0,
            "OS must have assigned a Wyoming port"
        );
        assert!(handles.http_actual.ip().is_loopback());
        assert!(handles.wyoming_actual.ip().is_loopback());

        // Probe /health via raw TCP HTTP/1.1 (no reqwest dep here — keep the
        // unit test self-contained). We just need to see "HTTP/1.1 200".
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut sock = tokio::net::TcpStream::connect(handles.http_actual)
            .await
            .expect("connect /health");
        sock.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut buf = Vec::new();
        // Bounded read: the response is tiny (< 200 bytes) and the server
        // closes the connection so EOF terminates the read.
        tokio::time::timeout(Duration::from_secs(2), sock.read_to_end(&mut buf))
            .await
            .expect("health read timeout")
            .expect("health read");
        let head = std::str::from_utf8(&buf[..buf.len().min(64)]).unwrap_or_default();
        assert!(
            head.starts_with("HTTP/1.1 200"),
            "expected 200 OK, got: {head:?}"
        );

        // Verify Wyoming accepts a TCP connection (T03 closes it).
        let _ = tokio::net::TcpStream::connect(handles.wyoming_actual)
            .await
            .expect("wyoming accept");

        // Graceful shutdown must complete promptly.
        trigger.trigger();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // ---------------------------------------------------------------------
    // T04 service wiring: Config → ServiceConfig mapping.
    // ---------------------------------------------------------------------

    #[test]
    fn wiring_no_models_maps_to_health_only() {
        let cfg = Config::default();
        let mapped = service_config_from_config(&cfg).expect("no models must be Ok");
        assert!(mapped.is_none(), "no model paths ⇒ health-only (None)");
        // build_service agrees: nothing to load, no engine.
        assert!(build_service(&cfg).expect("ok").is_none());
    }

    #[test]
    fn wiring_partial_model_config_is_hard_error() {
        // whisper-base without piper-plus is a half-wired config → refuse to
        // start rather than boot an ASR-only, TTS-404 server (FR-EX-08).
        let cfg = Config {
            whisper_base_gguf: Some("/models/whisper-base.gguf".into()),
            ..Config::default()
        };
        let err = service_config_from_config(&cfg).expect_err("partial config must error");
        assert!(matches!(err, ServiceError::InvalidConfig(_)), "got {err}");

        // Symmetric: piper without whisper is equally rejected.
        let cfg = Config {
            piper_plus_gguf: Some("/models/piper.gguf".into()),
            ..Config::default()
        };
        assert!(matches!(
            service_config_from_config(&cfg).expect_err("partial config must error"),
            ServiceError::InvalidConfig(_)
        ));
    }

    #[test]
    fn wiring_complete_config_forwards_optional_paths() {
        use std::path::Path;
        let cfg = Config {
            whisper_base_gguf: Some("/m/base.gguf".into()),
            piper_plus_gguf: Some("/m/piper.gguf".into()),
            kokoro_gguf: Some("/m/kokoro.gguf".into()),
            voxtral_gguf: Some("/m/voxtral.gguf".into()),
            silero_vad_gguf: Some("/m/vad.gguf".into()),
            ..Config::default()
        };
        let sc = service_config_from_config(&cfg).expect("ok").expect("some");
        assert_eq!(sc.whisper_base_gguf.as_path(), Path::new("/m/base.gguf"));
        assert_eq!(sc.piper_plus_gguf.as_path(), Path::new("/m/piper.gguf"));
        assert_eq!(sc.kokoro_gguf.as_deref(), Some(Path::new("/m/kokoro.gguf")));
        assert_eq!(
            sc.voxtral_gguf.as_deref(),
            Some(Path::new("/m/voxtral.gguf"))
        );
        assert_eq!(
            sc.silero_vad_gguf.as_deref(),
            Some(Path::new("/m/vad.gguf"))
        );
    }

    #[test]
    fn wiring_missing_gguf_is_hard_startup_error() {
        // FR-EX-08: the required minimum is present but the files do not
        // exist, so the eager loader fails and `build_service` must NOT
        // return Ok. No real weights are needed — the failure IS the point.
        let cfg = Config {
            whisper_base_gguf: Some("/nonexistent/vokra-whisper-base.gguf".into()),
            piper_plus_gguf: Some("/nonexistent/vokra-piper.gguf".into()),
            ..Config::default()
        };
        // `Arc<InferenceService>` is not `Debug`, so match rather than
        // `expect_err` (which would require the Ok type to be `Debug`).
        let err = match build_service(&cfg) {
            Ok(_) => panic!("missing GGUF must be a hard startup error, got Ok"),
            Err(e) => e,
        };
        assert!(
            matches!(err, ServiceError::ModelLoadFailed { .. }),
            "expected ModelLoadFailed, got {err}"
        );
    }

    // ---------------------------------------------------------------------
    // T06/T09 HTTP router wiring: build_http_app mounts / omits the API
    // routes based on whether an inference service is present. A fake
    // TranscribeService stands in for a real engine (no weights).
    // ---------------------------------------------------------------------

    struct FakeTranscribe;
    impl TranscribeService for FakeTranscribe {
        fn transcribe(&self, _model: &str, _pcm: &[f32]) -> Result<String, ServiceError> {
            Ok("mock-transcript".to_string())
        }
    }

    /// Minimal 16 kHz mono PCM16 WAV so `decode_pcm_wav` accepts the upload
    /// and the request reaches the injected service. Mirrors the openai.rs
    /// unit-test fixture; kept local rather than reaching across the module
    /// boundary into another module's `#[cfg(test)]` helpers.
    fn tiny_wav(num_samples: usize) -> Vec<u8> {
        let sample_rate: u32 = 16_000;
        let block_align: u16 = 2; // mono, 16-bit
        let data_bytes = (num_samples as u32) * u32::from(block_align);
        let mut buf = Vec::with_capacity(44 + data_bytes as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // channels
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_bytes.to_le_bytes());
        for i in 0..num_samples {
            let s = ((i as i32 % 32) - 16) as i16 * 512;
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf
    }

    /// Build an OpenAI-shaped `multipart/form-data` body (file + model +
    /// response_format) built byte-by-byte to avoid string-continuation
    /// pitfalls.
    fn multipart_wav_body(boundary: &str, wav: &[u8]) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
        body.extend_from_slice(wav);
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
        body.extend_from_slice(b"whisper-1\r\n");
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"response_format\"\r\n\r\n");
        body.extend_from_slice(b"json\r\n");
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        body
    }

    #[tokio::test]
    async fn http_app_routes_transcription_to_injected_service() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let app = build_http_app(Some(Arc::new(FakeTranscribe) as Arc<dyn TranscribeService>));
        let boundary = "vokra-server-wiring-boundary";
        let body = multipart_wav_body(boundary, &tiny_wav(3200));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "OpenAI route must be mounted and reach the injected service"
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            text.contains("mock-transcript"),
            "response must carry the injected service's output, got {text}"
        );
    }

    #[tokio::test]
    async fn http_app_mounts_vllm_contract_route_when_service_wired() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let app = build_http_app(Some(Arc::new(FakeTranscribe) as Arc<dyn TranscribeService>));
        // A well-formed completions request → 501 NotImplemented (v0.5
        // contract-only), proving the vLLM router is mounted (never a 404).
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"x","prompt":"y"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::NOT_IMPLEMENTED,
            "vLLM /v1/completions must be mounted and answer 501 (contract-only)"
        );
    }

    #[tokio::test]
    async fn http_app_omits_api_routes_without_service() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let app = build_http_app(None);
        // /health is always present.
        let health = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);

        // Without a service the API routes are OMITTED — an honest 404, never
        // a silently no-op route (FR-EX-08).
        let asr = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/transcriptions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(asr.status(), StatusCode::NOT_FOUND);

        let vllm = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(vllm.status(), StatusCode::NOT_FOUND);
    }
}
