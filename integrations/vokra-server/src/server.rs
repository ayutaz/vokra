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

use crate::api::wyoming::{
    BargeIn, WyomingBackend, run_describe_only_connection, run_wyoming_connection,
    write_wyoming_error,
};
use crate::config::Config;
use crate::error::spawn_isolated_wyoming_task;
use crate::scheduler::Scheduler;
use crate::shutdown::{ShutdownSignal, ShutdownTrigger, install_shutdown_signal};
use axum::Router;
use axum::http::StatusCode;
use axum::routing::get;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

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

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let (signal, _trigger) = install_shutdown_signal();
        let handles = spawn_server(cfg, signal.clone()).await?;
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
/// The two share the passed [`ShutdownSignal`] so shutdown drains them
/// together.
pub async fn spawn_server_with_service(
    cfg: Config,
    signal: ShutdownSignal,
    service: Option<Arc<dyn WyomingBackend>>,
    scheduler: Option<Arc<Scheduler>>,
) -> std::io::Result<ServerHandles> {
    // ---- HTTP ----
    let http_listener = TcpListener::bind(cfg.http_bind).await?;
    let http_actual = http_listener.local_addr()?;
    let http_signal = signal.clone();
    tokio::spawn(async move {
        // Bare-minimum router: /health only. T06/T09/T11 hang the API
        // routers off `app`.
        let app: Router = Router::new().route("/health", get(health_handler));
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
}
