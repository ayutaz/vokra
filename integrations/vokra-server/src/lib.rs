//! `vokra-server` — Vokra single-binary API server (isolated integration crate).
//!
//! This crate hosts the OpenAI-compatible / vLLM-compatible / piper-plus HTTP
//! endpoints and the Wyoming Protocol TCP listener. It is deliberately kept
//! OUT of the Vokra root workspace (see `Cargo.toml`) so that its HTTP /
//! async / serde dependencies never touch the root `Cargo.lock`
//! (NFR-DS-02, enforced by `scripts/check-zero-deps.sh`).
//!
//! # Scope (FR-SV-01..05, milestones.md §6 M2-09)
//!
//! Full FR-SV-01..05 scope, out-of-scope items, and cross-crate invariants
//! (FR-EX-08 / NFR-RL-01 / NFR-RL-07 / NFR-LC-04 / NFR-DS-02) are documented
//! at the top of `main.rs`, which is this crate's single source of truth
//! for the M2-09 scope-boundary doc. This file exposes the T03 runtime
//! primitives so they can be reused from the binary AND from integration
//! tests.
//!
//! # T03 surface (this change)
//!
//! * [`config`] — CLI + env + optional TOML parsing into [`Config`].
//! * [`server`] — HTTP axum listener (default `127.0.0.1:8080`) + Wyoming
//!   TCP listener (default `127.0.0.1:10300`) on ONE tokio runtime, plus
//!   the `/health` endpoint that returns `200 OK`.
//! * [`shutdown`] — ctrl_c + SIGTERM watcher; both listeners drain
//!   together (NFR-RL-07).
//! * [`enforce_c_numeric_locale`] — pins `LC_NUMERIC=C` at boot
//!   (NFR-RL-01) before any tokio worker thread starts.
//!
//! Modules `api`, `error`, `service` are the T01 skeleton and stay
//! declared (but unused at T03) so T04..T17 add symbols without editing
//! the module list.

// NOTE: no `#![forbid(unsafe_code)]` at the crate root because
// `enforce_c_numeric_locale` needs `unsafe { std::env::set_var(...) }`
// on edition 2024. Every `unsafe` block below carries a `// SAFETY:` note.

// T01 skeleton (empty modules; filled in T04..T17).
pub mod api;
pub mod error;
pub mod service;

// T03 additions — the startup foundation itself.
pub mod config;
pub mod server;
pub mod shutdown;

// M3-15 multi-session support.
pub mod latency;
pub mod scheduler;
pub mod session;

pub use api::wyoming::{BargeIn, WyomingBackend};
pub use config::{Config, ConfigError, HELP_TEXT, parse_args};
pub use latency::{LatencyRecorder, LatencyReport};
pub use scheduler::{Scheduler, SchedulerConfig, SchedulerError, SchedulerSession};
pub use server::{
    ServerHandles, run_with_config, spawn_server, spawn_server_for_test,
    spawn_server_for_test_wired, spawn_server_for_test_with_service, spawn_server_full,
    spawn_server_with_service,
};
pub use session::{
    RegistryError, ServerSession, SessionGuard, SessionId, SessionRegistry, SessionRegistryConfig,
    StreamSlot,
};
pub use shutdown::{ShutdownSignal, ShutdownTrigger, install_shutdown_signal};

/// Pin `LC_NUMERIC=C` (and `LC_ALL=C`) BEFORE the tokio runtime spawns
/// worker threads, so any C library reached transitively (e.g. a JSON
/// parser using `strtod`) treats `.` as the decimal separator regardless
/// of the process's ambient locale. This is the NFR-RL-01
/// "European-locale `strtod` crash" mitigation called out in CLAUDE.md.
///
/// Must be called BEFORE any thread that might read the environment is
/// spawned (single-threaded precondition).
pub fn enforce_c_numeric_locale() {
    // SAFETY: `std::env::set_var` is `unsafe` in edition 2024 because
    // it is not safe to call from multiple threads (or concurrently
    // with a reader). The documented precondition of this function is
    // that the caller invokes it BEFORE any tokio worker thread is
    // built (see `server::run_with_config`, which calls it before
    // `tokio::runtime::Builder::new_multi_thread().build()`), so no
    // other thread can be reading the environment concurrently.
    unsafe { std::env::set_var("LC_NUMERIC", "C") };
    // SAFETY: same precondition — single-threaded startup path.
    unsafe { std::env::set_var("LC_ALL", "C") };
}

#[cfg(test)]
mod lc_numeric_tests {
    use super::*;

    #[test]
    fn startup_lc_numeric_is_pinned_to_c() {
        enforce_c_numeric_locale();
        assert_eq!(std::env::var("LC_NUMERIC").as_deref(), Ok("C"));
        assert_eq!(std::env::var("LC_ALL").as_deref(), Ok("C"));
    }
}

/// T20 smoke tests for the security/ops posture documented in
/// `docs/security-ops.md`. This module is intentionally named `security`
/// so the completion command from that document —
/// `cargo test security::default_bind_loopback` — matches exactly.
///
/// A single anchor test guards against silent widening of the default
/// bind. Any future edit that drops the loopback default MUST update
/// this test AND `docs/security-ops.md` in the same PR.
#[cfg(test)]
mod security {
    use super::*;

    /// Clear the `VOKRA_*_BIND` env vars so `parse_args` observes a
    /// clean baseline. Uses the same SAFETY reasoning as `config.rs`'s
    /// `clear_env` (single-threaded startup path in `cargo test`).
    fn clear_bind_env() {
        // SAFETY: only this test touches VOKRA_* env keys, and it does
        // so before any observer thread reads them (single-threaded
        // startup — cargo test spawns one thread per #[test]).
        unsafe {
            std::env::remove_var("VOKRA_HTTP_BIND");
            std::env::remove_var("VOKRA_WYOMING_BIND");
            std::env::remove_var("VOKRA_CONFIG");
        }
    }

    /// The default bind for BOTH listeners must be loopback (127.0.0.1)
    /// and never `0.0.0.0`. External exposure requires an explicit
    /// `--http-bind 0.0.0.0:<port>` — see `docs/security-ops.md` §1.
    #[test]
    fn default_bind_loopback() {
        clear_bind_env();
        let cfg = parse_args(["vokra-server"]).expect("defaults parse");

        // HTTP listener defaults to loopback.
        assert!(
            cfg.http_bind.ip().is_loopback(),
            "HTTP default must be loopback, got {}",
            cfg.http_bind
        );
        // Guard against IPv6 dual-stack quirks or an accidental widening
        // that happens to still satisfy is_loopback() but isn't 127.0.0.1.
        assert_eq!(
            cfg.http_bind.to_string(),
            "127.0.0.1:8080",
            "HTTP default must be exactly 127.0.0.1:8080"
        );

        // Wyoming listener defaults to loopback.
        assert!(
            cfg.wyoming_bind.ip().is_loopback(),
            "Wyoming default must be loopback, got {}",
            cfg.wyoming_bind
        );
        assert_eq!(
            cfg.wyoming_bind.to_string(),
            "127.0.0.1:10300",
            "Wyoming default must be exactly 127.0.0.1:10300"
        );
    }
}
