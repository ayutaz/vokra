//! # vokra-cli-bench-server — zero-dep HTTP-boundary TTS/ASR latency bench
//!
//! Excluded-workspace binary + library that measures real HTTP `POST` round-trip
//! latency against a running [`vokra-server`](../vokra-server) process using
//! **only** the Rust standard library — no `ureq`, no `serde_json`, no `hyper`.
//!
//! ## Why this exists
//!
//! `docs/m3-15-server-latency-handover.md` § 4 offers three paths for
//! NFR-PF-05 (75 ms TTFA) reference measurement:
//!
//! * **Option A** — in-process bench
//!   (`integrations/vokra-server/benches/tts_latency.rs`); FakeSynth floor,
//!   cannot see the wire.
//! * **Option B** — `curl` + `awk` shell loop; wire but fragile percentiles.
//! * **Option C** — a Rust binary that reports p50 / p95 / p99. Two crates
//!   fill this: `integrations/vokra-server-bench/` (`ureq` + `serde_json`)
//!   and THIS crate (pure-std). Both emit the same JSON schema; pick the
//!   one whose Cargo.lock posture matches your review constraint.
//!
//! ## Boundary being measured
//!
//! Per NFR-PF-05, TTFA (Time To First Audio byte) is the wall time between
//! the client finishing the request write and receiving the first byte of
//! the audio response. Because HTTP/1.1 headers arrive as a single burst
//! *before* the body, and because
//! [`crate::http::send_request`] returns the moment the `\r\n\r\n`
//! terminator is received, we capture:
//!
//! * `ttfa`  = `Instant::now()` before writing the request → `\r\n\r\n`
//!   header terminator received;
//! * `total` = `Instant::now()` before writing the request → body fully
//!   drained (either up to `Content-Length` bytes or EOF for
//!   `Connection: close`).
//!
//! For the current non-streaming vocoder (see
//! `integrations/vokra-server/src/service.rs`, "single-shot response
//! body") the two numbers are within one `read_to_end` of each other.
//! Both are captured so the schema stays stable once a real streaming
//! vocoder lands.
//!
//! ## FR-EX-08 (no silent fallback)
//!
//! * Unknown / malformed CLI flags → [`cli::ExitCode::BadArgs`] (2).
//! * All requests fail with a transport error → [`cli::ExitCode::AllTransportFailed`]
//!   (3). Never "gracefully continued with no data".
//! * `https://` is deliberately rejected at URL-parse time (pure-std has
//!   no TLS) — see [`http::parse_target`].
//!
//! ## Zero-dep posture (NFR-DS-02)
//!
//! Every module below imports only `std::*`. The excluded-workspace
//! `Cargo.toml` has an empty `[dependencies]` table.

// Every module is `#![forbid(unsafe_code)]` via the workspace lint
// (see `Cargo.toml`). No module uses `unsafe`.

pub mod bench;
pub mod cli;
pub mod http;
pub mod stats;

// Public re-exports for the integration tests and `main.rs`.
pub use bench::{Timing, TransportOutcome, run_bench};
pub use cli::{Args, ExitCode, OutputFormat, ParseError, parse_args};
pub use http::{HttpTarget, parse_target, send_request};
pub use stats::{PercentileBundle, Summary, emit_json, emit_kv, summarize, verdict};
