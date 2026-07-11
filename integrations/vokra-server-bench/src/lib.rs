//! # vokra-server-bench — HTTP-boundary TTS latency benchmark client
//!
//! Excluded-workspace binary + library crate that measures real
//! HTTP `POST /api/tts` round-trip latency against a running
//! [`vokra-server`](../vokra-server) process.
//!
//! ## Why this exists
//!
//! `docs/m3-15-server-latency-handover.md` § 4 offers three paths for
//! NFR-PF-05 (75 ms TTFA) reference measurement:
//!
//! * **Option A** — in-process bench
//!   (`integrations/vokra-server/benches/tts_latency.rs`); measures the
//!   schema-layer *floor* with a deterministic `FakeSynth`. Cannot see
//!   the wire.
//! * **Option B** — `curl` + `awk` shell loop; measures the wire but is
//!   fragile (percentile quality depends on the operator's shell math)
//!   and does not exercise sustained concurrency.
//! * **Option C** — a Rust binary that speaks HTTP and reports p50 /
//!   p95 / p99 across N concurrent workers. This crate IS Option C.
//!
//! ## Boundary being measured
//!
//! Per NFR-PF-05, TTFA (Time To First Audio byte) is the wall time
//! between the client finishing the request write and receiving the
//! first byte of the audio response. ureq's `.send_bytes(...)` returns
//! a `Response` the moment status + headers are fully read from the
//! socket, which — for `POST /api/tts` where the server produces the
//! WAV up-front (see `integrations/vokra-server/src/service.rs` L957
//! "single-shot response body") — is the wire event that starts the
//! audio byte stream. We therefore measure:
//!
//! * `ttfa_ms`  = start → `.send_bytes(...)` returns (status + headers
//!   received);
//! * `total_ms` = start → response body fully drained.
//!
//! For the current non-streaming vocoder the two numbers are within one
//! `read_to_end` of each other. Both are captured so the schema stays
//! stable once a real streaming vocoder lands (M3-05 `flow_sampler` +
//! server-side `istft_streaming` generator).
//!
//! ## What the tool decides
//!
//! Deliberately **nothing** on the pass/fail axis. Per the M2-14 /
//! M3-15 handover conventions, this bench emits reference numbers and
//! a computed `verdict` string against `--budget-ms`; the quarterly
//! Go/No-go review (NFR-MT-05) reads the artifact and decides. The
//! process exits `0` on any successful measurement window (including
//! `verdict=FAIL`) so a slower-than-reference CI host does not block
//! main PRs. Non-zero exits are reserved for setup / transport failures
//! — see [`ExitCode`].
//!
//! ## FR-EX-08 (no silent fallback)
//!
//! * Unknown / malformed CLI flags → [`ExitCode::BadArgs`] (2).
//! * All requests fail with a transport error (server unreachable) →
//!   [`ExitCode::AllTransportFailed`] (3).
//! * `--budget-ms` is echoed into the artifact untouched; the bench
//!   never invents a fallback threshold.
//!
//! ## Zero-dep posture (NFR-DS-02)
//!
//! This is an excluded-workspace crate (see the top of `Cargo.toml`
//! for the `[workspace]` isolation). It links `ureq` + `serde_json`;
//! the root `Cargo.lock` is untouched. `scripts/check-zero-deps.sh`
//! asserts on the root graph only, so this bench does not participate
//! in the invariant.

// The crate is `#![forbid(unsafe_code)]` via the workspace-lint at the
// top of `Cargo.toml`. Every module below stays safe Rust.

pub mod bench;
pub mod cli;
pub mod stats;

// Public re-exports for the integration tests and `main.rs`.
pub use bench::{Timing, TransportOutcome, run_bench};
pub use cli::{Args, ExitCode, OutputFormat, ParseError, parse_args};
pub use stats::{Summary, emit_json, emit_kv, summarize};
