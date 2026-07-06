//! `vokra-server` — single-binary API-compatible server (M2-09).
//!
//! T03: parses CLI/env/optional TOML into a [`Config`], enforces
//! `LC_NUMERIC=C`, spawns HTTP + Wyoming listeners on a shared tokio
//! runtime, and waits for ctrl_c / SIGTERM before draining. This file is
//! the single source of truth for the crate's scope-boundary doc; the
//! library root (`lib.rs`) re-states only the T03 surface.
//!
//! # Scope (FR-SV-01..05, milestones.md §6 M2-09)
//!
//! * **FR-SV-01** Single-binary server, Docker not required. Distributed as a
//!   musl-static `x86_64-unknown-linux-musl` release plus normal macOS /
//!   Windows builds. Verified by CI in T19.
//! * **FR-SV-02** OpenAI `/v1/audio/transcriptions`-compatible endpoint
//!   fronting the native Whisper base + large-v3 engines (M1 ASR + M2-06).
//!   Response schema is a drop-in for faster-whisper.
//! * **FR-SV-03** vLLM `/v1/completions` + `/v1/chat/completions` — CONTRACT
//!   FIRST ONLY. v0.5 has no LLM in-tree (FR-MD-04 is Whisper large-v3,
//!   FR-MD-05 is Kokoro TTS). Requests return either 501 Not Implemented or
//!   a schema-valid placeholder; we NEVER fabricate generation text
//!   (NFR-RL-06). Real LLM inference is v1.0+ (Voxtral / CosyVoice2 LLM
//!   path) and v1.5+ (Moshi / Helium).
//! * **FR-SV-04** piper-plus `/api/tts`-compatible endpoint driving Vokra's
//!   native MB-iSTFT-VITS2 implementation (M0-07). eSpeak-NG is NEVER linked
//!   (GPL-3.0). G2P comes from the isolated `integrations/vokra-piper-g2p`
//!   crate across the `vokra_piper_plus::Phonemizer` trait boundary.
//! * **FR-SV-05** Wyoming Protocol JSONL/TCP server so Home Assistant Voice
//!   can use Vokra as a Wyoming-compatible ASR + TTS backend. Actual HA
//!   adoption is a Kill-switch-J signal reviewed quarterly in M2-15 (owner
//!   call, not automated).
//!
//! # Explicitly OUT OF SCOPE for M2-09
//!
//! * **Numeric kernels of any kind.** This crate does not implement STFT /
//!   GEMM / attention / vocoder / anything sample-shaped. It ONLY calls the
//!   existing `vokra-core` / `vokra-models` / `vokra-piper-plus` engines via
//!   their Rust API (`Arc<Engine>`) — never the C ABI, never a re-impl.
//! * **Multi-session serving (FR-SV-06).** v0.5 selects ONE model per
//!   request; concurrent multi-model orchestration lands in v1.0.
//! * **Real LLM inference** (see FR-SV-03).
//! * **Watermark / C2PA embedding.** Owner dropped these from M1 on
//!   2026-07-04. The registry accepts `WatermarkConfig` as a forward-compat
//!   hook only; nothing is embedded in TTS output at v0.5.
//! * **ONNX / protobuf / gRPC of any flavour** (FR-LD-05, NFR-DS-02). The
//!   runtime is native Rust only.
//! * **VC / voice cloning endpoints.** Those live in the separate
//!   `vokra-voiceclone-experimental` repo (ELVIS Act + NO FAKES Act
//!   distributor-liability boundary — see CLAUDE.md §8).
//!
//! # Invariants this crate must preserve
//!
//! * Root `Cargo.lock` stays `vokra-*`-only. This crate is an excluded
//!   workspace with its own `Cargo.lock`; `scripts/check-zero-deps.sh` scans
//!   the root lockfile and MUST keep passing (NFR-DS-02).
//! * **FR-EX-08:** no silent CPU fallback. If a backend (Metal / CUDA) does
//!   not implement a requested op, the engine returns `UnsupportedOp` and we
//!   surface it as an HTTP 501 with `type: "unsupported_op"`. We do NOT
//!   rewrite the request to CPU behind the caller's back.
//! * **NFR-RL-07:** API-boundary safety. Every handler is wrapped so a panic
//!   inside one request becomes a 500 JSON error, never a runtime crash.
//!   Wyoming TCP sessions are equally isolated per tokio task.
//! * **NFR-RL-01:** LC_NUMERIC-safe. Startup pins `LC_NUMERIC=C` and we do
//!   not call `strtod` directly anywhere.
//! * **NFR-LC-04:** cargo-deny gate. No GPL / LGPL deps (all MIT /
//!   Apache-2.0 — see `docs/adr-http-stack.md`).

#![forbid(unsafe_code)]

use std::process::ExitCode;

use vokra_server::{Config, ConfigError, HELP_TEXT, parse_args, run_with_config};

fn main() -> ExitCode {
    match parse_args(std::env::args()) {
        Ok(cfg) => run(cfg),
        Err(ConfigError::HelpRequested) => {
            println!("{HELP_TEXT}");
            ExitCode::SUCCESS
        }
        Err(ConfigError::VersionRequested) => {
            println!("vokra-server {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Err(err) => {
            // Unknown flag / bad value / bad TOML — hard startup error.
            // Exit 2 is the conventional "misuse of shell command" code.
            eprintln!("vokra-server: {err}");
            eprintln!("try `vokra-server --help`");
            ExitCode::from(2)
        }
    }
}

fn run(cfg: Config) -> ExitCode {
    match run_with_config(cfg) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("vokra-server: fatal: {err}");
            ExitCode::from(1)
        }
    }
}
