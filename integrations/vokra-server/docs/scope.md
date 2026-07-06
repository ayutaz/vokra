# vokra-server — Scope Boundary (M2-09-T01 kickoff)

This document nails down the scope of the `vokra-server` crate at the
WP kickoff (T01), so subsequent tickets can be graded against a fixed
target. Source of truth for the WP is `docs/milestones.md` §6 M2-09 and
the M2-09 ticket spec (`docs/tickets/m2/M2-09-vokra-server`).

## What this crate delivers (FR-SV-01..05, NFR-PF-05)

1. A **single-binary** server (Docker NOT required — FR-SV-01) that
   exposes four API surfaces on top of Vokra's native engines:
   - **OpenAI** `/v1/audio/transcriptions` (FR-SV-02, faster-whisper
     drop-in schema for Whisper base + large-v3).
   - **vLLM** `/v1/completions` + `/v1/chat/completions` (FR-SV-03,
     contract-first — see §"Not in scope" below).
   - **piper-plus** `/api/tts` (FR-SV-04, same request/response schema
     as the upstream piper-plus HTTP server, backed by Vokra's native
     MB-iSTFT-VITS2 implementation and the Kokoro registry).
   - **Wyoming Protocol** (FR-SV-05, JSONL over TCP for Home Assistant
     integration — implementation lives here; hardware bring-up lives
     in M2-15).
2. **Server TTS latency target: ~90 ms** on the reference desktop
   (NFR-PF-05); measurement harness lands in T18. The reference
   measurement boundary and hardware are pinned in T18 and documented
   there.
3. **Static musl build** for `x86_64-unknown-linux-musl` so a single
   binary drop suffices in production (T19 CI verifies).

## What this crate does NOT do

- **No numerical kernels of its own.** All ASR/TTS/VAD math runs inside
  the existing engines (`WhisperAsr`, `PiperPlusTts`, `KokoroTts`,
  `SileroVadV5`) reached through their Rust API (D5 in the plan). The
  server layer only wires request → engine → response.
- **No silent CPU fallback (FR-EX-08).** If a backend (`Metal` / `Cuda`)
  returns `UnsupportedOp` for a given op, the server surfaces it as an
  explicit HTTP error (T05 error mapping → 501 + `type:"unsupported_op"`).
  It never silently reroutes to CPU. Clients may opt in to fallback via
  an explicit request header in a future ticket, but the default is
  fail-explicit.
- **No real LLM inference in v0.5.** The vLLM endpoints are
  **contract-first**: request/response schemas parse cleanly, but the
  server MUST NOT fabricate generations (NFR-RL-06). Behaviour is
  either 501 Not Implemented with an explicit error body or a
  clearly-marked placeholder string; final choice is made in T09.
- **No multi-session serving (FR-SV-06).** v0.5 is per-request single
  model selection. FR-SV-06 is a v1.0 concern.
- **No watermark / C2PA embedding.** Dropped by the client on
  2026-07-04; this crate holds forward-compat hooks only, wired into
  the registry (T21) for future re-enablement in M2-13.
- **No Home Assistant hardware bring-up.** M2-15 (four-quarter Go/No-go
  review) owns the real HA integration and the Kill switch J decision.
  This crate ships the Wyoming server + docs (T15..T17, T23).
- **No voice-cloning endpoints (RVC / GPT-SoVITS / speaker-only VC).**
  Those live in the separate `vokra-voiceclone-experimental` repo under
  ELVIS Act / NO FAKES Act isolation (see CLAUDE.md item 8).
- **No ONNX / protobuf / gRPC dependency.** The runtime is `vokra-*`
  only; the server layer adds `tokio` + `axum` + `hyper` + `serde` +
  `tower` (all MIT) confined to this crate's own `Cargo.lock`. Root
  `Cargo.lock` is not observed by this crate's graph.

## Why this crate lives OUT-OF-WORKSPACE

Same pattern as `integrations/vokra-piper-g2p/`:

- Root `Cargo.toml` sets `members = ["crates/*", "tests/parity"]` and
  `exclude = ["integrations"]`. The glob does not include
  `integrations/`, so this crate is not a member of the root workspace.
- The empty `[workspace]` table in this crate's `Cargo.toml` makes this
  directory its own workspace root. Cargo emits a separate
  `Cargo.lock` here.
- `scripts/check-zero-deps.sh` scans **only** the root `Cargo.lock` for
  non-`vokra-*` entries, so anything this crate pulls in is invisible
  to the zero-dep invariant check (NFR-DS-02).
- The path deps to `vokra-core` / `vokra-models` / `vokra-piper-plus`
  use `path = "../../crates/..."` (not `workspace = true`, which only
  works within the owning workspace). Those runtime crates stay
  zero-dep; nothing from this crate's graph leaks back into them.

## Runtime integration model

The server layer takes `Arc`s of the pre-warmed engines directly
through the Rust API (no C ABI, no foreign call overhead):

- `Arc<WhisperAsr>` for OpenAI transcriptions + Wyoming ASR.
- `Arc<PiperPlusTts>` and (optionally) `Arc<KokoroTts>` for piper-plus
  `/api/tts` + Wyoming TTS.
- `Arc<SileroVadV5>` for optional streaming chunk-boundary detection
  in Wyoming.
- A `Box<dyn Phonemizer + Send + Sync>` supplied by the caller /
  binary (`integrations/vokra-piper-g2p` for real 8-language G2P; a
  stub in tests).

All these engines are `Send + Sync` (compile-time proof in
`crates/vokra-core/src/stream/mod.rs`), so the tokio-based server can
share them across worker tasks with zero-cost `Arc::clone`.

## Ticket map (from the plan, T01..T24, 30 min each)

| T# | Deliverable | Depends on |
|----|-------------|-----------|
| T01 | This scope doc + crate skeleton + empty workspace + path deps | — |
| T02 | HTTP/async stack ADR + cargo-deny gate | T01 |
| T03 | Launch base (CLI, config, bind, graceful shutdown, LC_NUMERIC=C) | T02 |
| T04 | Inference service layer (registry, pre-warm, FR-EX-08 no silent fallback) | T03 |
| T05 | error → JSON schema + structured logs + panic isolation (NFR-RL-07) | T03, T04 |
| T06..T08 | OpenAI `/v1/audio/transcriptions` (route + schema + compat test) | T04, T05 |
| T09..T10 | vLLM `/v1/completions` + `/v1/chat/completions` (contract-first) | T04, T05 |
| T11..T13 | piper-plus `/api/tts` (route + service wiring + compat test) | T04, T05 |
| T14..T17 | Wyoming Protocol (spec confirm, ASR, TTS, compat test) | T04, T05 |
| T18 | Server TTS 90 ms latency harness (NFR-PF-05) | T12, T16 |
| T19 | Single-binary deployment check (musl static, Docker-free) | T03 |
| T20 | Security/ops doc (bind default, TLS/CORS via reverse proxy) | T03 |
| T21 | Compliance / watermark forward-compat hooks + research-flag gate | T04 |
| T22 | CI required-check integration (NFR-MT-07) | T08, T10, T13, T17 |
| T23 | Docs (README, deploy, compat matrix, faster-whisper + HA examples) | after impl |
| T24 | WP completion verification (PR → CI green → merge) | all |
