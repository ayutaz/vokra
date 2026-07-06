# ADR: HTTP / async stack for vokra-server

- **Ticket**: M2-09-T02
- **Status**: Accepted (2026-07-06)
- **Scope**: `integrations/vokra-server/` only. Does not affect any
  `crates/vokra-*` runtime crate or the root workspace `Cargo.lock`.

## 1. Context

M2-09 delivers `vokra-server` — a single-binary compatibility server exposing
four APIs on top of Vokra's zero-dependency runtime:

1. OpenAI `/v1/audio/transcriptions` (multipart upload, WAV → text; FR-SV-02)
2. vLLM `/v1/completions` + `/v1/chat/completions` (schema contract only in
   v0.5; FR-SV-03)
3. piper-plus `/api/tts` (text → WAV; FR-SV-04)
4. Home Assistant Wyoming Protocol over TCP (FR-SV-05)

Constraints inherited from the plan and CLAUDE.md:

- **Zero-dep invariant (NFR-DS-02)**: the root workspace `Cargo.lock` must
  remain `vokra-*` only. `scripts/check-zero-deps.sh` only scans the root
  lockfile, so the server's HTTP dependencies must live in an isolated
  workspace under `integrations/` (same pattern as `vokra-piper-g2p`).
- **License policy (NFR-LC-02 / NFR-LC-04)**: dependency crates may be
  Apache-2.0 / MIT / BSD only; GPL / LGPL / AGPL / SSPL forbidden.
- **No silent CPU fallback (FR-EX-08)**: server error paths must surface
  `UnsupportedOp` explicitly; ruling out any framework that traps panics into
  200 responses.
- **API-boundary safety (NFR-RL-07)**: every request handler must be
  panic-isolated; one bad request must not tear down the runtime.
- **No ONNX / protobuf / gRPC** anywhere in the graph.
- **LC_NUMERIC safety (NFR-RL-01)**: JSON number parsing must not go through
  locale-sensitive `strtod`.

## 2. Decision

Adopt the following stack, all pulled into
`integrations/vokra-server/Cargo.toml` only:

| Layer      | Crate         | License | Role                                          |
|------------|---------------|---------|-----------------------------------------------|
| async rt   | `tokio`       | MIT     | multi-threaded runtime, timers, signal, TCP   |
| HTTP core  | `hyper` 1.x   | MIT     | HTTP/1.1 + HTTP/2 server, streaming bodies    |
| routing    | `axum`        | MIT     | typed extractors, router, SSE, WebSocket      |
| middleware | `tower`       | MIT     | `Layer` composition, `CatchPanic` for T05     |
| serde      | `serde`       | MIT/Apache-2.0 | derive-based DTOs                       |
| JSON       | `serde_json`  | MIT/Apache-2.0 | locale-independent number parsing       |
| multipart  | `multer` (via `axum::extract::Multipart`) | MIT | RFC 7578 multipart for OpenAI transcriptions upload |

Wyoming Protocol (JSONL over TCP) rides on the same `tokio` runtime via
`tokio::net::TcpListener` — no additional HTTP framework needed; JSONL is
parsed with `tokio::io::BufReader::read_until('\n')` plus `read_exact` for
the binary payload (see plan §5 R5).

Concrete version pins land in the crate skeleton wave (M2-09-T03); this ADR
records only the license / architecture selection.

## 3. Alternatives considered and rejected

### 3.1 `tiny_http` (Apache-2.0/MIT)

Rejected. It uses blocking thread-per-connection, so serving N Home
Assistant satellites over Wyoming multiplies stack cost linearly. It has no
HTTP/2 or WebSocket, no built-in multipart, and its streaming-body support
is thin — SSE for OpenAI `stream=true` and the audio-chunk fan-out for
Wyoming would each need bespoke framing code. It cannot share an event
loop with the Wyoming TCP listener, forcing two independent shutdown paths.

### 3.2 `warp` (MIT)

Rejected. Built on top of `hyper` + `tower` (same substrate we choose
directly via `axum`), but upstream development has stalled since 2023 and
several 1.x-compatible dependency updates have lagged behind axum. Picking
warp buys us nothing over axum and adds a maintainer-risk axis
(cf. Kill-switch F posture).

### 3.3 `actix-web` (MIT / Apache-2.0)

Rejected. The actor-based execution model is heavier than tokio + tower and
diverges from the rest of the Rust HTTP ecosystem (custom runtime,
non-standard error semantics). Historical `unsafe` incidents make panic
isolation harder to reason about at the API boundary (NFR-RL-07). Migrating
away later would be costly, and we gain no feature we need.

### 3.4 `smol` + `async-h1`

Rejected. Lighter than tokio but ships no first-party multipart,
WebSocket, or SSE story, and the ecosystem around it is thin. tokio's
scheduler is required anyway for Wyoming (long-lived TCP sessions with
timers), so introducing a second runtime is pure overhead.

## 4. Isolation guarantees

- `integrations/vokra-server/` has an empty `[workspace]` table in
  `Cargo.toml`, giving it its own `Cargo.lock`. The root workspace lists
  `exclude = ["integrations"]` (belt-and-suspenders on top of the
  `crates/*` glob).
- Runtime crates (`vokra-core`, `vokra-models`, `vokra-piper-plus`) are
  referenced by `path = "../../crates/…"`. They are not permitted to
  import `hyper` / `tokio` / `axum` / `serde` — reviewers reject such
  changes, and T19 (M2-09) adds a CI check that `cargo tree -p vokra-core`
  contains no HTTP/async crate.
- `scripts/check-zero-deps.sh` remains unchanged; it only scans the root
  `Cargo.lock`, so the server-side graph is by construction invisible to
  the zero-dep gate.

## 5. License audit (cargo-deny)

`integrations/vokra-server/deny.toml` allow-lists Apache-2.0 / MIT /
BSD-2-Clause / BSD-3-Clause / ISC / Unicode-3.0 / Zlib, and explicitly
denies `tiny_http`, `warp`, `actix-web` (so a stray `cargo add` cannot
quietly reintroduce a rejected stack). GPL / LGPL / AGPL / SSPL are
rejected by omission from the allow-list (cargo-deny's default semantics).

**Verification command** (executed as part of T02 completion):

```
cd integrations/vokra-server && cargo deny check licenses
```

**Result**: `ok` — with no third-party dependencies pinned yet (T02 records
only the ADR; concrete `hyper` / `tokio` / `axum` version pins land in
T03), the license graph is trivially clean. The command is wired into CI in
T22 (M2-09-T22, NFR-MT-07 required-check integration) and re-runs on every
version bump under T03.

Zero GPL/LGPL dependencies are recorded here as the T02 exit criterion.

## 6. Consequences

- **Positive**: mature single ecosystem (tokio + hyper + tower + axum),
  first-class multipart and SSE, WebSocket available for future real-time
  ASR, natural fit for the Wyoming TCP listener sharing the tokio runtime,
  clear panic-isolation story via `tower`'s `CatchPanic`.
- **Negative**: the server binary is materially larger than the runtime
  crates (async runtime + HTTP + JSON). This is expected — NFR-DS-01's
  binary-size target applies to core, not to the server binary; documented
  in T19.
- **Reversibility**: because every server-side crate lives inside
  `integrations/vokra-server/`, swapping the stack later (e.g. if `axum`
  ever pivots away from `hyper`) is a local change with no root-workspace
  fallout.

## 7. References

- Plan §1.4 (HTTP stack options), §2 D2 (stack decision), §5 R1 / R3 / R4
  (risks this ADR addresses).
- CLAUDE.md — Kill switch E/F posture, NFR-DS-02 zero-dep invariant,
  FR-EX-08 no-silent-fallback rule.
- `integrations/vokra-piper-g2p/Cargo.toml` — reference implementation of
  the excluded-workspace + empty-`[workspace]` isolation pattern.
- `deny.toml` (root) — parallel policy applied to the runtime workspace.
