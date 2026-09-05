# vokra-kws-micro

> **2026-09-01 current-state boundary:** The phase/status and synthetic-chain
> results below do not claim a green real-weight host-parity result. The fixed
> reviewed stateful binder and conversion contract are implemented, but the
> authenticated 512-stage fixture verdict remains VAST-gated. The Rust source
> and focused gates are authoritative for behavior; current project completion
> and owner gates live in the M5 ledger.

microWakeWord-style keyword-spotting (KWS) forward as a `#![no_std]`
(+ `alloc`) subset. Sister crate to
[`vokra-vad-micro`](../vokra-vad-micro), following the same M5-03 IoT
Tier-3 topology (**NFR-PT-03**): the numeric forward is lifted out of
the std-heavy `vokra-models` so it cross-compiles for bare-metal
**Cortex-M55** (`thumbv8m-none`), and the std `vokra-models`-side
wrapper depends on this crate and re-exports it. The std and no_std
builds share the same forward source; the checks below cover
deterministic repeat calls, not a published cross-target binary
identity proof.

## Status

**Phase 3+ reviewed stateful binder and Phase 4 host-parity preflight.** Not
graduated to crates.io yet (`publish = false`). The sidecar preserves dense
`GGML_TYPE_I8` source-byte carriers and exact `GGML_TYPE_I32` bias carriers,
plus a fail-closed supported-topology manifest. The fixed reviewed
`hey_jarvis` authority is now exposed by
`Model::bind_authenticated_streaming()`, which checks exact provenance,
topology, tensor quantization, and weight/bias fingerprints before returning a
stateful executor. Candidate GGUFs and `bind_untrusted_topology` remain
untrusted; the exact 512-invocation LiteRT stage-trace parity run is still
VAST-gated, so this crate remains blocked from a production parity claim.

- **Feature extractor (`src/features.rs`)** — 40-band log-mel front-end
    (Phase 1, WF1). Real; parity harness covers it.
- **INT8 kernels (`src/kernels.rs`)** — Conv2D / DwConv2D / FC / Sigmoid
    / Softmax scalar path (Phase 2, WF1 Resume). Real; per-kernel unit
    tests.
- **Model loader (`src/model.rs`)** — reads Vokra `vokra.kws.*` GGUF
    emitted by `tools/parity/microwakeword/prepare_checkpoint.py`
    (Phase 2, WF1 Resume). Real; shape-generic F32/dense-I8/I32 tensor view
    with source affine metadata and exact I32 bias preservation. Candidate
    production carriers are dense `GGML_TYPE_I8` plus exact `GGML_TYPE_I32`;
    this loader does not confer authority by itself; the fixed reviewed
    `hey_jarvis` artifact must pass `bind_authenticated_streaming()`.
- **Interpreter (`src/interpreter.rs`)** — `LayerSpec` + `ChainConfig`
    ping-pong chain executor (Phase 3, WF2). Real; unit-tested end-to-
    end with a synthetic 2-layer chain.
- **`KwsMicro::detect()` (`src/lib.rs`)** — wires the log-mel front-end
    into a `ChainConfig` chain (Phase 3, WF2). Real. With no chain
    attached it refuses with `VokraError::ModelLoad`; it does not return
    `KwsEvent::Idle`, which is a legitimate per-frame result and would
    make an unconfigured detector look like a configured one hearing
    silence (FR-EX-08). `has_chain()` tells the two states apart.
- **Host parity harness
    (`tests/parity_microwakeword.rs`)** — Phase 4 (this doc). Env-gated
    reference comparison against a numpy log-mel transcription and the
    real upstream TFLite forward.
- **Authenticated streaming binder (`src/model.rs`)** — binds only the fixed
    reviewed `hey_jarvis` GGUF to the persistent 11-layer INT8 executor and
    exposes the final uint8 quantize plus stage trace. It is directly usable by
    callers holding a loaded `Model`; `KwsMicro` intentionally remains the
    separate feature-extractor + caller-attached `ChainConfig` convenience API
    and does not silently install model state.

## Design red lines (inherited from `vokra-vad-micro`)

- **Zero external deps (NFR-DS-02)** — only `vokra-core` (with
    `default-features = false` so the no_std subset compiles). Root
    `Cargo.lock` stays `vokra-*` only.
- **No `unsafe` (NFR-RL-07)** — workspace lint `unsafe_code = "deny"`;
    this crate contains none.
- **No `libm`** — `deny.toml` bans it. Transcendentals come from the
    self-contained `crate::scalar` module (mirroring
    [`vokra-vad-micro::scalar`](../vokra-vad-micro/src/scalar.rs)).
- **1:1 preservation** — microWakeWord is a dedicated subgraph, not
    lowered to generic audio-dialect ops.

## Shared std/no_std source and deterministic smoke

Two enforcement layers, cross-verified:

1. **Compile gate** —
    [`scripts/check-nostd-subset.sh`](../../scripts/check-nostd-subset.sh)
    builds this crate with `--no-default-features` (= `#![no_std]`) for
    `thumbv8m.main-none-eabi(hf)`. Any accidental `use std::…` fails
    the build.
2. **Determinism smoke** — `tests::kws_detect_is_bit_identical_across_
    repeat_calls_std_and_no_std` in `src/lib.rs` runs `detect()` twice
    on the same input and asserts bit-identical outputs (tolerance =
    0). Non-determinism (e.g. `HashMap` iteration, unseeded PRNG,
    environment read) would surface here.

Together these establish shared source and deterministic repeat-call
behavior for the tested build. They do not prove bit-identical artifacts
across targets, compilers, or floating-point implementations; a separate
cross-target parity result would be required for that claim.

The Rust test harness itself requires `std`, so `cargo test` runs only
the std build; the compile gate proves the no_std build is a subset of
what `cargo test` exercises.

## Owner walkthrough — thumbv8m cross-build sanity

Run once per checkout to confirm the Cortex-M55 (Tier-3) target
compiles cleanly:

```
rustup target add thumbv8m.main-none-eabihf
CARGO_BUILD_JOBS=1 cargo build -p vokra-kws-micro \
    --target thumbv8m.main-none-eabihf \
    --no-default-features
```

Notes:

- The Cargo.toml declares only two features: `default = ["std"]` and
    `std = ["vokra-core/std"]`. `alloc` is unconditional (via
    `extern crate alloc` in `src/lib.rs`) — do NOT pass
    `--features alloc` (no such feature exists).
- Also build the soft-float ABI variant for coverage:
    `--target thumbv8m.main-none-eabi`.
- `CARGO_BUILD_JOBS=1` mirrors the project's memory-safe cargo posture
    (16 GB M1 iMac). Drop it on larger machines.

Per M5-03 ADR the cross-build is **owner-triggered** for this crate:
`scripts/check-nostd-subset.sh` currently enforces the no_std subset
for `vokra-core` + `vokra-vad-micro` (the WF1 landing set). Adding
`vokra-kws-micro` to that script's `CRATES=(…)` list is a future ADR
question — the CI-side compile gate for this crate is not on yet
(intentional; Phase 4's scope is host parity + owner walkthroughs, not
new CI coverage). Actual **Cortex-M55 hardware verify** is also owner-
only (M5-03 ADR: no FVP / real-hardware CI job for Tier-3).

## Owner workflow — VAST reference and parity preflight

Model acquisition, conversion, reference execution, and real Rust parity
are not local maintainer walkthroughs. Local work is limited to the
stdlib-only self-tests, static checks, and safe package-scoped checks; do not
run a model, fetch a checkpoint, or synchronize the reference environment on
the maintainer machine.

The ordered remote workflow is:

1. Start from a clean checkout/bundle at the fixed worker commit. Run the
   isolated dependency/license audit in
   `tools/parity/microwakeword-reference/inspect.py` before any reference
   environment sync. Its current transitive-license result is
   `BLOCKED_UNREVIEWED_TRANSITIVE`, so fixture generation remains closed until
   the bounded primary-source review is recorded.
2. On the fixed clean VAST worker, use only the worker-reported absolute
   candidate paths and authenticated model identity. Produce a `NO_UPLOAD`
   candidate; the candidate is not a production artifact and arbitrary URL or
   caller-supplied digest authority is forbidden.
3. After the audit gate is cleared, generate the independent LiteRT fixture
   from the pinned upstream model. It must retain quantized input bytes, raw
   uint8 outputs, dequantized outputs, manifest hashes, and the fresh-interpreter
   reset replay for the stateful multi-invocation sequence. The dumper's
   mandatory dependency-evidence input must be the successful collection
   report from that same VAST environment; it is provenance evidence, not an
   owner license/publication approval.
4. Collect the evidence archive, run the env-gated Rust paths in the same
   controlled validation workflow, and destroy the VAST worker. Scaleway is
   reserved for the separate final Apple CPU/Metal verification; reference
   generation belongs on VAST.

The Rust harness has three explicit, ordered paths. Missing environment
variables are a clean skip, never a fabricated pass:

- **Path A** (`VOKRA_KWS_REAL_GGUF`) — authenticated VAST artifact-load smoke:
  verify the real file and `vokra.kws.*` metadata. A candidate or a manifest
  alone cannot authorize production.
- **Path B** (`VOKRA_KWS_REAL_FIXTURES`) — feature extractor parity at the
  registered `atol = 5e-2` boundary against the independent upstream
  transcription reference. This is a frontend result only, not end-to-end
  model parity.
- **Path C** (both variables) — authenticated streaming INT8 chain parity over
  every recorded invocation and every preserved intermediate stage, plus reset
  replay. The input contract is int8
  `[1, 3, 40]` with scale `0.10196078568696976`, zero-point `-128`; the output
  contract is uint8 `[1, 1]` with scale `1/256`, zero-point `0`. Once fixture
  checks pass, a missing authenticated streaming binder is a hard failure;
  Path C must never skip or report PASS in that state. The binder is now
  implemented, but only a green VAST fixture run can establish the numerical
  production verdict. The outer VAST gate must recompute artifact hashes
  before any parity claim.

## See also

- Design ADR: `docs/adr/M5-03b-kws-micro-no-std.md` (gitignored,
    local).
- Sister crate: [`vokra-vad-micro`](../vokra-vad-micro) — Silero VAD
    v5 no_std forward, the topology precedent this crate mirrors.
- VAST-only sidecars: `tools/parity/microwakeword/` (candidate inspection and
    conversion) plus `tools/parity/microwakeword-reference/` (independent
    LiteRT fixture and dependency audit).
- Upstream: <https://github.com/kahrendt/microWakeWord> (Apache-2.0).
- Curated model mirror:
    <https://github.com/esphome/micro-wake-word-models> (Apache-2.0).
