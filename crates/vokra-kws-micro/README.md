# vokra-kws-micro

> **2026-08-30 current-state boundary:** The phase/status and synthetic-chain
> results below describe the current scaffold contract, not a claim that a
> real `hey_jarvis.tflite` checkpoint has been acquired, converted, or accepted
> through host parity. The Rust source and focused gates are authoritative for
> behavior; current project completion and owner gates live in the M5 ledger.

microWakeWord-style keyword-spotting (KWS) forward as a `#![no_std]`
(+ `alloc`) subset. Sister crate to
[`vokra-vad-micro`](../vokra-vad-micro), following the same M5-03 IoT
Tier-3 topology (**NFR-PT-03**): the numeric forward is lifted out of
the std-heavy `vokra-models` so it cross-compiles for bare-metal
**Cortex-M55** (`thumbv8m-none`), and the std `vokra-models`-side
wrapper depends on this crate and re-exports it — one forward, shared
bit-identically between the std and no_std builds.

## Status

**Phase 3+ REAL detect() with typed topology binder and Phase 4 host-parity
harness.** Not graduated to crates.io yet (`publish = false`). The sidecar
emits Q8_0 source-byte carriers, exact dense I32 bias carriers, and a
fail-closed supported topology manifest. `Model::bind_untrusted_topology` consumes that
manifest, but a real `hey_jarvis.tflite` bind still requires VAST evidence and
independent parity.

- **Feature extractor (`src/features.rs`)** — 40-band log-mel front-end
    (Phase 1, WF1). Real; parity harness covers it.
- **INT8 kernels (`src/kernels.rs`)** — Conv2D / DwConv2D / FC / Sigmoid
    / Softmax scalar path (Phase 2, WF1 Resume). Real; per-kernel unit
    tests.
- **Model loader (`src/model.rs`)** — reads Vokra `vokra.kws.*` GGUF
    emitted by `tools/parity/microwakeword/prepare_checkpoint.py`
    (Phase 2, WF1 Resume). Real; shape-generic F32/Q8_0/I32 tensor view with
    source affine metadata and exact I32 bias preservation.
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

## Bit-identical std ↔ no_std by construction

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

Together these guarantee: the std and no_std builds compile the same
source, and that source produces deterministic outputs — therefore
bit-identical by construction (there is no code path that runs
different arithmetic under one feature and not the other).

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

## Owner walkthrough — host parity harness

Env-gated: absent env vars ⇒ clean skip (never fabricated pass). See
`tests/parity_microwakeword.rs`'s module doc for the full recipe and
each test path's contract. Short form:

```
# 1. Convert canonical hey_jarvis to a Vokra GGUF (Phase 1 sidecar):
cd tools/parity/microwakeword
uv sync
uv run python prepare_checkpoint.py \
    --url    https://github.com/esphome/micro-wake-word-models/raw/main/models/v2/hey_jarvis.tflite \
    --name   hey_jarvis \
    --output ~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.gguf

# 2. Download the raw .tflite once (dumper needs it):
curl -L -o ~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.tflite \
    https://github.com/esphome/micro-wake-word-models/raw/main/models/v2/hey_jarvis.tflite

# 3. Dump reference artefacts (Phase 4 sidecar):
uv run python dump_reference.py \
    --tflite-path ~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.tflite \
    --output-dir  ~/.cache/vokra-eval/fixtures/microwakeword \
    --verbose

# 4. Point the Rust parity harness at both artefacts:
export VOKRA_KWS_REAL_GGUF=~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.gguf
export VOKRA_KWS_REAL_FIXTURES=~/.cache/vokra-eval/fixtures/microwakeword
CARGO_BUILD_JOBS=1 cargo test -p vokra-kws-micro \
    --test parity_microwakeword -- --nocapture
```

Path breakdown:

- **Path A** (`VOKRA_KWS_REAL_GGUF`) — real GGUF load smoke: bind, walk
    `vokra.kws.*` metadata, assert tensor manifest lower bound. A real
    hey_jarvis result still requires the owner-reviewed VAST artifact.
- **Path B** (`VOKRA_KWS_REAL_FIXTURES`) — log-mel feature extractor
    parity at `atol = 1e-3` against the numpy reference. Real; validates
    transcription faithfulness of the standard log-mel algorithm.
- **Path C** (both) — end-to-end INT8 chain parity. **UNMET as of
    Phase 4**; Q8_0 carriers and per-tensor `(scale, zero_point)` metadata
    exist, but production topology authority and the Model → `ChainConfig`
    binding remain pending. See `src/model.rs`'s
    module doc for the boundary.

## See also

- Design ADR: `docs/adr/M5-03b-kws-micro-no-std.md` (gitignored,
    local).
- Sister crate: [`vokra-vad-micro`](../vokra-vad-micro) — Silero VAD
    v5 no_std forward, the topology precedent this crate mirrors.
- Offline sidecar: `tools/parity/microwakeword/` (TFLite → GGUF
    conversion + reference dumper).
- Upstream: <https://github.com/kahrendt/microWakeWord> (Apache-2.0).
- Curated model mirror:
    <https://github.com/esphome/micro-wake-word-models> (Apache-2.0).
