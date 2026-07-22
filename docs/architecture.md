# Architecture

**English** | [日本語](architecture.ja.md)

An orientation map for people reading or extending the Vokra source tree: what
each crate is for, how a model actually executes, and which design decisions
are fixed and will not be revisited in review.

For the requirement IDs cited here, see
[requirement-ids.md](requirement-ids.md). For review rules, see
[CONTRIBUTING.md](../CONTRIBUTING.md).

---

## 1. Crate map

The repository is a Cargo **virtual workspace**. Members are `crates/*` plus
two test-only crates; `integrations/` is deliberately excluded.

<!-- anchor: Cargo.toml -->

### 1.1 The runtime graph

Fourteen crates live under `crates/`. Every one of them is a first-party
`vokra-*` crate — that is what makes the zero-dependency invariant
(`NFR-DS-02`) achievable at all.

| Crate | Role |
|---|---|
| `vokra-core` <!-- anchor: crates/vokra-core/src/lib.rs --> | The IR and the execution engine. Holds the audio-graph descriptor (`FR-EX-01`), the graph evaluator, the `Backend` trait every backend implements, the GGUF loader and the task-level engine traits. Contains **no `unsafe`**. |
| `vokra-ops` <!-- anchor: crates/vokra-ops/src/lib.rs --> | The speech operators: STFT / iSTFT / mel filterbank / MFCC / DCT and the rest of the audio dialect (`FR-OP-*`), plus the CPU FFT lowering — a from-scratch Rust reimplementation of the pocketfft algorithm rather than a bound C library. |
| `vokra-backend-cpu` <!-- anchor: crates/vokra-backend-cpu/src/lib.rs --> | The first-class CPU backend (`FR-BE-01`): f32 compute kernels plus the single-binary runtime ISA dispatch that picks an implementation for the host CPU. |
| `vokra-backend-metal` <!-- anchor: crates/vokra-backend-metal/src/lib.rs --> | macOS / iOS GPU backend over **hand-written raw Objective-C runtime + Metal FFI**. No `metal` / `objc2` binding crate. |
| `vokra-backend-cuda` <!-- anchor: crates/vokra-backend-cuda/src/lib.rs --> | NVIDIA GPU backend over the CUDA Driver API + NVRTC, loaded at **runtime via `dlopen` / `LoadLibrary`**. No `cudarc` / `cust` / `rustacuda`, and no CUDA library is bundled or linked — the user's system installation is discovered at run time (this is what keeps distribution clear of NVIDIA's redistribution terms). |
| `vokra-backend-vulkan` <!-- anchor: crates/vokra-backend-vulkan/src/lib.rs --> | Android / Linux (and non-NVIDIA desktop) GPU backend over raw Vulkan FFI, also `dlopen`-loaded, with pre-compiled SPIR-V shaders. No `ash` / `vulkano` / `erupt`. |
| `vokra-backend-webgpu` <!-- anchor: crates/vokra-backend-webgpu/src/lib.rs --> | Browser backend (`FR-BE-05`) over a hand-written WASM extern-import shim onto the WebGPU API. No `wgpu` / `wasm-bindgen`: import-object resolution at instantiate time is the WASM equivalent of `dlopen`. |
| `vokra-models` <!-- anchor: crates/vokra-models/src/lib.rs --> | The native model implementations. Models are re-implemented in Rust whisper.cpp-style: the model *definition* lives here and only upstream **checkpoints** are consumed. This is also where the piper-plus TTS inference core lives. |
| `vokra-piper-plus` <!-- anchor: crates/vokra-piper-plus/src/lib.rs --> | The piper-plus **G2P reuse bridge** and voice-model conversion helpers — and only that. The inference core (MB-iSTFT-VITS2) is natively implemented in `vokra-models`; the earlier "wrap piper-plus" positioning was abolished by maintainer decision. |
| `vokra-convert` <!-- anchor: crates/vokra-convert/src/lib.rs --> | The offline checkpoint → GGUF converter (`FR-TL-01`). **The only place ONNX / protobuf handling is allowed to exist** — see red line R1 below. |
| `vokra-mmap` <!-- anchor: crates/vokra-mmap/src/lib.rs --> | True `mmap`-backed GGUF loading (`FR-LD-01`, `NFR-PF-11`). It exists as a separate crate so that the `unsafe` a real memory map requires stays out of `vokra-core`. |
| `vokra-capi` <!-- anchor: crates/vokra-capi/src/lib.rs --> | The C ABI surface (`IF-01`, `BR-04`). Its public product is a set of `extern "C"` symbols; the generated header is `include/vokra.h`. Everything Unity, Godot, Swift, Kotlin, Python and JS use sits on this. |
| `vokra-cli` <!-- anchor: crates/vokra-cli/src/main.rs --> | The umbrella command-line tool (`FR-TL-02`): `run`, `convert`, `bench`. **A binary crate** — it is the one crate with no `src/lib.rs`. Argument parsing is hand-written. |
| `vokra-eval` <!-- anchor: crates/vokra-eval/src/lib.rs --> | Evaluation metrics (`FR-OP-93`, `FR-TL-03`) — mel loss, WER, CER — as a reusable library plus a CLI. |

Two further workspace members are test-only:

- `tests/parity` <!-- anchor: tests/parity --> — the numerical parity harness
  (`NFR-QL-01`), published as the crate `vokra-parity`. It is what the
  `parity` required check runs.
- `tests/wasm-harness` <!-- anchor: tests/wasm-harness --> — the WASM entry
  crate (`vokra-wasm-harness`) exercising the browser `(ptr, len)` ABI.

### 1.2 Dependency direction

Normal (non-dev) dependencies only. The graph is acyclic and `vokra-core`
is the root — nothing in the workspace is upstream of it.

```
vokra-core            (no dependencies — the root)
  ├── vokra-ops
  ├── vokra-mmap
  ├── vokra-piper-plus
  ├── vokra-backend-cpu
  ├── vokra-backend-{metal,cuda,vulkan,webgpu}
  ├── vokra-eval        → core, ops, backend-cpu
  ├── vokra-convert     → core, ops, mmap
  ├── vokra-models      → core, ops, backend-cpu, piper-plus, mmap,
  │                       backend-{metal,cuda,vulkan,webgpu} (optional features)
  ├── vokra-capi        → core, models, ops, mmap
  └── vokra-cli         → core, models, ops, convert, mmap
```

A couple of *dev*-dependencies point the other way (`vokra-ops` uses
`vokra-models` in its tests, for instance). Those are test-only edges and do
not make the build graph cyclic.

### 1.3 `integrations/` — outside the invariant, on purpose

`integrations/` is **excluded** from the root workspace. It currently holds
five crates:

| Path | Purpose |
|---|---|
| `integrations/vokra-server` <!-- anchor: integrations/vokra-server --> | The HTTP server binary (`FR-SV-01`, `FR-SV-02`, `FR-SV-04`, `FR-SV-05`) |
| `integrations/vokra-piper-g2p` <!-- anchor: integrations/vokra-piper-g2p --> | The real 8-language G2P bridge |
| `integrations/vokra-godot` <!-- anchor: integrations/vokra-godot --> | The Godot GDExtension |
| `integrations/vokra-server-bench` <!-- anchor: integrations/vokra-server-bench --> | Server latency benchmark harness |
| `integrations/vokra-cli-bench-server` <!-- anchor: integrations/vokra-cli-bench-server --> | CLI-side benchmark server |

**Why they are allowed to use external crates.** The zero-dependency
invariant is a statement about *one specific file*: the root `Cargo.lock` must
resolve to `vokra-*` crates only. Each `integrations/` crate is its own
workspace with its own `Cargo.lock`, so whatever it depends on never enters
the root resolution. The runtime reaches them across a trait boundary, never
by linking them into the runtime graph. `scripts/check-zero-deps.sh` checks
the root lockfile and is a hard gate both locally and in CI.

<!-- anchor: scripts/check-zero-deps.sh -->

### 1.4 The `unsafe` boundary

The workspace is safe-by-default: `unsafe_code = "deny"` is set at the
workspace level, and every `unsafe` block must carry a `// SAFETY:` comment
(enforced by clippy). Exactly **nine** crates opt out locally, each for a
reason that cannot be met in safe Rust (`NFR-RL-07`):

| Crate | Why it needs `unsafe` |
|---|---|
| `vokra-ops` | SIMD intrinsics in operator hot paths |
| `vokra-backend-cpu` | SIMD intrinsics and ISA dispatch |
| `vokra-backend-metal` | Objective-C runtime + Metal FFI |
| `vokra-backend-cuda` | CUDA Driver API + NVRTC FFI |
| `vokra-backend-vulkan` | Vulkan FFI |
| `vokra-backend-webgpu` | WASM extern-import shim |
| `vokra-capi` | The C ABI boundary itself |
| `vokra-mmap` | POSIX `mmap` / Win32 file mapping |
| `vokra-wasm-harness` | The `(ptr, len)` WASM ABI boundary |

`vokra-core` is **not** on this list and must stay off it. Public API
boundaries remain safe in every crate, including these nine.

---

## 2. Execution model

Vokra runs models through **two paths that share their kernels**. Knowing
which one you are in is the single most useful orientation fact in the
codebase.

### 2.1 Path A — the graph evaluator

<!-- anchor: crates/vokra-core/src/runtime/mod.rs -->
<!-- anchor: crates/vokra-core/src/runtime/tensor.rs -->

`vokra-core`'s `runtime` module holds a data-carrying graph evaluator. The
audio-graph IR is a *descriptor* — its tensors carry shapes but no data — and
`run_graph` is the engine that threads real tensor values from node to node,
driving one op at a time through `Backend::eval_op`.

Its contract:

- **One graph, one backend, no silent fallback** (`FR-EX-08`). Before
  evaluating anything, `run_graph` verifies the backend covers *every* op in
  the graph. A single unsupported op is an explicit error. There is no per-op
  CPU fallback and no ONNX-Runtime-style execution-provider partitioning.
- **Deterministic schedule.** Nodes execute in topological order (Kahn,
  index-stable for independent nodes), so a graph evaluates identically on
  every run.
- **Validation lives in the engine**, not the backend: `eval_op` only
  computes, while `run_graph` checks output arity and shapes against the
  declared descriptors.

This path is the right shape for new, fused, and graph-first models.

### 2.2 Path B — the `Compute` seam

<!-- anchor: crates/vokra-models/src/compute.rs -->

The models that already existed (Whisper, piper-plus, CAM++) are written
**imperatively**: they call compute kernels directly in a zero-malloc hot path
with caller-owned scratch buffers. Rewriting them onto the graph engine would
add a large op surface and put their numerical parity at risk for no speed
gain, because it is the same kernels underneath either way.

So instead those call sites dispatch through a thin typed seam, `Compute`.
Swapping one enum arm moves the same GEMM from CPU to GPU.

### 2.3 Why this is not two implementations

**There is one kernel per `(backend, op)` pair, and both paths call it.**
`Compute::gemm_f32` on the CPU arm calls exactly the same
`vokra_backend_cpu::kernels::gemm_f32` that `Backend::eval_op` calls; on the
Metal arm it calls exactly the same `MetalContext::gemm_f32`. There is no
second kernel to keep in sync, and the imperative and graph paths stay
bit-for-bit consistent on a given backend.

The seam also enforces `FR-EX-08` at model granularity: `Compute::for_backend`
takes the model's *required* hot-op set and refuses to build a backend that
does not cover every op in it. Selecting the CPU is an explicit caller
choice, never a silent degradation.

> **Note on a stale pointer.** The header comment of `compute.rs` references
> a `scratchpad/graph-engine-plan.md` design note that is not part of the
> repository. This page is the public, canonical description of that design;
> prefer it over the dangling reference.

### 2.4 Backend layout: the six-file pattern

The four GPU / FFI backends share a common skeleton, so once you have read
one you can navigate the others:

| File | Responsibility |
|---|---|
| `sys.rs` | Raw FFI declarations — the hand-written binding layer |
| `probe.rs` | Runtime detection: is this device/driver actually usable? |
| `context.rs` | Device, queue and buffer lifetime; the compute kernels |
| `backend.rs` | The `vokra-core` `Backend` trait implementation, including honest op-coverage reporting |
| `eval.rs` | The `eval_op` dispatch for the graph path |
| `lib.rs` | Crate docs, the `unsafe` opt-out, and re-exports |

Backends add files beyond those six where they need to: CUDA carries
`fa_v3.rs` and `session_pool.rs`, Vulkan carries `kernels.rs`, `plan.rs` and
`spirv.rs`, WebGPU carries `plan.rs` and `wgsl.rs`.

**The CPU backend does not follow this pattern** — it is not an FFI backend
and has no device to probe. Its layout is `dispatch.rs`, `eval.rs`,
`features.rs`, `kernels/`, `lib.rs`, `pool.rs`, `selftest.rs`. Do not
generalise the six-file skeleton to all five backends.

---

## 3. Design red lines

These are settled decisions. A PR that crosses one will be declined
regardless of how well it is implemented — not because the idea is bad, but
because the cost of the decision was already paid and reopening it is more
expensive than the benefit. The rules are listed in
[CONTRIBUTING.md](../CONTRIBUTING.md) §5; the reasoning is here.

<!-- anchor: CONTRIBUTING.md -->

### R1 — No ONNX graph loading in the runtime (`FR-LD-05`)

ONNX models are handled **exclusively** by the offline converter
(`vokra-convert`). The runtime carries no onnxruntime, onnx or protobuf
dependency.

*Why.* The project exists because of a catalogue of problems in the ONNX
speech stack. Loading ONNX graphs at run time would drag protobuf, abseil and
onnx back into the dependency graph, which destroys `NFR-DS-02` — and the
zero-dependency property is precisely what makes single-binary distribution
into Unity, Godot and mobile targets viable. Keeping ONNX in one offline crate
is what lets the rest of the tree stay clean.

### R2 — No onnxruntime in the piper-plus inference path

The MB-iSTFT-VITS2 inference stack is natively reimplemented in Rust. Only
the G2P text preprocessing is reused from piper-plus, for now.

*Why.* Wrapping would leave onnxruntime in the end-to-end path of a project
whose entire claim is to be an alternative to it. The native implementation
also made the `istft` operator a real consumer, which is how the audio-dialect
design got validated rather than merely specified.

### R3 — No eSpeak-NG in the core

*Why.* It is GPL-3.0. Vokra targets Unity, Godot and other proprietary
embedding scenarios, where GPL is not acceptable to the people shipping the
product. G2P comes from piper-plus's own MIT implementation or from
IPA-dictionary approaches. The same reasoning excludes soxr and rubberband
(R5).

### R4 — No NNAPI backend

*Why.* Google deprecated it as of Android 15. Betting the Android
acceleration story on a deprecated single-vendor abstraction would buy a
migration project rather than performance. Android GPU acceleration goes
through Vulkan instead.

### R5 — No soxr, no rubberband

*Why.* GPL, as in R3. Resampling is implemented natively, based on the
speexdsp (BSD) resampler design.

### R6 — Unsupported means an error, never a silent fallback (`FR-EX-08`)

This one is not in the CONTRIBUTING list as a "red line" but is enforced just
as strictly, in both execution paths and at the backend probe level
(`NFR-RL-06`).

*Why.* A silent CPU fallback turns a missing kernel into a performance
mystery: the code runs, the numbers look plausible, and the regression shows
up as an unexplained latency change on someone else's machine weeks later. An
explicit error costs one bug report and saves the investigation.

The same instinct governs test tolerances: a parity `atol` is derived from an
architectural bound, not tuned until CI is green. Widening one to pass a check
is the same failure as a silent fallback, and reviewers treat it that way.

### The two sanctioned escapes from zero-dependency

`NFR-DS-02` is strict, but it is not a wall. There are exactly two ways
through it, and both preserve the property that the root `Cargo.lock`
resolves to `vokra-*` only:

1. **A first-party optional feature.** The GPU backends are ordinary `vokra-*`
   crates with hand-written raw FFI, gated OFF behind Cargo features, so a
   default build never even names them. This is how a new GPU or NPU path
   should arrive.
2. **An isolated integration workspace.** Code that genuinely needs an
   external crate lives under `integrations/` in its own workspace with its
   own lockfile, wired in across a trait boundary (§1.3).

If a change needs neither of these, it needs no new dependency.

---

## 4. Adding a backend

The six-file pattern in §2.4 is the starting point. In outline:

1. **Create `crates/vokra-backend-<name>/`** as a first-party crate, gated
   behind an optional Cargo feature so default builds are unaffected.
2. **Write `sys.rs` by hand.** Declare the FFI you need yourself rather than
   adding a binding crate — that is what keeps `NFR-DS-02` intact. Load the
   platform library at run time (`dlopen` / `LoadLibrary`, or import-object
   resolution on WASM) rather than link-time, so a machine without the driver
   still runs the binary.
3. **Write `probe.rs` next, not last.** Detection must be honest: report the
   device as unusable rather than proceeding hopefully (`NFR-RL-06`).
4. **Implement `context.rs` kernels against the CPU backend as the reference.**
   The CPU kernel is the numerical oracle; parity tests compare against it
   (`NFR-QL-01`).
5. **Implement `backend.rs` op coverage truthfully.** Report only ops you
   actually implement. Under-reporting yields an explicit error, which is
   correct; over-reporting yields wrong numbers, which is not (`FR-EX-08`).
6. **Wire `eval.rs` for the graph path and add the `Compute` arm** for the
   imperative path, reusing the same kernels (§2.3).

Every `unsafe` block needs a `// SAFETY:` comment, and the crate must be added
to the opt-out list in the workspace manifest (§1.4).

---

## 5. Further reading

- [requirement-ids.md](requirement-ids.md) — resolve any `FR-*` / `NFR-*` /
  `BR-*` / `IF-*` ID cited in the source
- [CONTRIBUTING.md](../CONTRIBUTING.md) — PR process, required checks,
  dependency policy
- [getting-started.md](getting-started.md) — a 5-minute build-and-run
- [design/m0-03-gguf-loader.md](design/m0-03-gguf-loader.md) — GGUF loader design
- [design/vokra-gguf-chunks.md](design/vokra-gguf-chunks.md) — the `vokra.*`
  metadata chunks (`FR-LD-02`, `IF-07`)
- [design/quantization-policy.md](design/quantization-policy.md) —
  quantization policy (`FR-QT-02`)
- [design/size-budget.md](design/size-budget.md) — binary size budget
  (`NFR-DS-01`)
- [license-audit.md](license-audit.md) — model and dependency licence audit
