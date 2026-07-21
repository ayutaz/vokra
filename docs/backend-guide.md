# Adding a backend

**English** | [日本語](backend-guide.ja.md)

This is the end-to-end guide for adding a **new compute backend** to Vokra —
the full checklist, with every step anchored to real source. It is the
step-by-step companion to [architecture.md](architecture.md), which is the
*orientation map*: read §2.4 and §4 there first for the concepts (the crate
map, the execution model, the six-file pattern), then come here to actually
build one. To resolve any `FR-*` / `NFR-*` / `IF-*` ID cited below, see
[requirement-ids.md](requirement-ids.md).

Vokra ships five compute backends today — CPU, Metal, CUDA, Vulkan and WebGPU
— and two delegate scaffolds (CoreML, QNN). Every one of them is a first-party
`vokra-*` crate with **no external binding crate**, because that is what keeps
the zero-dependency invariant (`NFR-DS-02`) intact.

## 1. Before you start: the two invariants a backend must not break

A new backend is one of only two sanctioned ways to reach outside the runtime
graph without breaking `NFR-DS-02` (the other is an isolated integration
workspace; see [CONTRIBUTING.md](../CONTRIBUTING.md) §7). <!-- anchor: CONTRIBUTING.md -->
Both hold for every backend:

- **Zero external dependencies.** Declare the platform FFI yourself in an
  `unsafe extern` block; do **not** add a `metal` / `cudarc` / `ash` / `wgpu`
  binding crate. The root `Cargo.lock` must stay `vokra-*`-only
  (`scripts/check-zero-deps.sh` enforces this).
- **No silent fallback (`FR-EX-08`).** An op a backend cannot run, or a device
  that is not present, is an **explicit error** — never a quiet substitution
  of the CPU. Under-reporting coverage yields a loud error, which is correct;
  over-reporting yields wrong numbers, which is not.

## 2. The six-file pattern

The GPU/FFI backends share a six-module layout. Use the Metal backend as the
canonical template:

| File | Role |
|---|---|
| `sys.rs` <!-- anchor: crates/vokra-backend-metal/src/sys.rs --> | Hand-written raw FFI: the `extern` declarations and runtime library loading (`dlopen` / `LoadLibrary`, or framework linking). No binding crate. |
| `probe.rs` <!-- anchor: crates/vokra-backend-metal/src/probe.rs --> | Honest device detection. Returns `VokraError::BackendUnavailable` when the device/driver is absent (`NFR-RL-06`). |
| `context.rs` <!-- anchor: crates/vokra-backend-metal/src/context.rs --> | The live device/queue/allocations and the compute kernels themselves. |
| `backend.rs` <!-- anchor: crates/vokra-backend-metal/src/backend.rs --> | The `Backend` trait impl: `supports()` + `eval_op()`, kept lock-step. |
| `eval.rs` <!-- anchor: crates/vokra-backend-metal/src/eval.rs --> | The graph-executor per-op arm that `run_graph` drives. |
| `lib.rs` <!-- anchor: crates/vokra-backend-metal/src/lib.rs --> | Crate root, feature gating and re-exports. |

**Honest note — the CPU backend is not six-file.** `vokra-backend-cpu` uses a
different layout (`dispatch.rs` / `features.rs` / `kernels/` / `pool.rs` /
`selftest.rs`) because it is the runtime ISA-dispatch backend (`FR-BE-01`), not
a device FFI backend. The six-file pattern is the template for **GPU/FFI/NPU**
backends; do not force the CPU backend into it. Vulkan and WebGPU also add a
couple of files (`spirv.rs` / `wgsl.rs` / `plan.rs`) for their shader stories.

## 3. The add-a-backend checklist

Steps 1–5 stand up the crate and its coverage contract; steps 6–9 wire it into
execution and prove it numerically.

1. **New crate `crates/vokra-backend-<x>/`**, gated behind an optional Cargo
   feature that is **OFF by default**, so default (and other-platform) builds
   never even name it. Target-gate the real FFI with `cfg(target_os = …)` /
   `cfg(target_arch = …)`.
2. **`sys.rs` — raw FFI, loaded at run time.** Load the platform library with
   `dlopen` / `LoadLibrary` (or wasm import-object resolution) rather than
   link-time, so a machine without the driver still runs the binary. No
   binding crate (`NFR-DS-02`).
3. **`probe.rs` next, not last.** Detection must be honest: return
   `VokraError::BackendUnavailable` when the device is not usable
   (`NFR-RL-06`) rather than proceeding hopefully.
4. **Add the `BackendKind` variant** in `vokra-core`. <!-- anchor: crates/vokra-core/src/backend.rs --> The enum is
   `#[non_exhaustive]`, so a new variant is backwards-compatible; NNAPI is the
   one variant that will never be added (`FR-BE-07`).
5. **Implement `Backend::supports()` + `Backend::eval_op()` lock-step.** The
   default `eval_op` returns `UnsupportedOp` for every op, so a half-wired
   backend still compiles; override only the ops you actually implement, and
   make `supports()` return `true` for exactly that set (`FR-EX-08`).
6. **Wire the graph-executor arm** (`eval.rs`), using the CUDA or Vulkan arm as
   the model. <!-- anchor: crates/vokra-backend-cuda/src/eval.rs -->
7. **Wire the `Compute` seam** in `vokra-models` for the imperative model hot
   path, reusing the *same* kernels — one kernel per (backend, op), never two
   implementations. <!-- anchor: crates/vokra-models/src/compute.rs -->
8. **Keep `FR-EX-08` end-to-end.** Every not-yet-covered op and every
   device-absent path is an explicit error at both the graph seam and the
   `Compute` seam.
9. **Parity against the CPU backend.** The CPU kernel is the numerical oracle;
   a differential test asserts the new backend matches it to `atol = 0.01`
   (`NFR-QL-01`). The parity tolerance is an architectural bound, not a knob —
   see §5.

## 4. Worked example: the most recent backend

The newest backends added are the **CoreML** (Apple ANE) and **QNN** (Qualcomm
Hexagon) *delegates* (`FR-BE-06`). They are the freshest example of the
crate-scaffold steps 1–4 above: `crates/vokra-backend-coreml/`
<!-- anchor: crates/vokra-backend-coreml/src/lib.rs --> is a first-party crate
with `sys.rs` / `probe.rs` / `backend.rs` / `lib.rs`, gated behind a
default-OFF `coreml` feature and target-gated to macOS / iOS.

**A delegate differs from the six-file GPU backends, and the guide says so
honestly.** A delegate hands a declared submodel to the vendor framework and
lets *it* place work onto ANE / GPU / CPU internally; that is not a Vokra-side
op partition (the `Backend` trait's uniform-coverage rule forbids one) and not
a silent fallback. So:

- The canonical **six-file** template is still the five GPU/FFI backends
  (Metal / CUDA / Vulkan / WebGPU) — use those when you add another *kernel*
  backend.
- CoreML / QNN are the template for a *delegate*; their op-execution path lands
  only after the model-supply ADR is ratified, so today every hot op is an
  explicit `UnsupportedOp` and a host with no reachable NPU is an explicit
  `BackendUnavailable`. That is the honest scaffold state, not a bug.

A C-level selector for the delegates is deliberately **not** exported during
the v1.0-rc window; the Rust surface (`with_backend`) is the only way to select
them until the post-bakeoff `IF-01` decision. Consult the on-disk crate before
you cite specifics — this section is re-verified on each backend landing (see
the meta block below).

## 5. Red lines and gotchas

- **NNAPI is permanently unsupported (`FR-BE-07`).** Android GPU is Vulkan;
  the Hexagon NPU is QNN (`FR-BE-06`). Do not add an NNAPI variant.
- **No GPL/LGPL code**, including codecs and resamplers — this holds inside a
  backend too ([CONTRIBUTING.md](../CONTRIBUTING.md) §5).
- **Precompile shaders; no JIT.** The Vulkan backend commits pre-compiled
  SPIR-V rather than compiling GLSL at run time; follow that model.
- **Every `unsafe` block needs a `// SAFETY:` comment**, and `vokra-core`
  itself stays `unsafe`-free — the `unsafe` lives in the backend crate.
- **Never relax a parity tolerance to go green.** `atol` values are
  architectural bounds (`NFR-QL-01`); a failing parity test means the kernel is
  wrong, not that the bound is too tight.

## 6. Owner / contributor boundary

This guide documents the *procedure*. It does **not** run devices: real GPU /
NPU parity and soak on physical hardware (an Apple Neural Engine, a Hexagon
device, an Android phone) are owner tasks. A contributor lands the crate, the
coverage contract and the CPU-oracle parity harness; the owner runs it on the
metal and signs off.

## Keeping this page current

**Last verified: 2026-07-21 — against the five shipping backends + the CoreML /
QNN delegate scaffolds.**

- **Update responsibility**: whoever lands a new backend (or changes the
  six-file layout / the `Backend` trait) updates this page and its Japanese
  twin in the same PR, and refreshes the "verified against" list above.
- **Review cadence**: revisited at the quarterly Go/No-go review (`NFR-MT-05`),
  because this page names backends whose op coverage grows over time.
- **Re-fetch the facts** (crate layout, trait contract, enum variants):

```sh
ls crates/vokra-backend-metal/src/     # the canonical six-file layout
sed -n '/pub trait Backend/,/^}/p' crates/vokra-core/src/backend.rs
sed -n '/pub enum BackendKind/,/^}/p' crates/vokra-core/src/backend.rs
```
