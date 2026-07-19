# NVIDIA CUDA / cuDNN — EULA compliance record

**Status**: normative for the `vokra-backend-cuda` crate and any distribution
that enables CUDA. **Scope**: this file records *how Vokra stays compliant with
NVIDIA's End User License Agreements*; it is **separate** from Vokra's own
Apache-2.0 license (Vokra bundles no NVIDIA code, so no NVIDIA license text is
vendored here). Cross-referenced from `NOTICE` and `docs/license-audit.md`.

> Requirement source IDs: FR-BE-08, NFR-LG-04, `docs/milestones.md` §6 M2-03,
> `docs/tickets/m2/M2-03-cuda-backend.md` (T02), CLAUDE.md
> "NVIDIA CUDA / cuDNN EULA 準拠", `docs/onnx-alternative-research.md` §16.2.

## Authoritative EULA texts

Read the current text before shipping; **do not paraphrase clause numbers from
memory** (they change between toolkit versions — CLAUDE.md hallucination
red line). Confirm the applicable version at implementation / release time:

- **CUDA Toolkit EULA** — <https://docs.nvidia.com/cuda/eula/index.html>
- **cuDNN Software License Agreement (SLA)** — <https://docs.nvidia.com/deeplearning/cudnn/sla/index.html>
- The redistribution terms attached to the specific toolkit version in use
  (the "Attachment A / redistributable" list shipped with that toolkit).

The load-bearing constraint Vokra designs around is the CUDA EULA's requirement
that redistributed NVIDIA runtime libraries be *"installed only in a private
(non-shared) directory location"*, plus the *"not tested … for use in … critical
applications"* disclaimer. Vokra's response to both is below.

## Vokra's compliance strategy — the 5 points (M2-03 WP completion condition)

1. **Bundle nothing.** Vokra does **not** ship, statically link, or otherwise
   redistribute `cudart` / `cudnn` / `cublas` / the driver in any release
   artifact. This sidesteps the "private (non-shared) directory" redistribution
   requirement entirely — there is nothing to place. In particular, the Unity
   Asset Store's *shared plugins directory* layout is a plausible EULA violation
   for a bundled NVIDIA runtime, so the Unity package (M2-11) must also ship no
   NVIDIA runtime (a constraint shared with M2-11).
   *Enforced by CI (M2-03-T23): a distribution scan fails the build if any
   `libcudart*` / `libcudnn*` / `libcublas*` / `cudart64_*.dll` / `cudnn*.dll` /
   `cublas*.dll` (or a static equivalent) is present in a release artifact.*

2. **Runtime detection (developer install model).** The developer installs the
   CUDA driver / toolkit system-wide; Vokra detects it at runtime with
   `dlopen("libcuda.so.1")` / `LoadLibrary("nvcuda.dll")` (and, for the
   NVRTC runtime compiler, `libnvrtc` / `nvrtc64_*.dll`). No build-time link
   against NVIDIA libraries — which is also why the crate **compiles on a host
   with no CUDA at all** (e.g. an Apple Mac). See
   `crates/vokra-backend-cuda/src/sys.rs`.

3. **Explicit version/availability probe, no silent fallback.**
   `vokra_cuda_probe()` reports the driver version, device count, device name and
   compute capability. A missing driver, an absent GPU, or (in later M2-03
   tickets) an incompatible CUDA/cuDNN version is an **explicit
   `VokraError::BackendUnavailable`** — Vokra never silently degrades to the CPU
   backend (FR-EX-08 / NFR-RL-06). Selecting the CPU is the caller's explicit
   choice. See `crates/vokra-backend-cuda/src/probe.rs`.

4. **cuDNN is never a required dependency.** Audio models are implemented with an
   op set that does **not** require cuDNN (e.g. `conv1d` via im2col + GEMM). cuDNN
   is used only as an *optional* optimisation where detected; its absence must
   never fail the CUDA backend. (This foundation slice ships no cuDNN dependency
   at all; the cuDNN-optional detection is M2-03-T06.)

5. **This file.** The EULA references and the compliance strategy are recorded
   here (self-referential point 5), and `NOTICE` / `docs/license-audit.md` point
   to it. Its presence is checked by CI (M2-03-T23).

## Critical-application disclaimer (critical-safe SKU, FR-BE-09, M5 / v1.0 GA)

The NVIDIA EULA disclaims suitability for critical applications. Vokra's
medical / automotive / military SKU is therefore **CPU + Vulkan-only** and is
**out of scope for M2-03 (it belongs to M5 / v1.0 GA, formerly labelled v2.0)**.
Because the entire CUDA path is confined
to the `vokra-backend-cuda` crate + an optional `cuda` feature in `vokra-models`,
a feature-off build already excludes CUDA, leaving room for that future SKU.

## Binding-crate note

Vokra uses **no** `cudarc` / `cust` / `rustacuda` binding crate: the CUDA Driver
API + NVRTC are hand-declared and loaded via dlopen (NFR-DS-02, the zero external
dependency invariant). This keeps the root `Cargo.lock` free of non-`vokra-*`
crates and means there is no third-party CUDA-wrapper license to audit beyond
this file and `docs/license-audit.md`.
