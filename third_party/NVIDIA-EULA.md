# NVIDIA CUDA / cuDNN — EULA compliance record

> **2026-10-23 source/terms readback:** [current status](../docs/current-status.md)
> separates source reachability from device and legal completion. The CUDA
> driver probe remains SDK-free; cuDNN detection is not implemented. The
> [current CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html), page dated
> 2026-01-26, has application-only access/distribution conditions and critical
> application limitations. Its distribution section does not contain the
> formerly quoted private-directory wording; do not attribute that quotation
> to the current text. Confirm the exact installed SDK/component terms before
> use or redistribution; this is no compliance certification or new grant.

**Status**: implementation/design boundary for the `vokra-backend-cuda` crate;
it is not a legal-compliance certification or a claim that a critical-safe SKU
has shipped. **Scope**: this file records *how Vokra is intended to remain
within NVIDIA's End User License Agreements*; it is **separate** from Vokra's own
Apache-2.0 license. Vokra bundles no proprietary NVIDIA CUDA/cuDNN runtime,
driver, or SDK code, so those vendor runtime terms are not shipped here.
Separately attributed MIT operator code, such as BigVGAN, is recorded in
`NOTICE` and `crates/vokra-ops/THIRD_PARTY_LICENSES/bigvgan-LICENSE.txt`; that
does not change the proprietary-runtime no-bundle boundary. Cross-referenced
from `NOTICE` and `docs/license-audit.md`.

> **2026-10-08 implementation boundary:** the current CUDA source confirms the
> no-bundle/raw-runtime-FFI and explicit no-silent-fallback design. The current
> foundation slice has no cuDNN dependency or cuDNN runtime detection; the
> optional cuDNN detection/compatibility work remains a planned M2-03-T06
> follow-on. Re-read the applicable NVIDIA terms at release time; this record
> does not itself establish current EULA compliance or authorize distribution.

> Requirement source IDs: FR-BE-08, NFR-LG-04, `docs/milestones.md` §6 M2-03,
> `docs/tickets/m2/M2-03-cuda-backend.md` (T02), `AGENTS.md` memory-safety and
> zero-dependency invariants, `docs/onnx-alternative-research.md` §16.2.

## Authoritative EULA texts

Read the current text before shipping; **do not paraphrase clause numbers from
memory** (they change between toolkit versions — repository no-invention
red line). Confirm the applicable version at implementation / release time:

- **CUDA Toolkit EULA** — <https://docs.nvidia.com/cuda/eula/index.html>
- **cuDNN Software License Agreement (SLA)** — <https://docs.nvidia.com/deeplearning/cudnn/sla/index.html>
- The redistribution terms attached to the specific toolkit version in use
  (the "Attachment A / redistributable" list shipped with that toolkit).

The current CUDA EULA §1.1.2 conditions SDK redistribution on an application
with material additional functionality and access limited to that application;
the CUDA supplement identifies distributable components. §1.2 also limits
critical-application use. The earlier private-directory quotation is not a
verified clause of the current page. Vokra's conservative no-bundle policy
below stays unchanged; it is not a legal conclusion about every SDK version.

## Vokra's compliance strategy — the 5 points (M2-03 WP completion condition)

1. **Bundle nothing.** Vokra does **not** ship, statically link, or otherwise
   redistribute `cudart` / `cudnn` / `cublas` / the driver in any release
   artifact. This policy avoids relying on a proprietary-runtime redistribution
   grant; it does not waive terms applicable to the developer's installed SDK.
   Unity packages (M2-11), including shared-plugin layouts, must likewise ship
   no NVIDIA runtime. Do not infer a directory-specific EULA violation without
   reviewing the applicable component/version terms.
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

3. **Explicit availability probe, no silent fallback.**
   `vokra_cuda_probe()` reports the driver version, device count, device name and
   compute capability. A missing driver or absent GPU is an **explicit
   `VokraError::BackendUnavailable`** — Vokra never silently degrades to the CPU
   backend (FR-EX-08 / NFR-RL-06). CUDA/cuDNN version-compatibility checks are
   not part of this foundation slice; any later compatibility gate requires its
   own implementation and evidence. Selecting the CPU is the caller's explicit
   choice. See `crates/vokra-backend-cuda/src/probe.rs`.

4. **cuDNN is not a current dependency.** The foundation slice ships no cuDNN
   dependency or cuDNN detection path; its current CUDA kernels do not require
   cuDNN. Optional cuDNN detection/optimization is a planned M2-03-T06 follow-on,
   not an implemented or shipped capability. Any future path must preserve the
   no-bundle rule and must not make cuDNN absence a silent CPU fallback.

5. **This file.** The EULA references and the compliance strategy are recorded
   here (self-referential point 5), and `NOTICE` / `docs/license-audit.md` point
   to it. Its presence is checked by CI (M2-03-T23).

## Planned critical-application boundary (critical-safe SKU, FR-BE-09, M5 / v1.0 GA)

The NVIDIA EULA disclaims suitability for critical applications. Vokra's
candidate medical / automotive / military SKU is therefore planned as **CPU +
Vulkan-only** and is **not a shipped SKU or completion claim**. It is out of
scope for M2-03 and belongs to M5 / v1.0 GA (formerly labelled v2.0), pending
the owner/product gates for that release.
Because the entire CUDA path is confined
to the `vokra-backend-cuda` crate + an optional `cuda` feature in `vokra-models`,
a feature-off build already excludes CUDA, leaving room for that future SKU.

## Binding-crate note

Vokra uses **no** `cudarc` / `cust` / `rustacuda` binding crate: the CUDA Driver
API + NVRTC are hand-declared and loaded via dlopen (NFR-DS-02, the zero external
dependency invariant). This keeps the root `Cargo.lock` free of non-`vokra-*`
crates and means there is no third-party CUDA-wrapper license to audit beyond
this file and `docs/license-audit.md`.
