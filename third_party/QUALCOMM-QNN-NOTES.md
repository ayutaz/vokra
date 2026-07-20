# Qualcomm AI Engine Direct (QNN) — EULA / distribution compliance record

**Status**: normative for the `vokra-backend-qnn` crate and any distribution
that enables QNN (Qualcomm Hexagon NPU). **Scope**: this file records *how Vokra
stays compliant with Qualcomm's license terms for the Qualcomm AI Engine Direct
SDK*; it is **separate** from Vokra's own Apache-2.0 license (Vokra bundles no
Qualcomm code, so no Qualcomm license text is vendored here). It is the
authoritative reference for the `vokra-backend-qnn` crate, modelled on
`third_party/NVIDIA-EULA.md` (the CUDA install-model precedent). The `NOTICE`
and `docs/license-audit.md` distribution entries land when the QNN backend ships
a functional artifact (the SDK-gated re-issue wave), mirroring how the NVIDIA
NOTICE entry landed with M2-03 substance rather than at scaffold time — this
scaffold ships no functional QNN artifact.

> Requirement source IDs: FR-BE-06, NFR-PF-12, NFR-DS-02, NFR-PT-01, NFR-LG-04,
> FR-EX-08, `docs/milestones.md` §9 M5-02, CLAUDE.md backend-priority + zero-dep
> sections. **Not NNAPI** (FR-BE-07): QNN is the Qualcomm Hexagon NPU delegate
> reached through Qualcomm's own SDK; NNAPI — the Android abstraction Google
> deprecated in Android 15 — is permanently unsupported and is a different thing.

## Authoritative license texts (owner confirms at ship time)

Read the current text before shipping; **do not paraphrase clause numbers or
terms from memory** (they change between SDK versions — CLAUDE.md hallucination
red line, the same rule NVIDIA-EULA.md applies). Confirm the applicable license
and version at implementation / release time from the SDK you actually install:

- The **Qualcomm AI Engine Direct SDK** license / terms as shipped with the
  specific SDK version in use (accept during owner T11 = SDK download + EULA
  acceptance). The exact URL / document name is owner-verified — this scaffold
  was authored on a host with **no** SDK, so no term is transcribed here.
- Any redistribution terms attached to that SDK version (whether the runtime
  libraries may be redistributed, and under what directory / packaging
  constraints) — owner confirms; Vokra's design (below) avoids relying on any
  redistribution grant by bundling nothing.

## Vokra's compliance strategy — the 5 points (M5-02 posture)

1. **Bundle nothing.** Vokra does **not** ship, statically link, or otherwise
   redistribute `libQnnHtp` / any Qualcomm AI Engine Direct runtime library in
   any release artifact. There is nothing to place, so no redistribution grant
   or directory-placement constraint has to be satisfied. In particular, the
   Unity / Godot shared-plugins layout must also ship no QNN runtime (the same
   constraint the NVIDIA runtime is held to — `scripts/compliance/`).

2. **Runtime detection (developer install model).** The developer installs the
   Qualcomm AI Engine Direct SDK / runtime; Vokra detects it at runtime with
   `dlopen("libQnnHtp.so")` / `LoadLibrary("QnnHtp.dll")` (or a path supplied
   via `VOKRA_QNN_LIB`). No build-time link against Qualcomm libraries — which
   is also why the crate **compiles on a host with no QNN at all** (e.g. the
   Apple Mac this slice was authored on: the FFI is target-gated to Android /
   Linux / Windows and the probe returns `BackendUnavailable`). See
   `crates/vokra-backend-qnn/src/sys.rs`.

3. **Explicit availability probe, no silent fallback.** `vokra_qnn_probe()`
   loads the QNN library and checks a representative interface entry symbol
   resolves. A missing library, a missing symbol, or an off-target/feature-off
   build is an **explicit `VokraError::BackendUnavailable`** — Vokra never
   silently degrades to the CPU backend (FR-EX-08 / NFR-RL-06). Selecting the
   CPU is the caller's explicit choice. See
   `crates/vokra-backend-qnn/src/probe.rs`.

4. **The SDK is never a required build dependency.** The QNN backend is an
   optional `qnn` feature (default off) in `vokra-models`; a feature-off build
   excludes it entirely, and no non-`vokra-*` crate is added to `Cargo.lock`
   (NFR-DS-02). No `qnn-sys` / `hexagon` / Qualcomm wrapper crate is used — the
   FFI is hand-declared dlopen (deny.toml bans those families, M5-02-T08).

5. **This file.** The license references and the compliance strategy are
   recorded here (self-referential point 5), and `NOTICE` / `docs/license-audit.md`
   point to it.

## Critical-application posture (critical-safe SKU, FR-BE-09 / M5-08)

The critical-safe (medical / automotive / military) SKU is **CPU + Vulkan-only**
and **excludes QNN** — the `qnn` feature being off by default is exactly what
makes that exclusion automatic (the same way the CPU + Vulkan-only build target
excludes metal / cuda). Whether a Qualcomm SDK term additionally disclaims
critical-application suitability is owner-confirmed at ship time; either way QNN
is out of the critical-safe SKU.

## Binding-crate note

Vokra uses **no** `qnn` / `qnn-sys` / `hexagon` binding crate: the QNN runtime is
hand-declared and loaded via dlopen (NFR-DS-02, the zero external dependency
invariant). This keeps the root `Cargo.lock` free of non-`vokra-*` crates and
means there is no third-party QNN-wrapper license to audit beyond this file and
`docs/license-audit.md`.

## What is NOT done in this scaffold (SDK-gated re-issue wave, owner T11 gates)

The scaffold probes reachability only. QNN graph construction (`QnnGraph_create`
→ `addNode` → `finalize` → `execute`), the op-mapping table, struct/enum layout
verification against the real SDK header, and the "execution ran on the Hexagon
NPU (not a CPU/GPU backend fallback)" device readout all require the SDK header
on disk and are a **CC re-issue wave gated on owner T11** — not owner
implementation work. See `docs/handoff/m5-02.md`.
