//! # vokra-backend-coreml
//!
//! CoreML **delegate** backend for Vokra (FR-BE-06: Apple ANE) — a concrete
//! implementation of the `vokra-core` [`Backend`](vokra_core::Backend) trait,
//! following `vokra-backend-metal` (M2-01) in shape. This **scaffold slice** of
//! M5-01 lands the pieces that do not depend on the model-supply decision:
//!
//! - an ANE-aware compute-device probe ([`vokra_coreml_probe`]) built on the
//!   public `MLAllComputeDevices()` API (macOS 14.0+ / iOS 17.0+), so "is the
//!   Apple Neural Engine reachable, and with how many cores" is answered from
//!   the framework rather than assumed;
//! - the [`CoreMlBackend`] trait handle whose op coverage is **empty** in this
//!   slice — every op is an explicit
//!   [`VokraError::UnsupportedOp`](vokra_core::VokraError::UnsupportedOp), never
//!   a silent CPU fall back (FR-EX-08). Coverage grows once the execution path
//!   lands.
//!
//! The op-execution path (model → `MLModel` → ANE, parity, bench) is **not** in
//! this slice: it turns on how a CoreML artifact is supplied to the runtime,
//! which is the subject of the M5-01-T02 ADR (owner-ratified). Until that ADR
//! is accepted, this crate deliberately stops at the probe + an
//! honest-empty-coverage handle. See `docs/adr/M5-01-coreml-delegate.md`.
//!
//! # Design record (M5-01-T01/T03, recorded here)
//!
//! Kept in crate docs — the same choice `vokra-backend-metal` made, because the
//! ADR tree is owned by a parallel work package. Fixed decisions, with source
//! IDs:
//!
//! - **(a) crate = `vokra-backend-coreml`** (SRS §1.3 `vokra-backend-*`), a
//!   [`Backend`](vokra_core::Backend) impl reusing the existing IR /
//!   [`OpKind`](vokra_core::OpKind).
//! - **(b) zero external dependencies, raw FFI** (NFR-DS-02, the M2-01 red
//!   line, inherited): **no `objc` / `objc2` / `objc2-core-ml` /
//!   `core-foundation` binding crate.** The Objective-C runtime and CoreML /
//!   Foundation frameworks are declared inline in `unsafe extern` blocks and
//!   linked with `#[link(name = "…", kind = "framework")]` ([`sys`]). The root
//!   `Cargo.lock` keeps only `vokra-*` crates (`scripts/check-zero-deps.sh`),
//!   and `deny.toml` bans the ObjC/CoreML binding-crate families (M5-01-T08).
//! - **(c) target-gated, feature-flagged off by default** (NFR-PT-01): all FFI
//!   is `#[cfg(any(target_os = "macos", target_os = "ios"))]`, and the crate is
//!   only pulled into `vokra-models` behind its `coreml` feature (default off,
//!   symmetric with `metal` / `cuda` / `vulkan`). On non-Apple targets
//!   [`CoreMlBackend`] exists but [`CoreMlBackend::new`] / [`vokra_coreml_probe`]
//!   return an explicit
//!   [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable).
//! - **(d) no silent CPU fallback** (FR-EX-08 / NFR-RL-06): an uncovered op is
//!   [`VokraError::UnsupportedOp`]; a missing ANE / device is
//!   [`VokraError::BackendUnavailable`]. Running on the CPU instead is the
//!   caller's *explicit* [`BackendKind::Cpu`](vokra_core::BackendKind) choice,
//!   never decided inside this backend.
//! - **(e) delegate, not op-partitioning** (FR-BE-06 × the `Backend` trait's
//!   permanent "same op coverage, no ONNX-Runtime EP partitioning" constraint):
//!   the intended execution unit is a *declared submodel*, and CoreML's own
//!   choice of ANE / GPU / CPU inside that submodel is Apple's runtime concern,
//!   not a Vokra-side silent fallback. The precise boundary is a T02 ADR point
//!   and is **not** fixed by this scaffold.
//! - **(f) no CPU-side JIT** (NFR-RL-05): CoreML model compilation is the OS
//!   framework's job; the host emits no executable pages (same framing as
//!   Metal's `newLibraryWithSource:` and Vulkan's SPIR-V → driver compile).
//!
//! # Unsafe policy (NFR-RL-07, SRS §5-(1))
//!
//! The Objective-C / CoreML FFI needs `unsafe`, so this crate opts out of the
//! workspace-wide `unsafe_code = "deny"` at its root (below), joining
//! vokra-ops / vokra-backend-cpu / vokra-backend-metal / vokra-capi / vokra-mmap
//! on the workspace's unsafe-boundary allow list. Public APIs stay safe:
//! device / availability failures are `Result` errors (never a panic across the
//! boundary), and **every `unsafe` block carries a `// SAFETY:` comment** naming
//! the selector and its true signature ([`sys`]).

// Local opt-out from the workspace `unsafe_code = "deny"` lint — see the
// crate-level "Unsafe policy" docs above (mirrors vokra-backend-metal).
#![allow(unsafe_code)]

// Raw Objective-C / CoreML / Foundation FFI is Apple-only.
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod sys;

// The probe and the Backend trait handle exist on every target (with explicit
// BackendUnavailable / UnsupportedOp errors off Apple), so downstream code can
// always name them.
mod backend;
mod probe;

pub use backend::CoreMlBackend;
pub use probe::{CoreMlCapabilities, vokra_coreml_probe};
