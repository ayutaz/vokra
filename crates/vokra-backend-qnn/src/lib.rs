//! # vokra-backend-qnn
//!
//! QNN **delegate** backend for Vokra (FR-BE-06: Qualcomm Hexagon NPU via the
//! Qualcomm AI Engine Direct SDK, formerly "QNN") — a concrete implementation
//! of the `vokra-core` [`Backend`](vokra_core::Backend) trait, following
//! `vokra-backend-cuda` (M2-03) in its raw-dlopen loader shape and
//! `vokra-backend-coreml` (M5-01) in its **delegate scaffold** shape. This
//! scaffold slice of M5-02 lands the pieces that do **not** depend on the QNN
//! SDK headers being on disk:
//!
//! - a library/symbol probe ([`vokra_qnn_probe`]) that dlopens the QNN runtime
//!   (`libQnnHtp.so` / `QnnHtp.dll` — HTP is the Hexagon backend) and reports
//!   whether it plus a representative interface entry symbol are reachable, and
//!   an explicit
//!   [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable)
//!   when they are not (FR-EX-08 / NFR-RL-06 — never a silent CPU fall back);
//! - the [`QnnBackend`] trait handle whose op coverage is **empty** in this
//!   slice — every op is an explicit
//!   [`VokraError::UnsupportedOp`](vokra_core::VokraError::UnsupportedOp).
//!
//! The op-execution path (QNN graph construction: `QnnGraph_create` →
//! `addNode` → `finalize` → `execute`, op mapping, parity, bench) is **not** in
//! this slice, and is **not owner work** — it is a CC re-issue wave gated on the
//! SDK headers arriving (owner T11 = SDK download + Qualcomm EULA acceptance +
//! real-header layout verification). The acceptance criteria for graph
//! construction cannot be written without the SDK header (struct layout / exact
//! symbol spelling / API version negotiation are UNKNOWN here), which is why
//! that phase is deferred rather than half-written. See
//! `docs/adr/M5-02-qnn-delegate.md` (Proposed) and `docs/handoff/m5-02.md`.
//!
//! # ⚠ Honest constraint — no QNN SDK on the authoring host
//!
//! Unlike M5-01 (CoreML.framework ships with macOS, so the ANE probe is a
//! first-hand measurement), this crate was authored on an Apple M1 host with
//! **no** Qualcomm AI Engine Direct SDK installed. Every QNN library name,
//! symbol name and struct layout below is **from the WP instruction, not
//! first-hand verified**; each is documented as such at its definition. The
//! `sys` module's raw FFI + its compile-time layout assert therefore compile
//! only on the Android / Linux / Windows CI arm (M5-02-T10a) — not on the
//! authoring host — and the layout assert is a *self-consistency guard*, not a
//! real-header check (owner T11 confirms the values against the SDK header, the
//! same verification M3-11's GDExtension asserts did with `clang -m64`, except
//! here there is no header to probe yet).
//!
//! # Design record (M5-02-T01, recorded here — the ADR tree is gitignored)
//!
//! Kept in crate docs, the same choice `vokra-backend-metal` / `-cuda` /
//! `-vulkan` / `-coreml` made. Fixed decisions, with source IDs:
//!
//! - **(a) crate = `vokra-backend-qnn`** (docs/milestones.md §9 M5-02), a
//!   [`Backend`](vokra_core::Backend) impl reusing the existing IR /
//!   [`OpKind`](vokra_core::OpKind).
//! - **(b) zero external dependencies, raw runtime FFI** (NFR-DS-02, the M2-03
//!   red line, inherited): **no `qnn-sys` / `hexagon` / equivalent binding
//!   crate.** The QNN library is loaded at **runtime** with dlopen /
//!   LoadLibrary and each symbol resolved by name ([`sys`]) — the Qualcomm EULA
//!   "install model" (the developer installs the SDK/runtime; Vokra bundles and
//!   links nothing, exactly like the NVIDIA CUDA install model in
//!   `third_party/NVIDIA-EULA.md` → `third_party/QUALCOMM-QNN-NOTES.md`). The
//!   root `Cargo.lock` therefore keeps only `vokra-*` crates
//!   (`scripts/check-zero-deps.sh`), and `deny.toml` bans the QNN/Hexagon
//!   binding-crate families (M5-02-T08).
//! - **(c) NOT NNAPI** (FR-BE-07, permanent): QNN targets the **Qualcomm
//!   Hexagon NPU** through Qualcomm's own SDK. NNAPI — the Android abstraction
//!   Google deprecated in Android 15 — is a *different thing* and is
//!   permanently unsupported. Android's general GPU path is Vulkan (M3-02); QNN
//!   is the Hexagon-NPU delegate. Do not conflate them.
//! - **(d) dlopen-gated + feature-gated** (NFR-PT-01 / NFR-DS-02): the FFI
//!   compiles only on `cfg(all(feature = "qnn", any(target_os = "android",
//!   target_os = "linux", target_os = "windows")))`. Default builds (no `qnn`
//!   feature) and non-QNN targets (macOS / iOS / WASM) reduce to
//!   [`BackendUnavailable`](vokra_core::VokraError::BackendUnavailable) stubs —
//!   never a silent CPU substitute. `vokra-models` gates its optional
//!   dependency on this crate behind its own `qnn` feature (forwarding
//!   `vokra-backend-qnn/qnn`) so default builds never even name it.
//! - **(e) no silent CPU fallback** (FR-EX-08 / NFR-RL-06): an uncovered op is
//!   [`VokraError::UnsupportedOp`]; a missing library / symbol / device is
//!   [`VokraError::BackendUnavailable`]. Running on the CPU instead is the
//!   caller's *explicit* [`BackendKind::Cpu`](vokra_core::BackendKind) choice.
//! - **(f) delegate, not op-partitioning** (FR-BE-06 × the `Backend` trait's
//!   permanent "same op coverage, no ONNX-Runtime EP partitioning" rule): the
//!   intended execution unit is a *declared submodel*. The precise boundary is
//!   inherited from the M5-01-T02 ADR (owner-ratified) and is **not** re-argued
//!   here (M5-01 spec: M5-02 leaves delegate commonisation as a handoff note).
//! - **(g) no CPU-side JIT** (NFR-RL-05, Android SELinux W^X): QNN graph
//!   compilation is the QNN runtime / Hexagon compiler's job; the host emits no
//!   executable pages (same framing as Vulkan SPIR-V → driver and CUDA
//!   PTX → SASS).
//!
//! # Unsafe policy (NFR-RL-07, SRS §5-(1))
//!
//! The QNN dlopen FFI needs `unsafe`, so this crate opts out of the
//! workspace-wide `unsafe_code = "deny"` at its root (below), joining
//! vokra-ops / vokra-backend-cpu / -metal / -cuda / -vulkan / -webgpu /
//! vokra-capi / vokra-mmap on the workspace unsafe-boundary allow list. Public
//! APIs stay safe: library / symbol failures are `Result` errors (never a panic
//! across the boundary), and **every `unsafe` block carries a `// SAFETY:`
//! comment** naming the symbol and its true signature ([`sys`]).

// Local opt-out from the workspace `unsafe_code = "deny"` lint — see the
// crate-level "Unsafe policy" docs above (mirrors vokra-backend-cuda /
// -vulkan). The QNN dlopen FFI is the only `unsafe` in this crate.
#![allow(unsafe_code)]
// The QNN C API uses non-snake-case symbol names (`QnnInterface_getProviders`)
// and `Qnn_*_t` type names; mirror them verbatim in `sys.rs` for
// cross-referencing with the SDK headers (once they exist — owner T11).
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

// The raw QNN dlopen FFI compiles only on Android / Linux / Windows AND only
// when the `qnn` feature is enabled. On other target/feature combinations
// (default builds, macOS / iOS / WASM) the probe / backend below fall back to
// explicit BackendUnavailable stubs — no QNN library link is ever seen.
#[cfg(all(
    feature = "qnn",
    any(target_os = "android", target_os = "linux", target_os = "windows")
))]
mod sys;

// The probe and the Backend trait handle exist on every target (with explicit
// BackendUnavailable / UnsupportedOp errors where the loader / feature / target
// is absent), so downstream code can always name them.
mod backend;
mod probe;

pub use backend::QnnBackend;
pub use probe::{QnnCapabilities, vokra_qnn_probe};
