//! # vokra-backend-metal
//!
//! Metal backend for Vokra (FR-BE-02: macOS / iOS) — the second concrete
//! implementation of the `vokra-core` [`Backend`](vokra_core::Backend) trait
//! after `vokra-backend-cpu` (M0-08). This **foundation slice** of M2-01 lands
//! the load-bearing pieces that can be *numerically verified on a real GPU*:
//!
//! - a device / GPU-family probe ([`vokra_metal_probe`]);
//! - an FP32 GEMM compute kernel ([`MetalContext::gemm_f32`]) checked for
//!   parity against the CPU backend (NFR-QL-01, FP32 `atol = 0.01`);
//! - the [`MetalBackend`] trait handle with honest, no-silent-fallback op
//!   coverage (FR-EX-08).
//!
//! The remaining M2-01 kernels (activation / softmax / layer_norm / conv1d /
//! attention / MPS-FFT `stft`·`istft` / `mel_filterbank`), the graph engine,
//! `Session` wiring, benches and the macOS CI job build on top of this slice.
//!
//! # Design record (M2-01-T01/T02, recorded here)
//!
//! Kept in crate docs rather than `docs/adr/` — the same choice
//! `vokra-backend-cpu` made, because the ADR tree is owned by a parallel work
//! package. Fixed decisions, with source IDs:
//!
//! - **(a) crate = `vokra-backend-metal`** (SRS §1.3 `vokra-backend-*`), a
//!   second [`Backend`](vokra_core::Backend) impl reusing the existing IR /
//!   [`OpKind`](vokra_core::OpKind) and the CPU backend as a differential
//!   oracle.
//! - **(b) zero external dependencies, raw FFI** (NFR-DS-02, the M2-01 red
//!   line): **no `metal` / `objc2` / `objc` / `core-foundation` binding
//!   crate.** The Objective-C runtime (`objc_msgSend` / `objc_getClass` /
//!   `sel_registerName`) and Metal / Foundation frameworks are declared inline
//!   in `unsafe extern` blocks and linked with
//!   `#[link(name = "…", kind = "framework")]` ([`sys`]). The root `Cargo.lock`
//!   therefore keeps only `vokra-*` crates (`scripts/check-zero-deps.sh`).
//! - **(c) target-gated, no feature flag needed** (NFR-PT-01): all FFI is
//!   `#[cfg(any(target_os = "macos", target_os = "ios"))]`. Because there is no
//!   external crate to gate, a Cargo *feature* buys nothing here (unlike CUDA's
//!   `cudarc`); `cfg(target_os)` alone keeps Metal / Foundation links out of
//!   Linux / Windows / WASM builds entirely. On those targets [`MetalBackend`]
//!   exists but [`MetalBackend::new`] / [`vokra_metal_probe`] return an explicit
//!   [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable).
//! - **(d) no silent CPU fallback** (FR-EX-08 / NFR-RL-06, the WP's core red
//!   line): an uncovered op is [`VokraError::UnsupportedOp`]; a missing device
//!   is [`VokraError::BackendUnavailable`]. Running on the CPU instead is the
//!   caller's *explicit* backend choice (`Session` wiring is a later ticket),
//!   never decided inside this backend.
//! - **(e) FP32 kernel, no MPS default precision** (NFR-QL-01): the GEMM is a
//!   hand-written MSL `float` kernel compiled at runtime with
//!   `newLibraryWithSource:options:error:` — Vokra does not route this parity
//!   path through MPS/MPSGraph, so there is no implicit FP16 fast path (FP16 /
//!   quantised tiers are M2-08).
//! - **(f) no CPU-side JIT** (NFR-RL-05): `newLibraryWithSource:` compiles GPU
//!   shader code via the Metal driver; the host emits no executable pages. iOS
//!   ships a W^X constraint on CPU pages and prefers a build-time `.metallib`
//!   precompile — that iOS precompile route is a **followup for M2-02** (this
//!   slice is macOS).
//!
//! # Unsafe policy (NFR-RL-07, SRS §5-(1))
//!
//! The Objective-C / Metal FFI needs `unsafe`, so this crate opts out of the
//! workspace-wide `unsafe_code = "deny"` at its root (below). Public APIs stay
//! safe: shapes are validated at the boundary
//! ([`VokraError::InvalidArgument`](vokra_core::VokraError::InvalidArgument) on
//! a mismatch), device / compile failures are `Result` errors (never a panic
//! across the boundary), and **every `unsafe` block carries a `// SAFETY:`
//! comment** (enforced by `clippy::undocumented_unsafe_blocks`). Each
//! `objc_msgSend` transmute names the selector and its true signature ([`sys`]).

// Local opt-out from the workspace `unsafe_code = "deny"` lint — see the
// crate-level "Unsafe policy" docs above (M0-02-T03). The Metal backend joins
// vokra-ops / vokra-backend-cpu / vokra-capi / vokra-mmap on the workspace's
// unsafe-boundary allow list (root Cargo.toml).
#![allow(unsafe_code)]

// Raw Objective-C / Metal / Foundation FFI and the GPU context are Apple-only.
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod context;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod sys;

// The probe and the Backend trait handle exist on every target (with explicit
// BackendUnavailable errors off Apple), so downstream code can always name
// them.
mod backend;
mod probe;

pub use backend::MetalBackend;
pub use probe::{MetalCapabilities, vokra_metal_probe};

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use context::MetalContext;
