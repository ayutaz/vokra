//! # vokra-backend-webgpu
//!
//! WebGPU backend for Vokra (FR-BE-05: browser WASM; "Unity WebGL 差別化")
//! — the fifth concrete implementation of the `vokra-core`
//! [`Backend`](vokra_core::Backend) trait after CPU (M0-08), Metal (M2-01),
//! CUDA (M2-03) and Vulkan (M3-02/M4-13).
//!
//! # Design record (M4-01-T01/T02; full ADR: docs/adr/M4-01-webgpu-wasm.md,
//! gitignored-local — the load-bearing decisions are mirrored here the way
//! the Metal / CUDA / Vulkan crates carry theirs)
//!
//! - **(a) Integration = raw extern-import shim, NO `wgpu` crate**
//!   (NFR-DS-02): [`sys`] declares a
//!   `#[link(wasm_import_module = "vokra_webgpu")]` import surface and the
//!   hand-written JS glue (`glue/vokra_webgpu.js`, zero npm dependencies)
//!   satisfies it against the browser `navigator.gpu` API. Import-object
//!   resolution at instantiate time is the WASM equivalent of the dlopen
//!   runtime-linking model the other GPU backends use (there is no dlopen in
//!   a browser). `deny.toml` bans the `wgpu` and `wasm-bindgen` crate
//!   families (ADR M3-02 §4-(d) wording, wired by M4-01). The "wgpu" wording
//!   in CLAUDE.md / FR-BE-05 / milestones §8 is a documented divergence —
//!   owner approval + doc follow-up is the T02 hand-over.
//! - **(b) Sync bridge = Worker + SharedArrayBuffer + `Atomics.wait`**:
//!   WebGPU readback (`mapAsync`) is async-only; the inference wasm runs in
//!   a dedicated Web Worker whose glue forwards GPU calls to a main-thread
//!   proxy over a SAB command channel and blocks on `Atomics.wait`
//!   (worker-legal; main thread never waits). COOP/COEP is REQUIRED for SAB
//!   — absence is an explicit init error pointing at the deployment doc
//!   (FR-EX-08, no silent degradation). Readback consolidates at run
//!   boundaries (the M2-01 6N+1 → 1 lesson).
//! - **(c) SIMD128 = 2-artifact distribution** (`scripts/build-wasm.sh` +
//!   loader `WebAssembly.validate` probe): WASM has no runtime feature
//!   detection. Relaxed SIMD not adopted (Safari-partial; relaxed-fma
//!   nondeterminism vs NFR-QL-01).
//! - **(d) WGSL text + `include_str!` is NOT host JIT** (NFR-RL-05): the Web
//!   standard has no binary shader format; `createShaderModule` compilation
//!   is the browser/driver's responsibility (M4-13's driver-compile
//!   separation). Drift gate = SHA-256 source pins ([`wgsl`]).
//! - **(e) No silent CPU fallback** (FR-EX-08 / NFR-RL-06): missing adapter
//!   = [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable);
//!   uncovered op = [`VokraError::UnsupportedOp`](vokra_core::VokraError::UnsupportedOp).
//!   The WASM CPU (SIMD128) path is only ever the caller's explicit
//!   `BackendKind::Cpu` choice.
//! - **(f) FP32 storage/accumulator fixed** in every WGSL kernel
//!   (NFR-QL-01; BF16-mantissa rule); FA v3 is forbidden here (M4-07 only) —
//!   attention stays standard GEMM + softmax.
//!
//! # Unsafe policy (NFR-RL-07)
//!
//! The extern-import calls are `unsafe`, so this crate opts out of the
//! workspace-wide `unsafe_code = "deny"` (below), joining the other backend
//! crates on the unsafe-boundary allow list. Public APIs stay safe
//! (`Result` errors, never a panic across the boundary), and **every
//! `unsafe` block carries a `// SAFETY:` comment** citing the [`sys`] import
//! contract (enforced by `clippy::undocumented_unsafe_blocks`).

// Local opt-out from the workspace `unsafe_code = "deny"` lint — see the
// crate-level "Unsafe policy" docs above (M0-02-T03).
#![allow(unsafe_code)]

// The extern-import shim and everything that calls through it compile only
// for wasm32 builds with the `webgpu` feature; all other target/feature
// combinations reduce to explicit BackendUnavailable stubs (FR-EX-08).
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
mod context;
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
mod eval;
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
mod sys;

// The probe, the Backend handle, the host-portable plans and the WGSL
// manifest exist on every target so downstream code (and the native test
// suite) can always name them.
mod backend;
pub mod plan;
mod probe;
pub mod wgsl;

pub use backend::{WebGpuBackend, graph_op_backing_shader};
pub use probe::{WebGpuCapabilities, vokra_webgpu_probe};

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
pub use context::WebGpuContext;
