//! # vokra-backend-cuda
//!
//! CUDA backend for Vokra (FR-BE-03: Windows / Linux NVIDIA GPUs) — the third
//! concrete implementation of the `vokra-core` [`Backend`](vokra_core::Backend)
//! trait after `vokra-backend-cpu` (M0-08) and `vokra-backend-metal` (M2-01).
//! This **foundation slice** of M2-03 lands the load-bearing pieces that can be
//! *numerically verified on a real GPU* (a vast.ai RTX 4090 — this crate is
//! authored on an Apple Mac that has no NVIDIA GPU):
//!
//! - a driver / device probe ([`vokra_cuda_probe`]);
//! - an FP32 GEMM compute kernel ([`CudaContext::gemm_f32`], NVRTC-compiled PTX)
//!   with the exact shape/semantics contract of the CPU backend's
//!   `kernels::gemm_f32` (checked for parity, NFR-QL-01, FP32 `atol = 0.01`);
//! - the [`CudaBackend`] trait handle with honest, no-silent-fallback op
//!   coverage (FR-EX-08).
//!
//! The remaining M2-03 tickets (cuBLAS GEMM, candle-kernels wiring for
//! elementwise / activation / softmax / layer-norm, cuDNN-free conv1d,
//! FlashAttention **v2** — *FA v3 is pushed to v1.5+ and must not be
//! implemented* — GPU KV cache, CUDA Graph capture/replay, Whisper-base CUDA
//! wiring, the C-ABI probe and the self-hosted GPU CI) build on top of this
//! slice.
//!
//! # Design record (M2-03-T01, recorded here)
//!
//! Kept in crate docs rather than `docs/adr/` — the same choice
//! `vokra-backend-metal` made. Fixed decisions, with source IDs:
//!
//! - **(a) zero external dependencies, raw runtime FFI** (NFR-DS-02, the M2-03
//!   red line): **no `cudarc` / `cust` / `rustacuda` binding crate.** The CUDA
//!   Driver API and NVRTC are loaded at **runtime** with dlopen / LoadLibrary
//!   and each symbol `transmute`d to its exact C signature ([`sys`]). The root
//!   `Cargo.lock` therefore keeps only `vokra-*` crates
//!   (`scripts/check-zero-deps.sh`). This is a deliberate departure from the
//!   original M2-03 plan of using `cudarc` behind a `cuda` feature: the raw-FFI
//!   route keeps the **default and CUDA builds alike dependency-free**.
//! - **(b) NVIDIA EULA install model** (FR-BE-08, `third_party/NVIDIA-EULA.md`):
//!   Vokra bundles/statically-links **no** `cudart` / `cudnn` / `cublas` / the
//!   driver. The developer installs CUDA system-wide; the runtime detects it via
//!   `dlopen("libcuda.so.1")` / `LoadLibrary("nvcuda.dll")`. A missing library
//!   is a runtime [`VokraError::BackendUnavailable`], not a build error — which
//!   is also why this crate **compiles on a CUDA-less host** (the whole
//!   all-target build stays intact, NFR-PT-01).
//! - **(c) dlopen-gated, not feature-gated inside this crate** (NFR-PT-01): the
//!   FFI compiles on every `cfg(any(unix, windows))` target and is a runtime
//!   no-op elsewhere (WASM), where [`CudaBackend::new`] / [`vokra_cuda_probe`]
//!   return an explicit [`VokraError::BackendUnavailable`]. `vokra-models` still
//!   gates *its* optional dependency on this crate behind a `cuda` feature so
//!   Linux/Windows/WASM default builds never even name it.
//! - **(d) no silent CPU fallback** (FR-EX-08 / NFR-RL-06, the WP's core red
//!   line): an uncovered op is [`VokraError::UnsupportedOp`]; a missing driver /
//!   device is [`VokraError::BackendUnavailable`]. Running on the CPU instead is
//!   the caller's *explicit* backend choice, never decided inside this backend.
//! - **(e) FP32 kernel, no Tensor-Core fast path** (NFR-QL-01): the GEMM is a
//!   hand-written CUDA C `float` kernel; there is no implicit TF32/FP16 path
//!   (FP16 / quantised tiers are M2-08).
//! - **(f) device-side JIT only, no CPU codegen** (NFR-RL-05): NVRTC compiles
//!   GPU PTX; the host emits no executable CPU pages.
//! - **(g) FlashAttention v2 only** (FR-BE-03, CLAUDE.md): the (future)
//!   attention kernel is FA **v2** CPU-port based; **FA v3 (WGMMA/TMA, Hopper)
//!   is pushed to v1.5+ and must not be implemented** in this WP.
//!
//! # Unsafe policy (NFR-RL-07, SRS §5-(1))
//!
//! The CUDA / NVRTC FFI needs `unsafe`, so this crate opts out of the
//! workspace-wide `unsafe_code = "deny"` at its root (below). Public APIs stay
//! safe: shapes are validated at the boundary
//! ([`VokraError::InvalidArgument`](vokra_core::VokraError::InvalidArgument) on
//! a mismatch), driver / compile failures are `Result` errors (never a panic
//! across the boundary), and **every `unsafe` block carries a `// SAFETY:`
//! comment** (enforced by `clippy::undocumented_unsafe_blocks`). Each `dlsym`
//! `transmute` names the symbol and its true signature ([`sys`]).

// Local opt-out from the workspace `unsafe_code = "deny"` lint — see the
// crate-level "Unsafe policy" docs above (M0-02-T03). The CUDA backend joins
// vokra-ops / vokra-backend-cpu / vokra-backend-metal / vokra-capi / vokra-mmap
// on the workspace's unsafe-boundary allow list (root Cargo.toml).
#![allow(unsafe_code)]

// M2-02 iOS build red line (belt-and-suspenders against the `cfg(unix)` loophole
// noted in the iOS build ADR §1.3): `target_os = "ios"` satisfies `cfg(unix)`,
// so `vokra-models/Cargo.toml`'s `cfg(any(unix, windows))` CUDA gate would
// otherwise let `--features cuda` reach an iOS target. CUDA is absent on iOS
// (no libcuda, no NVRTC) and user-side `dlopen` is App-Store-forbidden — turn
// this into a compile-time failure rather than a runtime dlopen crash (R1).
// FR-EX-08: explicit error, no silent fallback. NFR-RL-03: iOS is static-only.
#[cfg(target_os = "ios")]
compile_error!(
    "vokra-backend-cuda cannot be built for iOS; do not pass --features cuda on iOS targets."
);

// The raw CUDA / NVRTC FFI and the GPU context need a dynamic loader (dlopen /
// LoadLibrary), so they compile only on Unix / Windows. On other targets (WASM)
// the probe / backend below fall back to explicit BackendUnavailable stubs.
#[cfg(any(unix, windows))]
mod context;
#[cfg(any(unix, windows))]
mod eval;
// M4-07: FlashAttention v3 (Hopper WGMMA, sm_90a). The ONLY module tree where
// FA v3 code is legal (design constraint §5-(7) unlock point; containment is
// machine-checked by scripts/check-fa-v3-confinement.sh). Kept as a separate
// NVRTC program from `context::KERNELS_CUDA` — see fa_v3.rs module docs.
#[cfg(any(unix, windows))]
mod fa_v3;
#[cfg(any(unix, windows))]
pub mod session_pool;
#[cfg(any(unix, windows))]
mod sys;

// The probe and the Backend trait handle exist on every target (with explicit
// BackendUnavailable errors where the loader is absent), so downstream code can
// always name them.
mod backend;
mod probe;

pub use backend::CudaBackend;
pub use probe::{CudaCapabilities, vokra_cuda_probe};

#[cfg(any(unix, windows))]
pub use context::{CudaContext, CudaKvCache};
// M4-07 diagnostic / test surface (doc(hidden) — not a supported public API):
// the arch-explicit NVRTC compile entry the `compute_90a` feasibility test
// drives, and the FA v3 kernel sources it compiles. NVRTC needs no GPU, only
// the toolkit library, so the test can run on any CUDA-toolkit host.
#[cfg(any(unix, windows))]
#[doc(hidden)]
pub use context::nvrtc_compile_for_arch;
#[cfg(any(unix, windows))]
#[doc(hidden)]
pub use fa_v3::{FA_V3_FEASIBILITY_SNIPPET, KERNELS_CUDA_FA_V3};
// M4-07-T08: the FA v3 scalar-geometry validator, public so the negative
// (input-validation) tests stay green on CUDA-less hosts.
#[cfg(any(unix, windows))]
pub use fa_v3::flash_attn_v3_validate_args;
// `CudaDecodeSession` is the M2 Phase-3b device-resident decoder-step driver
// (the CUDA sibling of `vokra-backend-metal`'s `MetalDecodeSession`); re-exported
// here so `vokra-models`' `Compute::new_decoder_step_session` (its Cuda arm)
// can build it.
#[cfg(any(unix, windows))]
pub use context::CudaDecodeSession;
