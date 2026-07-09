//! # vokra-backend-vulkan
//!
//! Vulkan backend for Vokra (FR-BE-04: Android / Linux subgroup +
//! cooperative-matrix; also Windows non-NVIDIA / Intel Arc) — the fourth
//! concrete implementation of the `vokra-core`
//! [`Backend`](vokra_core::Backend) trait after `vokra-backend-cpu` (M0-08),
//! `vokra-backend-metal` (M2-01), and `vokra-backend-cuda` (M2-03).
//!
//! This **foundation slice** of M3-02 lands the load-bearing pieces that can
//! be *verified without a Vulkan-capable runtime* on the authoring host
//! (Apple Mac uses Metal; there is no Vulkan loader here):
//!
//! - a Vulkan loader / device probe ([`vokra_vulkan_probe`]) that returns an
//!   honest capability struct on a Vulkan host, and an explicit
//!   [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable)
//!   elsewhere (FR-EX-08 / NFR-RL-06 — never a silent CPU fall back);
//! - the [`VulkanBackend`] trait handle with honest, no-silent-fallback op
//!   coverage — no SPIR-V kernel ships yet, so every op reports
//!   [`VokraError::UnsupportedOp`](vokra_core::VokraError::UnsupportedOp)
//!   until T14〜T22 land;
//! - a `VkInstance` create + destroy round-trip through the loader (T06 +
//!   T07 partial).
//!
//! The remaining M3-02 tickets (SPIR-V shaders for GEMM /
//! GEMV / softmax / softmax_causal / layer_norm / gelu / conv1d / elementwise
//! / activation / shape-ops with subgroup + cooperative-matrix fallback path
//! — T14〜T22; graph-executor Vulkan arm — T26〜T29; Whisper-base parity —
//! T32〜T34; CI wiring — T36〜T38) build on top of this slice. Android arm64
//! cross-build (T37) and Android real-device RTF (T39〜T40, owner run) are
//! future work.
//!
//! # Design record (M3-02-T01, recorded here)
//!
//! Kept in crate docs rather than `docs/adr/` — the same choice
//! `vokra-backend-metal` and `vokra-backend-cuda` made (the ADR tree is
//! gitignored in this repo). Fixed decisions, with source IDs:
//!
//! - **(a) crate = `vokra-backend-vulkan`** (docs/milestones.md §7.2 M3-02
//!   main deliverable). A fourth [`Backend`](vokra_core::Backend) impl reusing
//!   the existing IR / [`OpKind`](vokra_core::OpKind) and the CPU backend as a
//!   differential oracle (M0-08).
//! - **(b) zero external dependencies, raw runtime FFI** (NFR-DS-02, the
//!   M3-02 red line): **no `ash` / `vulkano` / `erupt` / `gpu-alloc` binding
//!   crate.** The Vulkan loader is loaded at **runtime** with dlopen /
//!   LoadLibrary and each symbol `transmute`d to its exact C signature
//!   ([`sys`]). The root `Cargo.lock` therefore keeps only `vokra-*` crates
//!   (`scripts/check-zero-deps.sh`).
//! - **(c) Android GPU path is Vulkan only; NNAPI is permanently
//!   unsupported** (FR-BE-07, project design constraint 8, README §4 (2)):
//!   Google deprecated NNAPI in Android 15 (2024-10). Vokra pins the Android
//!   GPU path to Vulkan 1.1+ from day one. `deny.toml` (M0-02) bans
//!   `android-ndk-sys`'s NNAPI shim family via cargo-deny (T38).
//! - **(d) dlopen-gated + feature-gated** (NFR-PT-01 / NFR-DS-02): the FFI
//!   compiles on `cfg(all(feature = "vulkan", any(target_os = "linux",
//!   target_os = "android", target_os = "windows")))`. Default builds (no
//!   `vulkan` feature) and non-Vulkan targets (macOS / iOS / WASM) reduce to
//!   [`BackendUnavailable`](vokra_core::VokraError::BackendUnavailable) stubs
//!   — never a silent CPU substitute. `vokra-models` will gate *its* optional
//!   dependency on this crate behind a `vulkan` feature so default builds
//!   never even name it.
//! - **(e) no silent CPU fallback** (FR-EX-08 / NFR-RL-06, the WP's core red
//!   line): an uncovered op is
//!   [`VokraError::UnsupportedOp`](vokra_core::VokraError::UnsupportedOp);
//!   a missing loader / device is
//!   [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable).
//!   Running on the CPU instead is the caller's *explicit* backend choice,
//!   never decided inside this backend.
//! - **(f) FP32 shader path, no FP16 / cooperative-matrix fast path in the
//!   parity gate** (NFR-QL-01): SPIR-V shaders keep their storage class at
//!   FP32 for the numerical-parity CI (T33/T34). FP16 fast paths / layer-wise
//!   quantisation policy (FR-QT-02/03) are M4+.
//! - **(g) no CPU-side JIT — SPIR-V precompiled at build time** (NFR-RL-05,
//!   Android SELinux W^X constraint): `.spv` blobs are produced by `glslc`
//!   (Vulkan SDK, developer-side) and committed to the repo as
//!   `kernels/precompiled/*.spv`, then embedded via `include_bytes!` (T13).
//!   The GPU driver's own SPIR-V → GPU ISA translation is the driver's
//!   responsibility, not host JIT. `build.rs` verifies existence of every
//!   expected `.spv` and does NOT invoke `glslc` at build time (keeps
//!   `cargo build` dependency-free — no `spirv-tools` crate).
//! - **(h) subgroup + cooperative-matrix precompiled as two `.spv` per op
//!   where relevant** (M3-02-T14 GEMM): a `_coopmat.spv` and a
//!   `_subgroup.spv` for GEMM; the probe (T30/T31) determines which pipeline
//!   is bound at run-time. This is capability-driven pipeline selection, not
//!   silent-fallback op behaviour — the op still runs, just with the shader
//!   the hardware actually supports.
//! - **(i) FlashAttention v3 forbidden here** (project constraint 7): Vulkan
//!   has no Hopper WGMMA/TMA equivalent, but the attention path in this WP
//!   is standard GEMM + softmax anyway; FA v3 is v1.5+.
//!
//! # Unsafe policy (NFR-RL-07, SRS §5-(1))
//!
//! The Vulkan FFI needs `unsafe`, so this crate opts out of the
//! workspace-wide `unsafe_code = "deny"` at its root (below). Public APIs
//! stay safe: driver load / instance create / device probe failures are
//! `Result` errors (never a panic across the boundary), and **every
//! `unsafe` block carries a `// SAFETY:` comment** (enforced by
//! `clippy::undocumented_unsafe_blocks`). Each `dlsym` /
//! `vkGetInstanceProcAddr` `transmute` pairs the C symbol name with the
//! exact `Fn*` alias declared in [`sys`].

// Local opt-out from the workspace `unsafe_code = "deny"` lint — see the
// crate-level "Unsafe policy" docs above (M0-02-T03). The Vulkan backend joins
// vokra-ops / vokra-backend-cpu / vokra-backend-metal / vokra-backend-cuda /
// vokra-capi / vokra-mmap on the workspace's unsafe-boundary allow list (root
// Cargo.toml).
#![allow(unsafe_code)]
// The FFI declares Vulkan handles like `VkInstance` etc. using non-snake-case
// identifiers to match the C prototypes verbatim. Silence the resulting
// `non_snake_case` warning for this crate only.
#![allow(non_snake_case)]
// The Vulkan C API uses `Pfn`-style function-pointer typedefs (e.g.
// `PFN_vkVoidFunction`); we mirror those names in `sys.rs` for cross-referencing
// with vulkan_core.h.
#![allow(non_camel_case_types)]

// The raw Vulkan FFI and the GPU context need a dynamic loader (dlopen /
// LoadLibrary), so they compile only on Vulkan-supported targets AND only
// when the `vulkan` feature is enabled. On other target/feature combinations
// (default builds, macOS / iOS / WASM) the probe / backend below fall back to
// explicit BackendUnavailable stubs.
#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
mod context;
#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
mod eval;
#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
mod sys;

// The probe and the Backend trait handle exist on every target (with explicit
// BackendUnavailable errors where the loader / feature is absent), so
// downstream code can always name them.
mod backend;
mod probe;
// The SPIR-V manifest / dispatcher lives at the crate root (no Vulkan target
// gating): it is the structural surface T14〜T22 landings extend, and the
// manifest is compile-time-only (no dlopen), so exposing it uniformly keeps
// the module tree stable.
pub mod spirv;

pub use backend::{
    GemmPipelinePreference, GemmPipelineVariant, VulkanBackend, select_gemm_pipeline_variant,
};
pub use probe::{VendorFamily, VulkanCapabilities, vokra_vulkan_probe};

// ---------------------------------------------------------------------------
// smoke_dispatch_copy_f32 — the M3-02-T13 end-to-end proof point (ADR
// M3-02-spirv-generation §4 (d)).
// ---------------------------------------------------------------------------

/// Round-trip `input` through the hand-crafted `copy_f32` SPIR-V kernel and
/// return the GPU-observed output.
///
/// This is the smoke-test entry point that proves the entire T08〜T12 + T25
/// Vulkan object stack — device, buffer, memory, descriptor set, pipeline,
/// command buffer, fence, dispatch — actually functions against a real
/// Vulkan driver. On a working Vulkan host the output is bit-identical to
/// `input` (the SPIR-V body is `dst[i] = src[i]` — a pure copy). Uses the
/// hand-crafted SPIR-V blob in
/// [`spirv::handcrafted_copy_f32`]; **no** `glslc` is invoked at build time
/// or runtime (NFR-DS-02 zero-dep + NFR-RL-05 no CPU-side JIT).
///
/// # Constraints
///
/// * `input.len()` must be a multiple of `spirv::handcrafted_copy_f32::LOCAL_SIZE_X`
///   (`64`). The hand-crafted shader has no bounds check by design (keeps
///   the bytecode small). This is a smoke-test API, not a general-purpose
///   memcpy — pad the caller's data if necessary.
///
/// # Errors
///
/// - [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable)
///   on non-Vulkan targets (macOS / iOS / WASM), default-features builds
///   (feature `vulkan` off), or when no Vulkan loader / ICD / compute queue
///   is present on the host. Callers **must** treat this as "the smoke test
///   is not applicable here" and skip — never fall back to a CPU
///   implementation (FR-EX-08).
/// - Any other error is a driver-side failure worth logging.
///
/// # Portability
///
/// - `libvulkan.so.1` (Linux + lavapipe / any ICD), `vulkan-1.dll` (Windows),
///   or an Android `libvulkan.so` present on-device — the dispatch runs
///   normally.
/// - macOS host **without** the LunarG SDK's MoltenVK ICD → returns
///   `BackendUnavailable`. Installing MoltenVK exposes a Vulkan-on-Metal
///   translator that would let this run on macOS as well.
#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
pub fn smoke_dispatch_copy_f32(input: &[f32]) -> vokra_core::Result<Vec<f32>> {
    context::smoke_dispatch_copy_f32_impl(input)
}

/// Stub for non-Vulkan builds — returns an explicit `BackendUnavailable`
/// error, never a silent CPU substitute (FR-EX-08 / NFR-RL-06).
#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
pub fn smoke_dispatch_copy_f32(_input: &[f32]) -> vokra_core::Result<Vec<f32>> {
    Err(vokra_core::VokraError::BackendUnavailable(
        "vokra-backend-vulkan compiled without the `vulkan` feature or on a non-Vulkan target \
         (macOS / iOS / WASM). The M3-02 smoke dispatch requires a Vulkan loader; install one \
         (Linux: libvulkan1 + mesa-vulkan-drivers; macOS: LunarG Vulkan SDK w/ MoltenVK) and \
         rebuild with `--features vulkan` on a supported target."
            .to_owned(),
    ))
}

/// Element-wise sum `a[i] + b[i] → out[i]` through the hand-crafted
/// `add_f32` SPIR-V kernel and return the GPU-observed output.
///
/// This is the M3-02-T24 three-SSBO proof point: it extends the
/// [`smoke_dispatch_copy_f32`] contract to a compute pipeline that binds
/// two readable SSBOs and one writable SSBO in the same dispatch, plus the
/// smallest arithmetic op (`OpFAdd`). On a working Vulkan host the output
/// equals the host sum under IEEE-754 f32 (bit-identical for the finite
/// inputs the smoke tests send).
///
/// # Panics
///
/// Panics if `a.len() != b.len()` or if the length is not a multiple of
/// [`spirv::handcrafted_add_f32::LOCAL_SIZE_X`] (`64`). The hand-crafted
/// shader has no bounds check by design — pad the caller's data.
///
/// # Errors
///
/// - [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable)
///   on non-Vulkan targets (macOS / iOS / WASM), default-features builds
///   (feature `vulkan` off), or when no Vulkan loader / ICD / compute queue
///   is present on the host. Callers **must** treat this as "the smoke test
///   is not applicable here" and skip — never fall back to a CPU
///   implementation (FR-EX-08).
/// - Any other error is a driver-side failure worth logging.
#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
pub fn smoke_dispatch_add_f32(a: &[f32], b: &[f32]) -> vokra_core::Result<Vec<f32>> {
    context::smoke_dispatch_add_f32_impl(a, b)
}

/// Stub for non-Vulkan builds — returns an explicit `BackendUnavailable`
/// error, never a silent CPU substitute (FR-EX-08 / NFR-RL-06).
#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
pub fn smoke_dispatch_add_f32(_a: &[f32], _b: &[f32]) -> vokra_core::Result<Vec<f32>> {
    Err(vokra_core::VokraError::BackendUnavailable(
        "vokra-backend-vulkan compiled without the `vulkan` feature or on a non-Vulkan target \
         (macOS / iOS / WASM). The M3-02 smoke dispatch requires a Vulkan loader; install one \
         (Linux: libvulkan1 + mesa-vulkan-drivers; macOS: LunarG Vulkan SDK w/ MoltenVK) and \
         rebuild with `--features vulkan` on a supported target."
            .to_owned(),
    ))
}
