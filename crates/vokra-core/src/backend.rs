//! Backend abstraction (M0-02-T10).
//!
//! SRS §1.2 requires a single backend abstraction ("backend trait, 統一 op
//! coverage" — uniform op coverage). Concrete backends live in their own
//! crates (`vokra-backend-cpu` in M0; Metal / CUDA / Vulkan / ... follow the
//! roadmap).

use crate::error::{Result, VokraError};
use crate::ir::{AudioGraph, OpKind};
use crate::runtime::Tensor;

/// Abstraction implemented by every compute backend.
///
/// # Uniform op coverage (FR-EX-08, permanent constraint)
///
/// All backends guarantee the *same* op coverage. An op a backend cannot
/// execute is an **explicit error**
/// ([`VokraError::UnsupportedOp`](crate::VokraError::UnsupportedOp)); silent
/// CPU fallback is never the default, and Vokra does not adopt ONNX
/// Runtime's execution-provider graph partitioning.
pub trait Backend {
    /// Human-readable backend name (e.g. `"cpu"`).
    fn name(&self) -> &str;

    /// Whether this backend can execute `op`.
    ///
    /// Implementations must answer `false` for unknown (future) op kinds so
    /// that unsupported ops surface as explicit errors (FR-EX-08).
    fn supports(&self, op: &OpKind) -> bool;

    /// Validates that this backend covers every op in `graph`.
    ///
    /// This is the **coverage-check** entry point: it returns
    /// [`VokraError::UnsupportedOp`](crate::VokraError::UnsupportedOp) for an
    /// op the backend cannot run (FR-EX-08) and otherwise reports
    /// [`VokraError::NotImplemented`](crate::VokraError::NotImplemented) — it
    /// carries no tensor data and never computes a result.
    ///
    /// Data-carrying execution is [`run_graph`](crate::run_graph), which drives
    /// [`eval_op`](Self::eval_op) node by node; `execute` is retained for
    /// coverage verification (and may be revisited for deprecation later).
    fn execute(&self, graph: &AudioGraph) -> Result<()>;

    /// Evaluates a single op on already-resolved input tensors, allocating and
    /// returning its output tensors.
    ///
    /// This is the per-op compute surface the graph evaluator
    /// [`run_graph`](crate::run_graph) drives (one call per node, in
    /// topological order). A backend derives each output's shape from the op
    /// semantics and the inputs; [`run_graph`] checks that shape against the
    /// declared [`TensorDesc`](crate::TensorDesc), so a backend only computes.
    ///
    /// # Contract (FR-EX-08, permanent)
    ///
    /// An op for which [`supports`](Self::supports) returns `false` MUST return
    /// [`VokraError::UnsupportedOp`] — never a silent fallback to another
    /// backend. The default implementation returns `UnsupportedOp` for every
    /// op, so a backend that has not wired its per-op kernels yet still
    /// compiles; concrete backends override this for the ops they cover and
    /// keep the override set in sync with [`supports`](Self::supports).
    fn eval_op(&self, op: &OpKind, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        let _ = inputs;
        Err(VokraError::UnsupportedOp(format!(
            "{}: eval_op has no kernel for {:?}",
            self.name(),
            op
        )))
    }
}

/// Selector for the backend a [`Session`](crate::Session) runs on; used by
/// [`with_backend`](crate::SessionBuilder::with_backend) (FR-API-02).
///
/// M0 provides only [`BackendKind::Cpu`] (FR-BE-01; the spike scope of the
/// CPU backend is AVX2 / NEON, implemented in M0-08). Further kinds (Metal,
/// CUDA, Vulkan, WebGPU, CoreML, QNN) are added when their backends land,
/// which is why the enum is `#[non_exhaustive]`.
///
/// **NNAPI will never be added** to this enum (FR-BE-07, permanent
/// constraint; Google deprecated NNAPI with Android 15 — see CLAUDE.md
/// "なぜ NNAPI に対応しないか").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BackendKind {
    /// CPU backend (first-class, FR-BE-01). Kernels + runtime ISA dispatch
    /// are implemented in `vokra-backend-cpu` (M0-08).
    Cpu,
    /// Metal backend (macOS / iOS, FR-BE-02). Implemented in
    /// `vokra-backend-metal` (M2-01); the imperative model hot path reaches it
    /// through the `vokra-models` `Compute` dispatcher. Selecting it for a model
    /// whose op set the Metal backend does not yet cover is an explicit
    /// [`VokraError::UnsupportedOp`](crate::VokraError::UnsupportedOp) — never a
    /// silent CPU fall back (FR-EX-08).
    Metal,
    /// CUDA backend (Windows / Linux NVIDIA GPUs, FR-BE-03). Implemented in
    /// `vokra-backend-cuda` (M2-03) with raw CUDA Driver API + NVRTC FFI loaded
    /// at runtime via dlopen (no `cudarc` binding crate, no bundled CUDA runtime
    /// — NVIDIA EULA install model, FR-BE-08). Reached through the `vokra-models`
    /// `Compute` dispatcher behind its `cuda` feature. A missing driver/GPU or an
    /// op the CUDA backend does not yet cover is an explicit
    /// [`VokraError::BackendUnavailable`](crate::VokraError::BackendUnavailable) /
    /// [`VokraError::UnsupportedOp`](crate::VokraError::UnsupportedOp) — never a
    /// silent CPU fall back (FR-EX-08 / NFR-RL-06).
    Cuda,
    /// Vulkan backend (Android / Linux / non-NVIDIA Windows GPUs, FR-BE-04).
    /// Implemented in `vokra-backend-vulkan` (M3-02) with raw Vulkan API FFI
    /// loaded at runtime via dlopen / LoadLibrary (no `ash` / `vulkano` / `erupt`
    /// binding crate) and pre-compiled SPIR-V compute shaders (subgroup + coop-
    /// matrix + subgroup-only fallback). Reached through the `vokra-models`
    /// `Compute` dispatcher behind its `vulkan` feature. Foundation slice
    /// (M3-02-T01〜T13 landed): SPIR-V kernels for GEMM / GEMV / softmax / … land
    /// in M3-02-T14 onwards, so every hot op is currently reported as
    /// [`VokraError::UnsupportedOp`](crate::VokraError::UnsupportedOp) — never a
    /// silent CPU fall back (FR-EX-08 / NFR-RL-06). A missing loader / device is
    /// [`VokraError::BackendUnavailable`](crate::VokraError::BackendUnavailable).
    ///
    /// **NNAPI is permanently unsupported** — Vokra's Android GPU path is
    /// Vulkan-only from day one (FR-BE-07 / CLAUDE.md design constraint 8).
    Vulkan,
    /// WebGPU backend (browser WASM, FR-BE-05). Implemented in
    /// `vokra-backend-webgpu` (M4-01) with a raw
    /// `#[link(wasm_import_module = "vokra_webgpu")]` extern-import shim plus
    /// hand-written JS glue that drives the browser `navigator.gpu` API — no
    /// `wgpu` / `wasm-bindgen` binding crate (ADR M4-01-webgpu-wasm; the
    /// import-object resolution at instantiate time is the WASM equivalent of
    /// the dlopen runtime-linking model the Metal / CUDA / Vulkan backends
    /// use). Reached through the `vokra-models` `Compute` dispatcher behind
    /// its `webgpu` feature, compiled only on
    /// `cfg(target_arch = "wasm32")`.
    ///
    /// A missing WebGPU adapter (`navigator.gpu` absent / `requestAdapter`
    /// null — non-WebGPU browsers or environments) is an explicit
    /// [`VokraError::BackendUnavailable`](crate::VokraError::BackendUnavailable);
    /// an op the WebGPU backend does not cover is an explicit
    /// [`VokraError::UnsupportedOp`](crate::VokraError::UnsupportedOp) —
    /// never a silent CPU fall back (FR-EX-08 / NFR-RL-06). Running on the
    /// WASM CPU (SIMD128) path instead is the caller's *explicit*
    /// [`BackendKind::Cpu`] choice.
    WebGpu,
    /// CoreML delegate backend (Apple ANE, FR-BE-06). Implemented in
    /// `vokra-backend-coreml` (M5-01) with raw Objective-C / CoreML framework
    /// FFI (no `objc` / `objc2` / `objc2-core-ml` / `core-foundation` binding
    /// crate), reached through the `vokra-models` `Compute` dispatcher behind
    /// its `coreml` feature (compiled only on macOS / iOS). Unlike the dlopen
    /// backends this is a *delegate*: the intended execution unit is a declared
    /// submodel, and CoreML's own placement onto ANE / GPU / CPU inside that
    /// submodel is Apple's runtime concern — not a Vokra-side op partition
    /// (which the [`Backend`] trait's "same op coverage" rule forbids) and not
    /// a silent fallback.
    ///
    /// **Scaffold status (M5-01):** the op-execution path lands after the
    /// model-supply ADR (M5-01-T02) is ratified, so every hot op is currently
    /// reported as
    /// [`VokraError::UnsupportedOp`](crate::VokraError::UnsupportedOp). A host
    /// with no reachable Apple Neural Engine (an Intel Mac, or any non-Apple
    /// target where the backend is compiled out) is an explicit
    /// [`VokraError::BackendUnavailable`](crate::VokraError::BackendUnavailable)
    /// — never a silent CPU fall back (FR-EX-08 / NFR-RL-06).
    ///
    /// A **C-level** selector for this delegate is intentionally *not* exported
    /// during the v1.0-rc window; that is an M5-13 decision after the
    /// real-hardware NPU bakeoff (`include/vokra.h`, `docs/handoff/m4-12.md`).
    /// The Rust-side surface (`with_backend` / `vokra-cli --backend coreml`) is
    /// the only way to select it for now.
    CoreMl,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::VokraError;

    /// Minimal in-crate implementation proving the trait is object-safe and
    /// usable via `dyn Backend`.
    struct NullBackend;

    impl Backend for NullBackend {
        fn name(&self) -> &str {
            "null"
        }

        fn supports(&self, _op: &OpKind) -> bool {
            false
        }

        fn execute(&self, _graph: &AudioGraph) -> Result<()> {
            Err(VokraError::NotImplemented("null backend never executes"))
        }
    }

    #[test]
    fn backend_trait_is_object_safe() {
        let b: Box<dyn Backend> = Box::new(NullBackend);
        assert_eq!(b.name(), "null");
        assert!(!b.supports(&OpKind::MatMul));
    }

    #[test]
    fn backend_kind_is_copy_and_comparable() {
        let k = BackendKind::Cpu;
        let k2 = k;
        assert_eq!(k, k2);
    }

    #[test]
    fn default_eval_op_is_unsupported() {
        // A backend that does not override `eval_op` inherits the default,
        // which is an explicit UnsupportedOp for every op (FR-EX-08) — never a
        // silent fallback.
        let b = NullBackend;
        let err = b.eval_op(&OpKind::MatMul, &[]).unwrap_err();
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }
}
