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
