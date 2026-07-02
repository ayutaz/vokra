//! Backend abstraction (M0-02-T10).
//!
//! SRS §1.2 requires a single backend abstraction ("backend trait, 統一 op
//! coverage" — uniform op coverage). Concrete backends live in their own
//! crates (`vokra-backend-cpu` in M0; Metal / CUDA / Vulkan / ... follow the
//! roadmap).

use crate::error::Result;
use crate::ir::{AudioGraph, OpKind};

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

    /// Executes `graph` on this backend.
    ///
    /// M0 skeleton: implementations are stubs that return
    /// [`VokraError::NotImplemented`](crate::VokraError::NotImplemented)
    /// (or `UnsupportedOp` for uncovered ops); real execution arrives with
    /// the kernel work packages (CPU: M0-08).
    fn execute(&self, graph: &AudioGraph) -> Result<()>;
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
}
