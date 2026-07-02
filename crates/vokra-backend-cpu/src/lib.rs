//! # vokra-backend-cpu
//!
//! CPU backend for Vokra — the first-class backend (FR-BE-01; SRS §1.3
//! `vokra-backend-*` family).
//!
//! M0-02 ships only the [`CpuBackend`] skeleton implementing the
//! `vokra-core` [`Backend`] trait. **AVX2 (x86-64) / NEON (ARM64) kernels
//! and the runtime ISA dispatch are implemented in M0-08** (FR-BE-01 spike
//! scope; single-binary dispatch reliability per NFR-RL-05); JIT is not
//! used (iOS W^X constraint). Wider ISA tiers (SSE2 baseline, AVX-512,
//! AMX, SVE/SVE2, SME, RVV, WASM SIMD) follow the roadmap in CLAUDE.md /
//! FR-BE-01.
//!
//! # Unsafe policy (NFR-RL-07, SRS §5-(1))
//!
//! SIMD intrinsics require `unsafe`, so this crate opts out of the
//! workspace-wide `unsafe_code = "deny"` below. Public APIs must stay safe,
//! and every `unsafe` block requires a `// SAFETY:` comment (enforced by
//! `clippy::undocumented_unsafe_blocks` at the workspace level).

// Local opt-out from the workspace `unsafe_code = "deny"` lint — see the
// crate-level "Unsafe policy" docs above (M0-02-T03).
#![allow(unsafe_code)]

use vokra_core::{AudioGraph, Backend, OpKind, Result, VokraError};

/// CPU backend skeleton (M0-02-T10).
///
/// Kernels and runtime ISA dispatch land in **M0-08**; until then
/// [`CpuBackend::execute`] is an explicit stub.
#[derive(Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// Creates a CPU backend handle.
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn supports(&self, op: &OpKind) -> bool {
        // Placeholder coverage answer for the M0 skeleton ops. Future op
        // kinds (added by M0-04/05/06) stay unsupported here until their
        // CPU kernels exist — an explicit error, never a silent fallback
        // (FR-EX-08).
        matches!(
            op,
            OpKind::MatMul | OpKind::Add | OpKind::Mul | OpKind::Softmax
        )
    }

    fn execute(&self, graph: &AudioGraph) -> Result<()> {
        for node in graph.nodes() {
            if !self.supports(node.op()) {
                return Err(VokraError::UnsupportedOp(format!(
                    "cpu backend has no kernel for {:?}",
                    node.op()
                )));
            }
        }
        Err(VokraError::NotImplemented(
            "CPU kernels + runtime ISA dispatch land in M0-08",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::{DType, GraphBuilder, TensorDesc};

    fn tiny_graph() -> AudioGraph {
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(TensorDesc::new("x", DType::F32, [2, 4]));
        let w = b.add_tensor(TensorDesc::new("w", DType::F32, [4, 8]));
        let y = b.add_tensor(TensorDesc::new("y", DType::F32, [2, 8]));
        b.add_node(OpKind::MatMul, &[x, w], &[y]);
        b.mark_input(x);
        b.mark_output(y);
        b.finish().expect("valid graph")
    }

    #[test]
    fn cpu_backend_reports_name_and_placeholder_coverage() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
        assert!(backend.supports(&OpKind::MatMul));
        assert!(backend.supports(&OpKind::Add));
        assert!(backend.supports(&OpKind::Mul));
        assert!(backend.supports(&OpKind::Softmax));
    }

    #[test]
    fn execute_is_an_explicit_stub_until_m0_08() {
        let backend = CpuBackend::new();
        let result = backend.execute(&tiny_graph());
        assert!(matches!(result, Err(VokraError::NotImplemented(_))));
    }

    #[test]
    fn cpu_backend_usable_as_dyn_backend() {
        let backend: Box<dyn Backend> = Box::new(CpuBackend::new());
        assert_eq!(backend.name(), "cpu");
    }
}
