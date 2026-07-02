//! # vokra-backend-cpu
//!
//! CPU backend for Vokra — the first-class backend (FR-BE-01; SRS §1.3
//! `vokra-backend-*` family). This crate provides the f32 compute kernels
//! ([`kernels`]) and the single-binary runtime ISA dispatch ([`active_isa`])
//! that Whisper (M0-06) and the Silero VAD subgraph (M0-05) build on.
//!
//! # Design record (M0-08-T01)
//!
//! Recorded here rather than in `docs/adr/` because the ADR tree is owned by
//! a parallel work package; the ticket allows the kernel-set / dispatch
//! design to live in crate docs (M0-08-T02). Fixed decisions with sources:
//!
//! - **(a) spike SIMD paths = AVX2 (x86-64) + NEON (ARM64) only.** AVX2+FMA3
//!   is the x86-64 main path (FR-BE-01; CLAUDE.md "AVX2 + FMA3 ... 主力パス").
//!   SSE2 baseline / AVX-512 / AVX-VNNI / AMX and the ARM upper tiers
//!   (dotprod / i8mm / bf16 / SVE / SVE2 / SME), RVV and WASM SIMD are later
//!   (FR-BE-01 "v0.1 spike (AVX2/NEON) → 拡張"; FR-EX-06 completion is M1,
//!   milestones.md §4.2 note 1).
//! - **(b) NEON is an ARMv8-A baseline** (CLAUDE.md), so on `aarch64` it is
//!   always available and installed without a runtime branch.
//! - **(c) No JIT** (NFR-RL-05, iOS W^X): dispatch swaps *function pointers*
//!   to statically compiled kernels (FR-EX-06 "llama.cpp/OpenBLAS 方式"); no
//!   runtime code generation or `PROT_EXEC` allocation exists in this crate.
//! - **(d) Scalar fallback** for x86-64 without AVX2 ([`kernels::scalar`]) is
//!   a portable Rust path reused as the SIMD differential oracle — **not** a
//!   preview of a future SSE2-optimised tier (that is M1+).
//!
//! # Unsafe policy (NFR-RL-07, SRS §5-(1))
//!
//! SIMD intrinsics require `unsafe`, so this crate opts out of the
//! workspace-wide `unsafe_code = "deny"`. Public APIs stay safe: every kernel
//! wrapper in [`kernels`] validates shapes at the boundary and returns
//! [`VokraError::InvalidArgument`](vokra_core::VokraError::InvalidArgument) on
//! a mismatch. SIMD `unsafe fn`s are reached only after
//! [`CpuFeatures::detect`] confirms the feature (the dispatch invariant), and
//! every `unsafe` block carries a `// SAFETY:` comment (enforced by
//! `clippy::undocumented_unsafe_blocks`).

// Local opt-out from the workspace `unsafe_code = "deny"` lint — see the
// crate-level "Unsafe policy" docs above (M0-02-T03).
#![allow(unsafe_code)]

mod dispatch;
mod features;
pub mod kernels;

pub use dispatch::active_isa;
pub use features::{CpuFeatures, IsaPath};

use vokra_core::{AudioGraph, Backend, OpKind, Result, VokraError};

/// CPU backend handle implementing the `vokra-core` [`Backend`] trait.
///
/// # Two entry points
///
/// 1. **Direct kernels** ([`kernels`]) are the primary surface M0-06 uses:
///    `gemm_f32`, `softmax_f32`, `conv1d_f32`, … dispatched onto
///    [`active_isa`]. This is where the numeric work and the scalar/AVX2/NEON
///    parity (differential tests) live.
/// 2. **Graph execution** ([`Backend::execute`]) validates op coverage and,
///    per FR-EX-08, returns an explicit error for ops it does not support —
///    never a silent fallback. The data-carrying graph evaluator is a later
///    work package (the M0 IR [`AudioGraph`] is a descriptor without tensor
///    storage), so `execute` reports [`VokraError::NotImplemented`] once
///    coverage is satisfied.
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
        // Op kinds with a wired CPU kernel. Future op kinds (added by
        // M0-04/05/06) stay unsupported until their kernels land — an
        // explicit error, never a silent fallback (FR-EX-08). Front-end ops
        // such as `stft` are executed by `vokra-ops`, not here (M0-04-T17).
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
        // Coverage is satisfied; the tensor-data-carrying graph evaluator is a
        // later WP. Until then, call the kernels in `crate::kernels` directly
        // (the M0-06 integration path).
        Err(VokraError::NotImplemented(
            "graph-level execution needs the data-carrying engine (later WP); use crate::kernels directly",
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
    fn cpu_backend_reports_name_and_kernel_coverage() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
        assert!(backend.supports(&OpKind::MatMul));
        assert!(backend.supports(&OpKind::Add));
        assert!(backend.supports(&OpKind::Mul));
        assert!(backend.supports(&OpKind::Softmax));
    }

    #[test]
    fn execute_passes_coverage_then_defers_to_data_engine() {
        // All ops in the graph are covered, so execution reaches the
        // explicit "later WP" stub rather than an UnsupportedOp error.
        let backend = CpuBackend::new();
        let result = backend.execute(&tiny_graph());
        assert!(matches!(result, Err(VokraError::NotImplemented(_))));
    }

    #[test]
    fn cpu_backend_usable_as_dyn_backend() {
        let backend: Box<dyn Backend> = Box::new(CpuBackend::new());
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn active_isa_is_reachable_from_public_api() {
        // Smoke check that M0-06's demo can query the selected path.
        let isa = active_isa();
        assert!(CpuFeatures::detect().supports(isa));
    }
}
