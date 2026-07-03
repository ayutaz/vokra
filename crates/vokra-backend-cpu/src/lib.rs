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
//! # Single-binary portable-baseline contract (M1-05)
//!
//! The completion bar for the x86-64 + ARM64 CPU backend is: **one binary**
//! runs on both architectures, picking the fastest kernels at run time
//! (FR-BE-01, FR-EX-06). Two invariants make that sound, and callers /
//! packagers must preserve them:
//!
//! - **(1) Compile at the ISA baseline.** The whole crate must build for the
//!   x86-64 baseline (`x86-64-v1`, i.e. SSE2 — every x86-64 CPU since 2003)
//!   and the AArch64 baseline (NEON). All above-baseline code (AVX2 + FMA3)
//!   lives **only** inside the per-function
//!   `#[target_feature(enable = "avx2,fma")]` cores in [`kernels::avx2`],
//!   reached only after [`CpuFeatures::detect`] confirms the feature. It is
//!   therefore a load-bearing rule for NFR-PT-02 (2010-era CPU support) that
//!   release builds **never raise the x86-64 baseline** — no
//!   `-Ctarget-cpu=native` / `x86-64-v2+`, no `-Ctarget-feature=+avx2`, in
//!   `.cargo/config.toml`, `RUSTFLAGS`, or `CARGO_ENCODED_RUSTFLAGS`. Raising
//!   it would let the *scalar fallback* and even [`features`]'s own detection
//!   code emit AVX2 and `SIGILL` on a pre-AVX2 CPU, defeating the whole
//!   dispatch design.
//! - **(2) No JIT, W^X-clean (NFR-RL-05).** Dispatch only swaps *function
//!   pointers* to statically compiled kernels ([`dispatch`]); there is no
//!   runtime code generation, no `PROT_EXEC` allocation, and no
//!   `__clear_cache`. This is what lets a single signed binary run under
//!   iOS / hardened-runtime W^X.
//!
//! [`selftest`] is the runtime proof of (1)+(2): a shipped binary can call it
//! to confirm, on the real host, that its selected SIMD path matches the
//! scalar oracle.
//!
//! The *formal* record of this contract (ADR-0004), the CI guard that fails a
//! build whose `RUSTFLAGS` raise the baseline, the `VOKRA_CPU_ISA` forced-path
//! CI leg, and the W^X forbidden-symbol scan live outside this crate and are
//! owned by the CI / docs work package (as the M0-08 design record above notes
//! for the ADR tree); the `tests/single_binary_dispatch.rs` integration test
//! provides the in-crate, `cargo test`-level forced-path and self-consistency
//! coverage so the guarantee does not depend on CI wiring.
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
mod selftest;

pub use dispatch::active_isa;
pub use features::{CpuFeatures, IsaPath};
pub use selftest::{SELFTEST_ATOL, SELFTEST_RTOL, SelftestReport, selftest};

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
    use vokra_core::ir::graph::StftAttrs;
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
    fn execute_rejects_unsupported_op_explicitly() {
        // A front-end op such as `stft` (owned by `vokra-ops`, no CPU kernel
        // here) must surface as an explicit UnsupportedOp error, never a
        // silent skip (FR-EX-08 "no silent fallback").
        let backend = CpuBackend::new();
        assert!(!backend.supports(&OpKind::Stft(StftAttrs::new(400, 160))));

        let mut b = GraphBuilder::new();
        let x = b.add_tensor(TensorDesc::new("x", DType::F32, [400]));
        let y = b.add_tensor(TensorDesc::new("y", DType::F32, [2, 201]));
        b.add_node(OpKind::Stft(StftAttrs::new(400, 160)), &[x], &[y]);
        b.mark_input(x);
        b.mark_output(y);
        let graph = b.finish().expect("structurally valid graph");

        let result = backend.execute(&graph);
        assert!(matches!(result, Err(VokraError::UnsupportedOp(_))));
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
