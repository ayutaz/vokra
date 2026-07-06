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
//!
//! The `parallel` worker pool (`pool`, M1-12) adds two further `unsafe` bridges
//! — lifetime-erasing the borrowed job closure onto the persistent workers (the
//! `std::thread::scope` pattern) and reconstructing each task's **disjoint**
//! `&mut` output row range — both `// SAFETY:`-documented and sound because the
//! completion barrier keeps the closure alive and the row ranges never overlap.
//! It spawns nothing on a single-core host or a WASM target; it is `std`-only
//! (no external crate), preserving NFR-DS-02.

// Local opt-out from the workspace `unsafe_code = "deny"` lint — see the
// crate-level "Unsafe policy" docs above (M0-02-T03).
#![allow(unsafe_code)]

mod dispatch;
// Graph-level per-op evaluation (Phase 1): the `Backend::eval_op` surface that
// `vokra_core::run_graph` drives, routing each op into `crate::kernels` (no
// second kernel implementation).
mod eval;
mod features;
pub mod kernels;
// Persistent row-parallel worker pool (M1-12). Native-only + feature-gated: the
// large GEMM/GEMV split over disjoint output rows (bit-identical to
// single-thread). WASM `std` has no thread spawning, so it is excluded there and
// the kernels run inline (see `pool` module docs, NFR-LC-03 / NFR-DS-02).
#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
mod pool;
mod selftest;

pub use dispatch::active_isa;
pub use dispatch::fused_log_mel_dispatch;
pub use features::{CpuFeatures, IsaPath};
pub use selftest::{SELFTEST_ATOL, SELFTEST_RTOL, SelftestReport, selftest};

/// Test-only probe exposing the M2-04-T06 fused log-mel `pub(crate)` kernels
/// to the `tests/fused_logmel_isa_parity.rs` integration harness. Not part of
/// the crate's public API — `#[doc(hidden)]` and only reachable via a
/// deliberately-named path.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
pub mod fused_logmel_test_probe {
    pub use crate::kernels::fused_logmel_avx2::{
        fused_logmel_apply_frame_avx2, fused_logmel_apply_frame_scalar,
    };
}

/// NEON companion of [`fused_logmel_test_probe`] (M2-04-T06). Exposes the
/// `pub(crate)` NEON kernel + its scalar oracle to the AArch64 integration
/// test in `tests/fused_logmel_isa_parity_neon.rs`. Not part of the crate's
/// public API.
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
pub mod fused_logmel_test_probe_neon {
    pub use crate::kernels::fused_logmel_neon::{
        fused_logmel_apply_frame_neon, fused_logmel_apply_frame_scalar,
    };
}

use vokra_core::{AudioGraph, Backend, OpKind, Result, Tensor, VokraError};

/// CPU backend handle implementing the `vokra-core` [`Backend`] trait.
///
/// # Two entry points
///
/// 1. **Direct kernels** ([`kernels`]) are the primary surface M0-06 uses:
///    `gemm_f32`, `softmax_f32`, `conv1d_f32`, … dispatched onto
///    [`active_isa`]. This is where the numeric work and the scalar/AVX2/NEON
///    parity (differential tests) live.
/// 2. **Graph execution.** [`Backend::eval_op`] evaluates one op on resolved
///    input [`Tensor`]s by routing into the same [`kernels`] as (1) — no
///    second implementation — and [`vokra_core::run_graph`] drives it node by
///    node. [`Backend::execute`] remains the coverage-check entry point:
///    per FR-EX-08 it returns an explicit error for ops it does not support
///    (never a silent fallback) and, once coverage holds,
///    [`VokraError::NotImplemented`] (it carries no tensor data).
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
        // Coverage is satisfied. `execute` stays a coverage-only check; the
        // data-carrying path is `vokra_core::run_graph`, which drives
        // `eval_op` below. Direct kernel use (`crate::kernels`) remains the
        // imperative M0-06 integration path.
        Err(VokraError::NotImplemented(
            "graph-level execution is vokra_core::run_graph (drives eval_op); execute is coverage-only",
        ))
    }

    fn eval_op(&self, op: &OpKind, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        crate::eval::eval_cpu_op(op, inputs)
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

    #[test]
    fn run_graph_matmul_add_softmax_is_bit_identical_to_direct_kernels() {
        // V1: a graph run through `vokra_core::run_graph(&CpuBackend, ..)` must
        // equal the same three kernels called directly, bit-for-bit (atol = 0):
        // the graph path routes into the very same `kernels::*` functions.
        //   h = x @ w ; g = h + bias ; out = softmax(g)
        let x_data: Vec<f32> = (0..8).map(|v| v as f32 * 0.25 - 1.0).collect(); // [2,4]
        let w_data: Vec<f32> = (0..32).map(|v| v as f32 * 0.1 - 1.5).collect(); // [4,8]
        let bias_data: Vec<f32> = (0..16).map(|v| v as f32 * 0.05).collect(); // [2,8]

        let mut b = GraphBuilder::new();
        let x = b.add_tensor(TensorDesc::new("x", DType::F32, [2, 4]));
        let w = b.add_tensor(TensorDesc::new("w", DType::F32, [4, 8]));
        let bias = b.add_tensor(TensorDesc::new("bias", DType::F32, [2, 8]));
        let h = b.add_tensor(TensorDesc::new("h", DType::F32, [2, 8]));
        let g = b.add_tensor(TensorDesc::new("g", DType::F32, [2, 8]));
        let out = b.add_tensor(TensorDesc::new("out", DType::F32, [2, 8]));
        b.add_node(OpKind::MatMul, &[x, w], &[h]);
        b.add_node(OpKind::Add, &[h, bias], &[g]);
        b.add_node(OpKind::Softmax, &[g], &[out]);
        b.mark_input(x);
        b.mark_output(out);
        let graph = b.finish().expect("valid graph");

        let outs = vokra_core::run_graph(
            &CpuBackend::new(),
            &graph,
            &[
                (x, Tensor::host_f32(vec![2, 4], x_data.clone()).unwrap()),
                (w, Tensor::host_f32(vec![4, 8], w_data.clone()).unwrap()),
                (
                    bias,
                    Tensor::host_f32(vec![2, 8], bias_data.clone()).unwrap(),
                ),
            ],
        )
        .expect("graph runs");

        // Direct kernel sequence with byte-for-byte identical arguments.
        let mut h_direct = vec![0.0f32; 16];
        kernels::gemm_f32(2, 8, 4, &x_data, &w_data, None, &mut h_direct).unwrap();
        let mut g_direct = vec![0.0f32; 16];
        kernels::add_f32(&h_direct, &bias_data, &mut g_direct).unwrap();
        let mut expected = vec![0.0f32; 16];
        kernels::softmax_f32(&g_direct, &mut expected, 2, 8).unwrap();

        assert_eq!(outs.len(), 1);
        assert_eq!(
            outs[0].as_f32().unwrap(),
            expected.as_slice(),
            "graph execution must be bit-identical to direct kernel calls"
        );
    }

    #[test]
    fn run_graph_rejects_unsupported_op_explicitly() {
        // V5-cpu: an uncovered op (Stft) inside a graph surfaces as an explicit
        // UnsupportedOp from the engine's coverage precheck — never a silent
        // skip (FR-EX-08).
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(TensorDesc::new("x", DType::F32, [400]));
        let y = b.add_tensor(TensorDesc::new("y", DType::F32, [3, 201]));
        b.add_node(OpKind::Stft(StftAttrs::new(400, 160)), &[x], &[y]);
        b.mark_input(x);
        b.mark_output(y);
        let graph = b.finish().expect("structurally valid graph");

        let err = vokra_core::run_graph(
            &CpuBackend::new(),
            &graph,
            &[(x, Tensor::zeros_f32(vec![400]))],
        )
        .unwrap_err();
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }
}
