//! M3-02-T35 gate — the Vulkan backend never silently falls back to the CPU
//! backend (FR-EX-08 / NFR-RL-06).
//!
//! Every uncovered op **must** surface as [`VokraError::UnsupportedOp`], and
//! every graph that carries an uncovered op **must** fail
//! [`Backend::execute`] with the same explicit error — never a silent
//! success.
//!
//! In the foundation slice **no** SPIR-V kernel is wired yet, so the Vulkan
//! backend's op coverage is the empty set. Every op is therefore an
//! explicit-error test today; as T14〜T22 land, the covered set grows and
//! the *uncovered* set (the negative-test surface) shrinks. This file stays
//! useful throughout: the coverage table is the single source of truth for
//! what IS covered, and the negative asserts cover the rest.
//!
//! Symmetric with `vokra-backend-metal` / `vokra-backend-cuda` where the
//! same FR-EX-08 red line is enforced.

use vokra_backend_vulkan::VulkanBackend;
use vokra_core::{Backend, DType, GraphBuilder, OpKind, TensorDesc, VokraError};

/// A graph carrying an op the Vulkan backend does NOT cover must fail
/// `execute` with an explicit `UnsupportedOp` — not silently succeed, and
/// not fall back to the CPU.
#[test]
fn uncovered_graph_is_explicit_unsupported() {
    // We use `execute` (the coverage check) rather than a live GPU dispatch
    // so this test runs on any host — the coverage precheck is
    // target-independent.
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(VokraError::BackendUnavailable(_)) => {
            eprintln!("no Vulkan on this host; execute() coverage precheck skipped");
            return;
        }
        Err(other) => panic!("expected BackendUnavailable off Vulkan, got {other}"),
    };
    // Foundation-slice coverage set = ∅. A trivial MatMul graph is
    // uncovered — must error explicitly.
    let mut mb = GraphBuilder::new();
    let a = mb.add_tensor(TensorDesc::new("a", DType::F32, [2, 4]));
    let b = mb.add_tensor(TensorDesc::new("b", DType::F32, [4, 8]));
    let c = mb.add_tensor(TensorDesc::new("c", DType::F32, [2, 8]));
    mb.add_node(OpKind::MatMul, &[a, b], &[c]);
    mb.mark_input(a);
    mb.mark_output(c);
    let g = mb.finish().expect("valid graph");
    let err = backend.execute(&g).unwrap_err();
    assert!(
        matches!(err, VokraError::UnsupportedOp(_)),
        "execute() must return UnsupportedOp for uncovered ops, got {err}",
    );
    // Also assert `supports()` and `execute()` are in lock-step — the two
    // MUST NOT diverge (M3-02-T35 core invariant).
    assert!(!backend.supports(&OpKind::MatMul));
    assert!(!backend.supports(&OpKind::Softmax));
    assert!(!backend.supports(&OpKind::Add));
}

/// Direct `eval_op` calls (which bypass the engine's coverage precheck) also
/// surface `UnsupportedOp` — no path can silently fall back to the CPU.
#[test]
fn eval_op_direct_is_explicit_unsupported() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(VokraError::BackendUnavailable(_)) => return,
        Err(other) => panic!("unexpected: {other}"),
    };
    let a = vokra_core::Tensor::zeros_f32(vec![2, 2]);
    let err = backend.eval_op(&OpKind::MatMul, &[&a, &a]).unwrap_err();
    assert!(matches!(err, VokraError::UnsupportedOp(_)));
}
