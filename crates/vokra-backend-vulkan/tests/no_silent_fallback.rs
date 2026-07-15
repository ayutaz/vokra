//! M3-02-T35 gate — the Vulkan backend never silently falls back to the CPU
//! backend (FR-EX-08 / NFR-RL-06).
//!
//! Every uncovered op **must** surface as [`VokraError::UnsupportedOp`], and
//! every graph that carries an uncovered op **must** fail
//! [`Backend::execute`] with the same explicit error — never a silent
//! success.
//!
//! In the M3-02 foundation slice the Vulkan backend covers `Copy` and `Add`
//! only (both hand-crafted SPIR-V smoke kernels — M3-02-T13 / T24). Every
//! other op is an explicit-error test today; as T14〜T22 SPIR-V shaders land
//! the covered set widens and this file's negative-test surface shrinks.
//! The coverage table (`VulkanBackend::supports`) is the single source of
//! truth for what IS covered, and the asserts below pin the rest.
//!
//! Symmetric with `vokra-backend-metal` / `vokra-backend-cuda` where the
//! same FR-EX-08 red line is enforced.

use vokra_backend_vulkan::{GemmPipelinePreference, VulkanBackend, graph_op_backing_shader, spirv};
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
    // A PERMANENTLY uncovered op (no Vulkan graph arm; the CUDA arm has
    // none either) — stays an explicit error even after the owner's blob
    // commit widens the blob-gated set (M4-13-T09).
    let mut mb = GraphBuilder::new();
    let a = mb.add_tensor(TensorDesc::new("a", DType::F32, [64]));
    let c = mb.add_tensor(TensorDesc::new("c", DType::F32, [64]));
    mb.add_node(OpKind::DcOffsetRemove, &[a], &[c]);
    mb.mark_input(a);
    mb.mark_output(c);
    let g = mb.finish().expect("valid graph");
    let err = backend.execute(&g).unwrap_err();
    assert!(
        matches!(err, VokraError::UnsupportedOp(_)),
        "execute() must return UnsupportedOp for uncovered ops, got {err}",
    );
    // `supports()` and `execute()` in lock-step — the two MUST NOT diverge
    // (M3-02-T35 core invariant). Since M4-13-T09 the blob-gated ops
    // (MatMul / Mul / Softmax) track blob availability instead of a
    // hard-coded false.
    let variant = backend
        .select_gemm_pipeline_variant(GemmPipelinePreference::default())
        .expect("default preference never errors");
    for op in [OpKind::MatMul, OpKind::Softmax, OpKind::Mul] {
        let shader = graph_op_backing_shader(&op, variant).unwrap();
        assert_eq!(
            backend.supports(&op),
            spirv::has_blob(shader),
            "supports({op:?}) must track blob availability of `{shader}`"
        );
    }
    assert!(!backend.supports(&OpKind::DcOffsetRemove));
    // Hand-crafted-backed ops are always covered.
    assert!(
        backend.supports(&OpKind::Copy),
        "Copy IS covered (hand-crafted `copy_f32`)"
    );
    assert!(
        backend.supports(&OpKind::Add),
        "Add IS covered (hand-crafted `add_f32`)"
    );
}

/// Direct `eval_op` calls (which bypass the engine's coverage precheck) also
/// surface `UnsupportedOp` — no path can silently fall back to the CPU.
/// Uses a permanently-uncovered op (front-end signal op) plus the blob-gated
/// MatMul arm in whichever state its blob is in.
#[test]
fn eval_op_direct_is_explicit_unsupported() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(VokraError::BackendUnavailable(_)) => return,
        Err(other) => panic!("unexpected: {other}"),
    };
    // Permanent: no graph arm.
    let x = vokra_core::Tensor::zeros_f32(vec![64]);
    let err = backend.eval_op(&OpKind::DcOffsetRemove, &[&x]).unwrap_err();
    assert!(matches!(err, VokraError::UnsupportedOp(_)));

    // Blob-gated: MatMul is UnsupportedOp exactly while its blob is absent.
    let variant = backend
        .select_gemm_pipeline_variant(GemmPipelinePreference::default())
        .expect("default preference never errors");
    let a = vokra_core::Tensor::zeros_f32(vec![2, 2]);
    let result = backend.eval_op(&OpKind::MatMul, &[&a, &a]);
    if spirv::has_blob(variant.shader_name()) {
        assert!(result.is_ok(), "blob present → MatMul dispatches");
    } else {
        assert!(matches!(result, Err(VokraError::UnsupportedOp(_))));
    }
}
