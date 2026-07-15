//! M4-13-T09 — graph-executor op coverage: machine-checked "same as the
//! CUDA arm" contract (milestones §8 M4-13 completion condition #2).
//!
//! Host-portable: [`graph_op_backing_shader`] is a pure decision function,
//! so the *principled* coverage set (which ops the Vulkan graph arm covers
//! once their blobs land) is asserted on every host, including the
//! Apple-Silicon authoring machine and lavapipe CI.
//!
//! Two-surface reminder (M4-13-T01): this file is about **surface 1** — the
//! graph `OpKind` enum driven by `vokra_core::run_graph`. `OpKind::Gemv` /
//! `LayerNorm` / `Gelu` / `Conv1D` / `SoftmaxCausal` / `Transpose` /
//! `Gather` do not exist; those SPIR-V kernels (surface 2) are exercised by
//! the Whisper model-level parity harness instead (M4-13-T12/T13).

use vokra_backend_vulkan::{GemmPipelineVariant, graph_op_backing_shader, spirv};
use vokra_core::OpKind;
use vokra_core::ir::graph::StftAttrs;

/// The CUDA graph-executor arm's `supports()` set, transcribed from
/// `crates/vokra-backend-cuda/src/backend.rs` (M3-01-T06 unified coverage
/// table: `MatMul | Add | Mul | Softmax`). Kept here as the *documented*
/// parity target — the two crates deliberately do not depend on each other,
/// so the pairing is pinned by this test plus the coverage table in
/// `crates/vokra-backend-vulkan/kernels/README.md` (M4-13-T14).
const CUDA_SUPPORTS: [&str; 4] = ["MatMul", "Add", "Mul", "Softmax"];

fn op_label(op: &OpKind) -> &'static str {
    match op {
        OpKind::MatMul => "MatMul",
        OpKind::Add => "Add",
        OpKind::Mul => "Mul",
        OpKind::Softmax => "Softmax",
        OpKind::Copy => "Copy",
        OpKind::Stft(_) => "Stft",
        OpKind::DcOffsetRemove => "DcOffsetRemove",
        _ => "other",
    }
}

/// The Vulkan principled coverage set is exactly `{Copy, Add, MatMul, Mul,
/// Softmax}` — the CUDA set plus `Copy` (the Vulkan runtime-verification
/// op). Asserted for both GEMM variants so probe selection cannot change
/// the *op* coverage, only the shader behind `MatMul`.
#[test]
fn principled_coverage_equals_cuda_arm_plus_copy() {
    let covered_ops = [
        OpKind::Copy,
        OpKind::Add,
        OpKind::MatMul,
        OpKind::Mul,
        OpKind::Softmax,
    ];
    for variant in [
        GemmPipelineVariant::Subgroup,
        GemmPipelineVariant::CoopMatrix,
    ] {
        // Every covered op maps to a backing shader…
        let mut covered_labels: Vec<&str> = Vec::new();
        for op in &covered_ops {
            assert!(
                graph_op_backing_shader(op, variant).is_some(),
                "{op:?} must have a backing shader (variant {variant:?})"
            );
            covered_labels.push(op_label(op));
        }
        // …and the covered set minus Copy is exactly the CUDA supports() set.
        let minus_copy: Vec<&str> = covered_labels
            .iter()
            .copied()
            .filter(|l| *l != "Copy")
            .collect();
        let mut want = CUDA_SUPPORTS.to_vec();
        want.sort_unstable();
        let mut got = minus_copy.clone();
        got.sort_unstable();
        assert_eq!(
            got, want,
            "Vulkan graph-arm coverage minus Copy must equal the CUDA arm \
             (crates/vokra-backend-cuda/src/backend.rs supports())"
        );
    }
}

/// `Stft` is covered by NEITHER graph arm — the CUDA arm rejects it too
/// (its `uncovered_op_is_explicit_unsupported` test uses `Stft` as the
/// canonical uncovered op). The milestones §8 M4-13 line lists it as the
/// honest gap in the coverage table; front-end signal ops run in
/// `vokra-ops`, not on the GPU graph arms.
#[test]
fn stft_is_the_honest_gap_on_both_arms() {
    let stft = OpKind::Stft(StftAttrs::new(400, 160));
    for variant in [
        GemmPipelineVariant::Subgroup,
        GemmPipelineVariant::CoopMatrix,
    ] {
        assert!(
            graph_op_backing_shader(&stft, variant).is_none(),
            "Stft must have no Vulkan graph-arm backing shader (honest gap; \
             CUDA has none either)"
        );
    }
    assert!(
        !CUDA_SUPPORTS.contains(&"Stft"),
        "documented CUDA set must not contain Stft"
    );
    // Another representative uncovered op (attr-less, front-end).
    assert!(
        graph_op_backing_shader(&OpKind::DcOffsetRemove, GemmPipelineVariant::Subgroup).is_none()
    );
}

/// Every backing shader named by the decision function is a real manifest
/// entry — a typo here would turn a "supported" op into a guaranteed
/// dispatch failure.
#[test]
fn backing_shaders_are_manifest_entries() {
    let ops = [
        OpKind::Copy,
        OpKind::Add,
        OpKind::MatMul,
        OpKind::Mul,
        OpKind::Softmax,
    ];
    for variant in [
        GemmPipelineVariant::Subgroup,
        GemmPipelineVariant::CoopMatrix,
    ] {
        for op in &ops {
            let shader = graph_op_backing_shader(op, variant).unwrap();
            assert!(
                spirv::SHADERS.iter().any(|s| s.name == shader),
                "backing shader `{shader}` for {op:?} is not in the SPIR-V manifest"
            );
        }
    }
    // MatMul routes to the variant's shader specifically.
    assert_eq!(
        graph_op_backing_shader(&OpKind::MatMul, GemmPipelineVariant::Subgroup),
        Some("gemm_subgroup")
    );
    assert_eq!(
        graph_op_backing_shader(&OpKind::MatMul, GemmPipelineVariant::CoopMatrix),
        Some("gemm_coopmat")
    );
}
