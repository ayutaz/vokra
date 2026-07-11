//! M3-02-T26 — the graph-executor Vulkan arm.
//!
//! Verifies that [`vokra_core::run_graph`] can thread real tensor values
//! through a Vulkan backend end-to-end when the graph's ops all fall inside
//! the [`VulkanBackend::supports`] set (M3-02-T24 foundation slice: `Copy`
//! and `Add`).
//!
//! The T24 lock-step gate says `supports()` and `eval_op()` MUST advertise
//! the same coverage — this test locks that in against real graph
//! evaluation, not just the direct `eval_op` path.
//!
//! Off Vulkan hosts the whole test skips cleanly via `BackendUnavailable`.

use vokra_backend_vulkan::{VulkanBackend, spirv};
use vokra_core::{DType, GraphBuilder, OpKind, Tensor, TensorDesc, VokraError, run_graph};

/// A `Copy → Add` graph: `y = copy(x) + z`. Evaluates on Vulkan, then
/// against the host reference, and asserts they agree bit-for-bit under
/// IEEE-754 f32 semantics.
///
/// Two ops in a chain proves the [`run_graph`] coverage precheck accepts
/// the whole graph AND the topological execution loop threads intermediate
/// tensors through the Vulkan backend's `eval_op` correctly.
#[test]
fn run_graph_dispatches_copy_then_add_through_vulkan() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan; skipping run_graph(Copy → Add) test");
        return;
    };
    let local = spirv::handcrafted_add_f32::LOCAL_SIZE_X as usize;
    let n = 2 * local;

    // Graph:
    //   x  ── Copy ── x_copy ──┐
    //                          ├── Add ── y
    //   z  ─────────────────────┘
    let mut gb = GraphBuilder::new();
    let x = gb.add_tensor(TensorDesc::new("x", DType::F32, [n]));
    let x_copy = gb.add_tensor(TensorDesc::new("x_copy", DType::F32, [n]));
    let z = gb.add_tensor(TensorDesc::new("z", DType::F32, [n]));
    let y = gb.add_tensor(TensorDesc::new("y", DType::F32, [n]));
    gb.add_node(OpKind::Copy, &[x], &[x_copy]);
    gb.add_node(OpKind::Add, &[x_copy, z], &[y]);
    gb.mark_input(x);
    gb.mark_input(z);
    gb.mark_output(y);
    let graph = gb.finish().expect("valid graph");

    // Two distinctive input arrays so a "zero-fill by mistake" bug shows up.
    let x_vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.25).collect();
    let z_vals: Vec<f32> = (0..n).map(|i| (i as f32) * -0.125 + 1.0).collect();
    let inputs = vec![
        (x, Tensor::host_f32(vec![n], x_vals.clone()).unwrap()),
        (z, Tensor::host_f32(vec![n], z_vals.clone()).unwrap()),
    ];

    let outs = run_graph(&backend, &graph, &inputs).expect("run_graph must succeed on Vulkan host");
    assert_eq!(outs.len(), 1, "graph has 1 output");
    let got = outs[0].as_f32().unwrap();
    assert_eq!(got.len(), n);
    // IEEE-754 f32 add is deterministic + associative-with-itself for pairs
    // of finite floats — GPU sum matches the host sum bit-for-bit.
    for (i, ((xv, zv), gv)) in x_vals.iter().zip(&z_vals).zip(got).enumerate() {
        let host_sum = xv + zv;
        assert_eq!(
            host_sum.to_bits(),
            gv.to_bits(),
            "run_graph(Copy → Add) diverged at index {i}: host={host_sum} got={gv}"
        );
    }
    eprintln!("run_graph(Copy → Add) matched host sum over {n} f32s");
}

/// A graph carrying an uncovered op is rejected by the [`run_graph`]
/// coverage precheck with an explicit `UnsupportedOp` — never a silent CPU
/// fallback. Host-independent (the precheck runs before any GPU dispatch).
#[test]
fn run_graph_rejects_uncovered_op_before_dispatch() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan; skipping run_graph uncovered-op test");
        return;
    };
    let mut gb = GraphBuilder::new();
    let a = gb.add_tensor(TensorDesc::new("a", DType::F32, [2, 4]));
    let b = gb.add_tensor(TensorDesc::new("b", DType::F32, [4, 3]));
    let out = gb.add_tensor(TensorDesc::new("out", DType::F32, [2, 3]));
    // MatMul is not covered by the foundation-slice Vulkan backend.
    gb.add_node(OpKind::MatMul, &[a, b], &[out]);
    gb.mark_input(a);
    gb.mark_input(b);
    gb.mark_output(out);
    let graph = gb.finish().expect("valid graph");

    let inputs = vec![
        (
            a,
            Tensor::host_f32(vec![2, 4], (0..8).map(|v| v as f32).collect()).unwrap(),
        ),
        (
            b,
            Tensor::host_f32(vec![4, 3], (0..12).map(|v| v as f32).collect()).unwrap(),
        ),
    ];
    let err = run_graph(&backend, &graph, &inputs).unwrap_err();
    assert!(
        matches!(err, VokraError::UnsupportedOp(_)),
        "run_graph with MatMul must be UnsupportedOp, got {err:?}"
    );
}

/// The minimal single-op graph — `Add` only — verifies the Vulkan backend
/// works for the simplest supported graph. Complements the two-op chain
/// above; if the two-op path passes but the single-op path fails, it's a
/// graph-executor bug rather than an op-dispatch bug.
#[test]
fn run_graph_single_add_matches_host_sum() {
    let Ok(backend) = VulkanBackend::new() else {
        return;
    };
    let local = spirv::handcrafted_add_f32::LOCAL_SIZE_X as usize;

    let mut gb = GraphBuilder::new();
    let a = gb.add_tensor(TensorDesc::new("a", DType::F32, [local]));
    let b = gb.add_tensor(TensorDesc::new("b", DType::F32, [local]));
    let c = gb.add_tensor(TensorDesc::new("c", DType::F32, [local]));
    gb.add_node(OpKind::Add, &[a, b], &[c]);
    gb.mark_input(a);
    gb.mark_input(b);
    gb.mark_output(c);
    let graph = gb.finish().expect("valid graph");

    let a_vals: Vec<f32> = (0..local).map(|i| (i as f32) * 0.5).collect();
    let b_vals: Vec<f32> = (0..local).map(|i| -(i as f32) * 0.25).collect();
    let inputs = vec![
        (a, Tensor::host_f32(vec![local], a_vals.clone()).unwrap()),
        (b, Tensor::host_f32(vec![local], b_vals.clone()).unwrap()),
    ];
    let outs = run_graph(&backend, &graph, &inputs).expect("run_graph must succeed on Vulkan host");
    let got = outs[0].as_f32().unwrap();
    for (i, ((av, bv), gv)) in a_vals.iter().zip(&b_vals).zip(got).enumerate() {
        let host = av + bv;
        assert_eq!(
            host.to_bits(),
            gv.to_bits(),
            "single-Add diverged at index {i}: host={host} got={gv}"
        );
    }
}

/// Off-target (macOS / iOS / WASM / default-features): `VulkanBackend::new()`
/// returns `BackendUnavailable`, which by contract means `run_graph` on this
/// backend is unreachable — the caller must skip. This test locks in the
/// off-target skip path.
#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
#[test]
fn run_graph_skip_is_honest_off_target() {
    match VulkanBackend::new() {
        Err(VokraError::BackendUnavailable(_)) => {
            // Expected: no libvulkan → skip run_graph. This is exactly what
            // the FR-EX-08 contract prescribes — the *caller* chooses a
            // different backend, we do not silently fall back here.
        }
        Ok(_) => panic!("VulkanBackend::new() must fail off-target / feature off"),
        Err(other) => panic!("expected BackendUnavailable, got {other}"),
    }
}
