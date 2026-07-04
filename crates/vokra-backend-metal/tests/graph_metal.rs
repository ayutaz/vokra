//! Metal graph-execution parity (Phase 2 of the GPU execution architecture).
//!
//! - **V2** — `vokra_core::run_graph` on the [`MetalBackend`] evaluates a MatMul
//!   graph on the GPU and must match the *same graph* on the [`CpuBackend`]
//!   within the FP32 bound (NFR-QL-01, `atol = 0.01`; the observed error is
//!   ~1e-5). Both runs go through the identical engine, so this isolates the GPU
//!   GEMM path.
//! - **V5-metal** — a graph containing an op the Metal backend does not cover
//!   (`Add`) surfaces as an explicit [`VokraError::UnsupportedOp`] from the
//!   engine's coverage precheck — never a silent CPU fallback (FR-EX-08).
//!
//! Device-gated exactly like `parity_metal.rs`: with no Metal device the suite
//! skips (returns) rather than fails; the macOS Metal CI job (M2-01-T21) runs it
//! for real.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use vokra_backend_cpu::CpuBackend;
use vokra_backend_metal::{MetalBackend, vokra_metal_probe};
use vokra_core::{Backend, DType, GraphBuilder, OpKind, Tensor, TensorDesc, VokraError, run_graph};

/// NFR-QL-01 FP32 parity ceiling.
const ATOL: f32 = 0.01;

/// Deterministic pseudo-random f32 in roughly [-1, 1) (xorshift64*), matching
/// the `parity_metal.rs` generator so inputs are reproducible.
fn rand_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
            bits as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// V2: a two-MatMul graph `y = (x @ w1) @ w2` — exercising the engine's
/// intermediate-tensor threading across two GPU dispatches — evaluated on Metal
/// must equal the same graph on the CPU backend within `atol`. Returns the
/// worst per-shape max|Δ| for the run-wide summary.
#[test]
fn matmul_graph_metal_matches_cpu_graph() {
    let caps = match vokra_metal_probe() {
        Ok(caps) => caps,
        Err(e) => {
            eprintln!("no Metal device ({e}); skipping Metal graph parity");
            return;
        }
    };
    eprintln!("Metal device: {}", caps.summary());

    let metal = MetalBackend::new().expect("build Metal backend");
    let cpu = CpuBackend::new();

    // Square powers of two, a ragged non-16-multiple case, and thin vectors.
    let shapes = [
        (2usize, 4usize, 8usize, 3usize),
        (16, 16, 16, 16),
        (33, 17, 9, 5),
        (1, 64, 32, 1),
        (64, 64, 64, 64),
    ];

    let mut global_worst = 0.0f32;
    for &(m, k, p, n) in &shapes {
        let x = rand_vec(0x11 ^ ((m * 131 + k) as u64), m * k);
        let w1 = rand_vec(0x22 ^ ((k * 17 + p) as u64), k * p);
        let w2 = rand_vec(0x33 ^ ((p * 29 + n) as u64), p * n);

        let mut b = GraphBuilder::new();
        let xt = b.add_tensor(TensorDesc::new("x", DType::F32, [m, k]));
        let w1t = b.add_tensor(TensorDesc::new("w1", DType::F32, [k, p]));
        let w2t = b.add_tensor(TensorDesc::new("w2", DType::F32, [p, n]));
        let tt = b.add_tensor(TensorDesc::new("t", DType::F32, [m, p]));
        let yt = b.add_tensor(TensorDesc::new("y", DType::F32, [m, n]));
        b.add_node(OpKind::MatMul, &[xt, w1t], &[tt]);
        b.add_node(OpKind::MatMul, &[tt, w2t], &[yt]);
        b.mark_input(xt);
        b.mark_output(yt);
        let graph = b.finish().expect("valid graph");

        // `run_graph` clones each supplied value internally, so one input set
        // drives both backends (the tensor ids are Copy).
        let inputs = [
            (xt, Tensor::host_f32(vec![m, k], x).unwrap()),
            (w1t, Tensor::host_f32(vec![k, p], w1).unwrap()),
            (w2t, Tensor::host_f32(vec![p, n], w2).unwrap()),
        ];

        let gpu = run_graph(&metal, &graph, &inputs).expect("metal graph run");
        let host = run_graph(&cpu, &graph, &inputs).expect("cpu graph run");

        assert_eq!(gpu.len(), 1, "single declared output");
        assert_eq!(gpu[0].shape, vec![m, n]);
        assert_eq!(host[0].shape, vec![m, n]);

        let d = max_abs_diff(gpu[0].as_f32().unwrap(), host[0].as_f32().unwrap());
        eprintln!(
            "graph parity  m={m:<4} k={k:<4} p={p:<4} n={n:<4}  max|Δ| vs cpu-graph = {d:.3e}"
        );
        assert!(
            d <= ATOL,
            "metal graph vs cpu graph max|Δ| {d:.3e} exceeds atol {ATOL} (m={m} k={k} p={p} n={n})"
        );
        global_worst = global_worst.max(d);
    }

    eprintln!("Metal graph parity: global max|Δ| = {global_worst:.3e} (atol = {ATOL})");
    assert!(global_worst <= ATOL);
}

/// V5-metal: an uncovered op in the graph is an explicit `UnsupportedOp` — the
/// engine's whole-graph coverage precheck rejects it before any evaluation,
/// with no silent CPU fallback (FR-EX-08).
#[test]
fn unsupported_op_graph_is_explicit_unsupported() {
    let Ok(metal) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping unsupported-op graph test");
        return;
    };

    // `Add` is covered by the CPU backend but not this Metal slice → the Metal
    // run must be an explicit error, not a quiet reroute onto the CPU.
    let mut b = GraphBuilder::new();
    let x = b.add_tensor(TensorDesc::new("x", DType::F32, [4]));
    let y = b.add_tensor(TensorDesc::new("y", DType::F32, [4]));
    let z = b.add_tensor(TensorDesc::new("z", DType::F32, [4]));
    b.add_node(OpKind::Add, &[x, y], &[z]);
    b.mark_input(x);
    b.mark_input(y);
    b.mark_output(z);
    let graph = b.finish().expect("valid graph");

    let err = run_graph(
        &metal,
        &graph,
        &[
            (x, Tensor::zeros_f32(vec![4])),
            (y, Tensor::zeros_f32(vec![4])),
        ],
    )
    .unwrap_err();
    assert!(matches!(err, VokraError::UnsupportedOp(_)));
}

/// Defense in depth (plan R1): calling `eval_op` directly — bypassing the
/// engine's precheck — must itself reject an uncovered op with `UnsupportedOp`,
/// so `supports()` and `eval_op()` stay in sync and never silently fall back.
#[test]
fn eval_op_direct_rejects_uncovered_op() {
    let Ok(metal) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping direct eval_op coverage test");
        return;
    };
    assert!(metal.supports(&OpKind::MatMul), "MatMul is the covered op");
    assert!(
        !metal.supports(&OpKind::Add),
        "Add is uncovered in this slice"
    );

    let a = Tensor::zeros_f32(vec![2, 2]);
    let err = metal.eval_op(&OpKind::Add, &[&a, &a]).unwrap_err();
    assert!(matches!(err, VokraError::UnsupportedOp(_)));
}
