//! Metal graph-execution parity (Phase 2 of the GPU execution architecture).
//!
//! - **V2** — `vokra_core::run_graph` on the [`MetalBackend`] evaluates a MatMul
//!   graph on the GPU and must match the *same graph* on the [`CpuBackend`]
//!   within the FP32 bound (NFR-QL-01, `atol = 0.01`; the observed error is
//!   ~1e-5). Both runs go through the identical engine, so this isolates the GPU
//!   GEMM path.
//! - **cc-27** — the `Mul` and `Copy` arms. `Mul` is asserted *bit-identical*
//!   to the CPU graph (with a negative control proving the oracle
//!   discriminates); `Copy` is asserted a bit-exact identity, including for
//!   `-0.0` and a subnormal, since the CPU backend has no `Copy` arm to compare
//!   against.
//! - **V5-metal** — a graph containing an op the Metal backend does not cover
//!   (`Stft`) surfaces as an explicit [`VokraError::UnsupportedOp`] from the
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

/// Element-wise `Add` graph on Metal must match the same graph on the CPU
/// backend within `atol`. The Metal arm routes into the `vokra_add_assign_f32`
/// kernel (via `MetalContext::residual_add_dev`) — the exact FP32 add the CPU
/// `kernels::add_f32` performs, so the two graph runs agree to the FP32 bound
/// (observed: bit-identical, since the add is a single FP32 op per element).
#[test]
fn add_graph_metal_matches_cpu_graph() {
    let Ok(metal) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping Metal Add graph parity");
        return;
    };
    let cpu = CpuBackend::new();

    // Powers of two plus a ragged non-16-multiple length and a thin vector.
    for &n in &[1usize, 4, 15, 64, 257] {
        let av = rand_vec(0x51 ^ n as u64, n);
        let bv = rand_vec(0x71 ^ n as u64, n);

        let mut b = GraphBuilder::new();
        let x = b.add_tensor(TensorDesc::new("x", DType::F32, [n]));
        let y = b.add_tensor(TensorDesc::new("y", DType::F32, [n]));
        let z = b.add_tensor(TensorDesc::new("z", DType::F32, [n]));
        b.add_node(OpKind::Add, &[x, y], &[z]);
        b.mark_input(x);
        b.mark_input(y);
        b.mark_output(z);
        let graph = b.finish().expect("valid graph");

        let inputs = [
            (x, Tensor::host_f32(vec![n], av).unwrap()),
            (y, Tensor::host_f32(vec![n], bv).unwrap()),
        ];
        let gpu = run_graph(&metal, &graph, &inputs).expect("metal Add graph run");
        let host = run_graph(&cpu, &graph, &inputs).expect("cpu Add graph run");

        assert_eq!(gpu[0].shape, vec![n]);
        let d = max_abs_diff(gpu[0].as_f32().unwrap(), host[0].as_f32().unwrap());
        eprintln!("Add graph parity  n={n:<5} max|Δ| vs cpu-graph = {d:.3e}");
        assert!(
            d <= ATOL,
            "metal Add graph max|Δ| {d:.3e} exceeds atol {ATOL} (n={n})"
        );
    }
}

/// Row-wise `Softmax` graph on Metal must match the same graph on the CPU
/// backend within `atol`. Both route into their `softmax_f32` kernel (same
/// signature/semantics: normalise over the innermost axis), so the GPU softmax
/// path is isolated here.
#[test]
fn softmax_graph_metal_matches_cpu_graph() {
    let Ok(metal) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping Metal Softmax graph parity");
        return;
    };
    let cpu = CpuBackend::new();

    // (rows, cols): square, ragged, single-row, and a wide single-column edge.
    for &(rows, cols) in &[(1usize, 5usize), (2, 3), (7, 17), (16, 16), (33, 9)] {
        let xv = rand_vec(0x91 ^ ((rows * 131 + cols) as u64), rows * cols);

        let mut b = GraphBuilder::new();
        let x = b.add_tensor(TensorDesc::new("x", DType::F32, [rows, cols]));
        let y = b.add_tensor(TensorDesc::new("y", DType::F32, [rows, cols]));
        b.add_node(OpKind::Softmax, &[x], &[y]);
        b.mark_input(x);
        b.mark_output(y);
        let graph = b.finish().expect("valid graph");

        let inputs = [(x, Tensor::host_f32(vec![rows, cols], xv).unwrap())];
        let gpu = run_graph(&metal, &graph, &inputs).expect("metal Softmax graph run");
        let host = run_graph(&cpu, &graph, &inputs).expect("cpu Softmax graph run");

        assert_eq!(gpu[0].shape, vec![rows, cols]);
        let d = max_abs_diff(gpu[0].as_f32().unwrap(), host[0].as_f32().unwrap());
        eprintln!(
            "Softmax graph parity  rows={rows:<4} cols={cols:<4} max|Δ| vs cpu-graph = {d:.3e}"
        );
        assert!(
            d <= ATOL,
            "metal Softmax graph max|Δ| {d:.3e} exceeds atol {ATOL} (rows={rows} cols={cols})"
        );
    }
}

/// cc-27: element-wise `Mul` graph on Metal must match the same graph on the
/// CPU backend. The Metal arm routes into the `vokra_mul_f32` kernel (via
/// `MetalContext::mul_dev`) — one FP32 multiply per element, the same single
/// rounding the CPU `kernels::mul_f32` applies, with no reduction order to
/// disagree about.
///
/// # Tolerance
///
/// Asserted **bit-identical** (`max|Δ| == 0`), not merely within `ATOL`. That
/// is the measured result on this M1 across every shape below, and it is what
/// the arithmetic predicts: a lone FP32 product is correctly rounded on both
/// sides. Pinning the measured value rather than the loose FP32 ceiling means a
/// future kernel change that silently starts approximating (e.g. an f16 path)
/// fails here instead of hiding under `atol = 0.01`.
///
/// Operands come from `rand_vec` (normal range, |x| < 1), so the fast-math
/// denormal flush MSL permits is out of scope for this assertion by
/// construction — see `context.rs`'s kernel comment.
#[test]
fn mul_graph_metal_matches_cpu_graph() {
    let Ok(metal) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping Metal Mul graph parity");
        return;
    };
    let cpu = CpuBackend::new();

    // Powers of two plus a ragged non-16-multiple length and a thin vector.
    for &n in &[1usize, 4, 15, 64, 257] {
        let av = rand_vec(0x13 ^ n as u64, n);
        let bv = rand_vec(0x37 ^ n as u64, n);

        let mut b = GraphBuilder::new();
        let x = b.add_tensor(TensorDesc::new("x", DType::F32, [n]));
        let y = b.add_tensor(TensorDesc::new("y", DType::F32, [n]));
        let z = b.add_tensor(TensorDesc::new("z", DType::F32, [n]));
        b.add_node(OpKind::Mul, &[x, y], &[z]);
        b.mark_input(x);
        b.mark_input(y);
        b.mark_output(z);
        let graph = b.finish().expect("valid graph");

        let inputs = [
            (x, Tensor::host_f32(vec![n], av).unwrap()),
            (y, Tensor::host_f32(vec![n], bv).unwrap()),
        ];
        let gpu = run_graph(&metal, &graph, &inputs).expect("metal Mul graph run");
        let host = run_graph(&cpu, &graph, &inputs).expect("cpu Mul graph run");

        assert_eq!(gpu[0].shape, vec![n]);
        let d = max_abs_diff(gpu[0].as_f32().unwrap(), host[0].as_f32().unwrap());
        eprintln!("Mul graph parity  n={n:<5} max|Δ| vs cpu-graph = {d:.3e}");
        assert_eq!(
            d, 0.0,
            "metal Mul graph max|Δ| {d:.3e} is not bit-identical to the CPU graph (n={n})"
        );
    }
}

/// Negative control for [`mul_graph_metal_matches_cpu_graph`]: the same
/// comparison against a *deliberately wrong* reference (the element-wise sum)
/// must fail the bit-identity check. Without this, a `Mul` kernel that
/// accidentally computed something else — or a comparison that silently
/// compared a buffer with itself — would still "pass" the parity test above.
///
/// The chosen operands make the discrimination unambiguous: `a + b` and `a * b`
/// differ by more than `ATOL` on this data, so the control is not riding on
/// floating-point noise.
#[test]
fn mul_graph_parity_rejects_a_wrong_reference() {
    let Ok(metal) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping Metal Mul negative control");
        return;
    };
    let cpu = CpuBackend::new();
    let n = 64usize;
    let av = rand_vec(0x13 ^ n as u64, n);
    let bv = rand_vec(0x37 ^ n as u64, n);

    let build = |op: OpKind| {
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(TensorDesc::new("x", DType::F32, [n]));
        let y = b.add_tensor(TensorDesc::new("y", DType::F32, [n]));
        let z = b.add_tensor(TensorDesc::new("z", DType::F32, [n]));
        b.add_node(op, &[x, y], &[z]);
        b.mark_input(x);
        b.mark_input(y);
        b.mark_output(z);
        (b.finish().expect("valid graph"), x, y)
    };

    let (mul_graph, mx, my) = build(OpKind::Mul);
    let (add_graph, ax, ay) = build(OpKind::Add);
    let mul_inputs = [
        (mx, Tensor::host_f32(vec![n], av.clone()).unwrap()),
        (my, Tensor::host_f32(vec![n], bv.clone()).unwrap()),
    ];
    let add_inputs = [
        (ax, Tensor::host_f32(vec![n], av).unwrap()),
        (ay, Tensor::host_f32(vec![n], bv).unwrap()),
    ];

    let gpu_mul = run_graph(&metal, &mul_graph, &mul_inputs).expect("metal Mul graph run");
    let cpu_add = run_graph(&cpu, &add_graph, &add_inputs).expect("cpu Add graph run");

    let d = max_abs_diff(gpu_mul[0].as_f32().unwrap(), cpu_add[0].as_f32().unwrap());
    eprintln!("Mul negative control  max|Δ| vs cpu ADD = {d:.3e} (must exceed atol {ATOL})");
    assert!(
        d > ATOL,
        "negative control did not discriminate: Metal Mul is within {ATOL} of the CPU SUM \
         (max|Δ| {d:.3e}) — the parity oracle would not catch a wrong kernel"
    );
}

/// cc-27: a `Copy` graph on Metal reproduces its input exactly. The CPU backend
/// has **no** `Copy` arm (only Vulkan / WebGPU / — now — Metal do), so the
/// oracle here is the identity rather than a cross-backend differential: an
/// FP32 move performs no arithmetic, so any difference at all is a bug.
///
/// The input deliberately includes a negative zero and a subnormal: a move must
/// preserve both bit patterns, which a kernel that (say) added zero or
/// round-tripped through a lower precision would not. Compared on raw bits so
/// `-0.0 == 0.0` cannot mask a sign flip.
#[test]
fn copy_graph_metal_is_bit_exact_identity() {
    let Ok(metal) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping Metal Copy graph identity");
        return;
    };

    for &n in &[1usize, 4, 15, 64, 257] {
        let mut xv = rand_vec(0x5A ^ n as u64, n);
        // Pin the two adversarial bit patterns in every shape that has room.
        if n >= 2 {
            xv[0] = -0.0;
            xv[1] = f32::from_bits(1); // smallest positive subnormal
        }

        let mut b = GraphBuilder::new();
        let x = b.add_tensor(TensorDesc::new("x", DType::F32, [n]));
        let y = b.add_tensor(TensorDesc::new("y", DType::F32, [n]));
        b.add_node(OpKind::Copy, &[x], &[y]);
        b.mark_input(x);
        b.mark_output(y);
        let graph = b.finish().expect("valid graph");

        let inputs = [(x, Tensor::host_f32(vec![n], xv.clone()).unwrap())];
        let gpu = run_graph(&metal, &graph, &inputs).expect("metal Copy graph run");

        assert_eq!(gpu[0].shape, vec![n]);
        let got = gpu[0].as_f32().unwrap();
        let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
        let want_bits: Vec<u32> = xv.iter().map(|v| v.to_bits()).collect();
        eprintln!(
            "Copy graph identity  n={n:<5} bit-exact = {}",
            got_bits == want_bits
        );
        assert_eq!(
            got_bits, want_bits,
            "metal Copy graph is not a bit-exact identity (n={n})"
        );
    }
}

/// V5-metal: an op with no Metal kernel is an explicit `UnsupportedOp` — the
/// engine's whole-graph coverage precheck rejects it before any evaluation,
/// with no silent CPU fallback (FR-EX-08).
///
/// `Mul` used to be the example here; cc-27 wired it, so the probe moved to
/// `Stft`, an audio-dialect op with no Metal graph kernel. The property under
/// test is unchanged: an uncovered op errors rather than quietly rerouting.
#[test]
fn unsupported_op_graph_is_explicit_unsupported() {
    let Ok(metal) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping unsupported-op graph test");
        return;
    };

    let mut b = GraphBuilder::new();
    let x = b.add_tensor(TensorDesc::new("x", DType::F32, [4]));
    let z = b.add_tensor(TensorDesc::new("z", DType::F32, [4]));
    b.add_node(
        OpKind::Stft(vokra_core::ir::graph::StftAttrs::new(400, 160)),
        &[x],
        &[z],
    );
    b.mark_input(x);
    b.mark_output(z);
    let graph = b.finish().expect("valid graph");

    let err = run_graph(&metal, &graph, &[(x, Tensor::zeros_f32(vec![4]))]).unwrap_err();
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
    // Covered ops (wired to real Metal kernels).
    assert!(metal.supports(&OpKind::MatMul), "MatMul is covered");
    assert!(metal.supports(&OpKind::Add), "Add is covered");
    assert!(metal.supports(&OpKind::Softmax), "Softmax is covered");
    // cc-27 wired `Mul` / `Copy` to real MSL kernels, so the Metal graph arm
    // now covers the same op set as the CUDA / Vulkan / WebGPU arms.
    assert!(metal.supports(&OpKind::Mul), "Mul is covered (cc-27)");
    assert!(metal.supports(&OpKind::Copy), "Copy is covered (cc-27)");

    // An op with no Metal kernel is still uncovered on both surfaces.
    let stft = OpKind::Stft(vokra_core::ir::graph::StftAttrs::new(400, 160));
    assert!(!metal.supports(&stft), "Stft has no Metal graph kernel");
    let a = Tensor::zeros_f32(vec![2, 2]);
    let err = metal.eval_op(&stft, &[&a]).unwrap_err();
    assert!(matches!(err, VokraError::UnsupportedOp(_)));
}
