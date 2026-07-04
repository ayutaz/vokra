//! Metal GEMM numerical parity (M2-01-T17/T18): the FP32 GPU GEMM vs the CPU
//! backend's `gemm_f32` kernel (M0-08), the same differential oracle the
//! scalar⇔SIMD harness uses. Ceiling is the NFR-QL-01 FP32 bound `atol = 0.01`
//! (the observed error is far smaller and logged per shape).
//!
//! Runs only where a Metal device is available: [`vokra_metal_probe`] gates the
//! suite, so a non-Apple / Metal-less host skips rather than fails (the same
//! "runner must have the device" policy as the GGUF-gated model parity tests).
//! The macOS Metal CI job (M2-01-T21) runs it for real.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use vokra_backend_cpu::kernels as cpu;
use vokra_backend_metal::{MetalContext, vokra_metal_probe};

/// NFR-QL-01 FP32 parity ceiling.
const ATOL: f32 = 0.01;

/// Deterministic pseudo-random f32 in roughly [-1, 1) (xorshift64*), matching
/// the CPU backend's bench/differential generator so inputs are reproducible.
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

/// Naive f64-accumulated GEMM — an independent second oracle (so parity is not
/// judged solely against the CPU backend's own FMA reduction order).
fn naive_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += a[i * k + l] as f64 * b[l * n + j] as f64;
            }
            if let Some(bias) = bias {
                acc += bias[j] as f64;
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// One (m, k, n) case with optional bias, checked against both oracles.
fn check_case(ctx: &MetalContext, m: usize, k: usize, n: usize, with_bias: bool) -> f32 {
    let a = rand_vec(0x1234 ^ ((m * 131 + k) as u64), m * k);
    let b = rand_vec(0x9E37 ^ ((k * 17 + n) as u64), k * n);
    let bias_vec = with_bias.then(|| rand_vec(0xABCD ^ (n as u64), n));
    let bias = bias_vec.as_deref();

    // GPU result.
    let mut gpu = vec![f32::NAN; m * n];
    ctx.gemm_f32(m, n, k, &a, &b, bias, &mut gpu)
        .expect("metal gemm must succeed");

    // CPU backend oracle (M0-08) + independent naive f64 oracle.
    let mut cpu_out = vec![0.0f32; m * n];
    cpu::gemm_f32(m, n, k, &a, &b, bias, &mut cpu_out).expect("cpu gemm oracle");
    let naive = naive_gemm(m, n, k, &a, &b, bias);

    let d_cpu = max_abs_diff(&gpu, &cpu_out);
    let d_naive = max_abs_diff(&gpu, &naive);
    let worst = d_cpu.max(d_naive);
    eprintln!(
        "GEMM parity  m={m:<4} k={k:<4} n={n:<4} bias={:<5}  max|Δ| vs cpu={d_cpu:.3e}  vs naive={d_naive:.3e}",
        with_bias
    );
    assert!(
        d_cpu <= ATOL,
        "metal vs cpu max|Δ| {d_cpu:.3e} exceeds atol {ATOL} (m={m} k={k} n={n} bias={with_bias})"
    );
    assert!(
        d_naive <= ATOL,
        "metal vs naive max|Δ| {d_naive:.3e} exceeds atol {ATOL} (m={m} k={k} n={n} bias={with_bias})"
    );
    worst
}

#[test]
fn gemm_metal_matches_cpu_and_naive_across_shapes() {
    // Gate: only run with a real Metal device (else skip, do not fail).
    let caps = match vokra_metal_probe() {
        Ok(caps) => caps,
        Err(e) => {
            eprintln!("no Metal device ({e}); skipping Metal GEMM parity");
            return;
        }
    };
    eprintln!("Metal device: {}", caps.summary());

    let ctx = MetalContext::new().expect("build Metal context");

    // Square powers of two, ragged non-multiples of the 16x16 threadgroup, thin
    // vectors (n=1 / m=1), identity-ish tiny cases, and a large-K reduction.
    let shapes = [
        (1usize, 1usize, 1usize),
        (2, 3, 4),
        (4, 4, 4),
        (8, 8, 8),
        (16, 16, 16),
        (16, 32, 24),
        (33, 17, 9),
        (1, 128, 64),
        (64, 1, 64),
        (64, 64, 64),
        (37, 100, 41),
        (128, 128, 128),
    ];

    let mut global_worst = 0.0f32;
    for &(m, k, n) in &shapes {
        global_worst = global_worst.max(check_case(&ctx, m, k, n, false));
    }
    // Bias path on a representative subset.
    for &(m, k, n) in &[(2usize, 3usize, 4usize), (16, 16, 16), (37, 100, 41)] {
        global_worst = global_worst.max(check_case(&ctx, m, k, n, true));
    }

    eprintln!("Metal GEMM parity: global max|Δ| = {global_worst:.3e} (atol = {ATOL})");
    assert!(global_worst <= ATOL);
}

/// Shape mismatches / zero dims are explicit `InvalidArgument`, not a GPU fault.
#[test]
fn gemm_rejects_bad_shapes_explicitly() {
    let Ok(ctx) = MetalContext::new() else {
        eprintln!("no Metal device; skipping shape-validation test");
        return;
    };
    // a should be m*k = 4 long, but is 2.
    let mut out = [0.0f32; 4];
    assert!(
        ctx.gemm_f32(2, 2, 2, &[1.0, 2.0], &[1.0; 4], None, &mut out)
            .is_err()
    );
    // zero dimension.
    assert!(
        ctx.gemm_f32(0, 2, 2, &[], &[1.0; 4], None, &mut [0.0; 0])
            .is_err()
    );
    // bias length != n.
    assert!(
        ctx.gemm_f32(2, 2, 2, &[1.0; 4], &[1.0; 4], Some(&[1.0; 1]), &mut out)
            .is_err()
    );
}
