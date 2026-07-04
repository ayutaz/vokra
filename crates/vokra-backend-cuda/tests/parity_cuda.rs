//! CUDA ⇄ CPU GEMM numerical parity (M2-03-T18/T19; NFR-QL-01, FP32
//! `atol = 0.01`).
//!
//! The CPU backend's `kernels::gemm_f32` is the differential oracle. Every test
//! here is **device-gated**: it constructs a [`CudaContext`], and if that fails
//! (no NVIDIA GPU — e.g. on the Apple Mac this crate is authored on) it prints a
//! skip note and returns. The `Ok` branch — the real GPU comparison — is meant
//! to run on the **vast.ai RTX 4090** GPU runner (M2-03-T24/T25), NOT on this
//! machine.

use vokra_backend_cpu::kernels;
use vokra_backend_cuda::CudaContext;

/// Largest absolute CUDA⇄CPU element difference; must stay within the FP32
/// parity bound (NFR-QL-01, `atol = 0.01`).
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn gemm_matches_cpu_within_fp32_tolerance() {
    let Ok(ctx) = CudaContext::new() else {
        eprintln!("no CUDA device; skipping GEMM parity (run on vast.ai RTX 4090)");
        return;
    };

    // A few non-square shapes with and without bias.
    let cases = [
        (1usize, 1usize, 1usize),
        (2, 3, 4),
        (5, 7, 3),
        (16, 16, 16),
        (33, 9, 17),
    ];
    for (m, n, k) in cases {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.021).cos()).collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 0.5).collect();

        for use_bias in [false, true] {
            let bias_arg = use_bias.then_some(bias.as_slice());

            let mut cpu = vec![0.0f32; m * n];
            kernels::gemm_f32(m, n, k, &a, &b, bias_arg, &mut cpu).unwrap();

            let mut gpu = vec![0.0f32; m * n];
            ctx.gemm_f32(m, n, k, &a, &b, bias_arg, &mut gpu).unwrap();

            let diff = max_abs_diff(&cpu, &gpu);
            assert!(
                diff <= 0.01,
                "CUDA GEMM {m}x{n}x{k} (bias={use_bias}) diff {diff} exceeds atol 0.01"
            );
        }
    }
    eprintln!("CUDA GEMM parity OK on this GPU");
}

#[test]
fn gemm_rejects_bad_shapes_without_a_device_touch() {
    // Shape validation happens before any GPU work, but the guard still needs a
    // context to call the method on — so this is device-gated like the rest.
    let Ok(ctx) = CudaContext::new() else {
        eprintln!("no CUDA device; skipping GEMM shape-validation (run on vast.ai)");
        return;
    };
    // a must be m*k = 4 long; pass 2 → explicit InvalidArgument (not a GPU fault).
    let mut out = [0.0f32; 4];
    let err = ctx
        .gemm_f32(2, 2, 2, &[1.0, 2.0], &[1.0; 4], None, &mut out)
        .unwrap_err();
    assert!(matches!(err, vokra_core::VokraError::InvalidArgument(_)));
}
