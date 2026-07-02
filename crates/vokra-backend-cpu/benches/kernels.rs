//! Reference benchmarks: scalar vs the host SIMD path (M0-08-T17).
//!
//! `harness = false` + `std::time::Instant` — no external bench crate, so the
//! workspace stays dependency-free (NFR-LC-02 / NFR-DS-02); this mirrors
//! M0-04-T19's hand-rolled measurement style. Results are **reference values
//! only**: M0 defines no CPU-kernel performance gate (Whisper base RTF < 0.3
//! is NFR-PF-01 = v0.1 MVP; the 5% regression check is NFR-PF-13), so nothing
//! here asserts a threshold.
//!
//! Run with `cargo bench -p vokra-backend-cpu`.

use std::hint::black_box;
use std::time::Instant;

use vokra_backend_cpu::kernels;
use vokra_backend_cpu::{CpuFeatures, IsaPath};

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

/// Returns nanoseconds per iteration for `iters` runs of `f`.
fn time<F: FnMut()>(iters: u32, mut f: F) -> f64 {
    // Warm-up.
    f();
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_nanos() as f64 / iters as f64
}

fn report(name: &str, scalar_ns: f64, simd_ns: f64, simd: IsaPath) {
    let speedup = if simd_ns > 0.0 {
        scalar_ns / simd_ns
    } else {
        f64::NAN
    };
    println!(
        "{name:<28} scalar {scalar_ns:>10.1} ns   {simd:<6} {simd_ns:>10.1} ns   x{speedup:.2}"
    );
}

fn main() {
    let simd = CpuFeatures::detect().best_isa();
    println!("vokra-backend-cpu kernel benchmarks (reference only; no gate)");
    println!("host SIMD path: {simd}\n");

    // GEMM 128x128x128.
    {
        let (m, n, k) = (128, 128, 128);
        let a = rand_vec(1, m * k);
        let b = rand_vec(2, k * n);
        let bias = rand_vec(3, n);
        let mut out = vec![0.0; m * n];
        let s = time(50, || {
            kernels::gemm_f32_on(IsaPath::Scalar, m, n, k, &a, &b, Some(&bias), &mut out).unwrap();
            black_box(&out);
        });
        let v = time(50, || {
            kernels::gemm_f32_on(simd, m, n, k, &a, &b, Some(&bias), &mut out).unwrap();
            black_box(&out);
        });
        report("gemm 128x128x128", s, v, simd);
    }

    // conv1d 32ch -> 64ch, len 512, k=5.
    {
        let (in_ch, in_len, out_ch, kernel, stride, pad) = (32, 512, 64, 5, 1, 2);
        let input = rand_vec(4, in_ch * in_len);
        let weight = rand_vec(5, out_ch * in_ch * kernel);
        let bias = rand_vec(6, out_ch);
        let out_len = (in_len + 2 * pad - kernel) / stride + 1;
        let mut out = vec![0.0; out_ch * out_len];
        let s = time(20, || {
            kernels::conv1d_f32_on(
                IsaPath::Scalar,
                &input,
                in_ch,
                in_len,
                &weight,
                out_ch,
                kernel,
                Some(&bias),
                stride,
                pad,
                &mut out,
            )
            .unwrap();
            black_box(&out);
        });
        let v = time(20, || {
            kernels::conv1d_f32_on(
                simd,
                &input,
                in_ch,
                in_len,
                &weight,
                out_ch,
                kernel,
                Some(&bias),
                stride,
                pad,
                &mut out,
            )
            .unwrap();
            black_box(&out);
        });
        report("conv1d 32->64 len512 k5", s, v, simd);
    }

    // softmax 64 rows x 1024 cols.
    {
        let (rows, cols) = (64, 1024);
        let input = rand_vec(7, rows * cols);
        let mut out = vec![0.0; rows * cols];
        let s = time(50, || {
            kernels::softmax_f32_on(IsaPath::Scalar, &input, &mut out, rows, cols).unwrap();
            black_box(&out);
        });
        let v = time(50, || {
            kernels::softmax_f32_on(simd, &input, &mut out, rows, cols).unwrap();
            black_box(&out);
        });
        report("softmax 64x1024", s, v, simd);
    }

    // layer_norm 64 rows x 1024 cols.
    {
        let (rows, cols) = (64, 1024);
        let input = rand_vec(8, rows * cols);
        let gamma = rand_vec(9, cols);
        let beta = rand_vec(10, cols);
        let eps = kernels::LAYER_NORM_DEFAULT_EPS;
        let mut out = vec![0.0; rows * cols];
        let s = time(50, || {
            kernels::layer_norm_f32_on(
                IsaPath::Scalar,
                &input,
                &mut out,
                rows,
                cols,
                &gamma,
                &beta,
                eps,
            )
            .unwrap();
            black_box(&out);
        });
        let v = time(50, || {
            kernels::layer_norm_f32_on(simd, &input, &mut out, rows, cols, &gamma, &beta, eps)
                .unwrap();
            black_box(&out);
        });
        report("layer_norm 64x1024", s, v, simd);
    }

    // elementwise add over 1<<20 elements.
    {
        let n = 1 << 20;
        let a = rand_vec(11, n);
        let b = rand_vec(12, n);
        let mut out = vec![0.0; n];
        let s = time(50, || {
            kernels::add_f32_on(IsaPath::Scalar, &a, &b, &mut out).unwrap();
            black_box(&out);
        });
        let v = time(50, || {
            kernels::add_f32_on(simd, &a, &b, &mut out).unwrap();
            black_box(&out);
        });
        report("add 1M", s, v, simd);
    }
}
