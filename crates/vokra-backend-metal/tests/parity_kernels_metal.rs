//! Metal numerical parity for the Phase-4 kernels (M2-01 T09-T13): the FP32 GPU
//! `gemv` / `softmax` / `layer_norm` / `gelu` / `conv1d` vs the `vokra-backend-cpu`
//! kernels (M0-08) that are the same differential oracle the scalar⇔SIMD harness
//! uses. Ceiling is the NFR-QL-01 FP32 bound `atol = 0.01` (the observed error is
//! far smaller and logged per shape).
//!
//! Like the GEMM parity suite, this runs only where a Metal device is available:
//! [`MetalContext::new`] gates each test, so a non-Apple / Metal-less host skips
//! rather than fails (the same policy as the GGUF-gated model parity tests). The
//! macOS Metal CI job runs it for real.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use vokra_backend_cpu::kernels as cpu;
use vokra_backend_metal::MetalContext;

/// NFR-QL-01 FP32 parity ceiling.
const ATOL: f32 = 0.01;

/// Deterministic pseudo-random f32 in roughly [-1, 1) (xorshift64*), matching the
/// GEMM parity suite's generator so inputs are reproducible.
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
    assert_eq!(x.len(), y.len(), "compared slices differ in length");
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Builds a context or prints a skip and returns `None` (no Metal device).
macro_rules! ctx_or_skip {
    ($what:literal) => {
        match MetalContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!(concat!("no Metal device (", $what, "); skipping: {}"), e);
                return;
            }
        }
    };
}

#[test]
fn gemv_metal_matches_cpu() {
    let ctx = ctx_or_skip!("gemv");
    // Thin vectors, ragged sizes, and a logits-head-class large-M reduction.
    let shapes = [
        (1usize, 1usize),
        (2, 3),
        (4, 4),
        (8, 8),
        (64, 64),
        (1, 128),
        (128, 1),
        (37, 100),
        (512, 384),
        (2048, 512),
    ];
    let mut worst = 0.0f32;
    for &(m, k) in &shapes {
        for with_bias in [false, true] {
            let a = rand_vec(0x51 ^ ((m * 131 + k) as u64), m * k);
            let x = rand_vec(0x9E ^ (k as u64), k);
            let bias_vec = with_bias.then(|| rand_vec(0xB1 ^ (m as u64), m));
            let bias = bias_vec.as_deref();

            let mut gpu = vec![f32::NAN; m];
            ctx.gemv_f32(m, k, &a, &x, bias, &mut gpu)
                .expect("metal gemv");
            let mut cpu_out = vec![0.0f32; m];
            cpu::gemv_f32(m, k, &a, &x, bias, &mut cpu_out).expect("cpu gemv");

            let d = max_abs_diff(&gpu, &cpu_out);
            eprintln!("gemv  m={m:<5} k={k:<5} bias={with_bias:<5} max|Δ|={d:.3e}");
            assert!(d <= ATOL, "gemv m={m} k={k} bias={with_bias}: {d} > {ATOL}");
            worst = worst.max(d);
        }
    }
    eprintln!("gemv Metal vs CPU: global max|Δ| = {worst:.3e} (atol {ATOL})");
}

#[test]
fn softmax_metal_matches_cpu() {
    let ctx = ctx_or_skip!("softmax");
    // rows × cols; cols up to a full 1500-key cross-attention row.
    let shapes = [
        (1usize, 1usize),
        (1, 4),
        (2, 8),
        (37, 41),
        (4, 448),
        (64, 64),
        (1, 1500),
        (8, 1500),
    ];
    let mut worst = 0.0f32;
    for &(rows, cols) in &shapes {
        let input = rand_vec(0x50F7 ^ ((rows * 97 + cols) as u64), rows * cols);
        let mut gpu = vec![f32::NAN; rows * cols];
        ctx.softmax_f32(&input, &mut gpu, rows, cols)
            .expect("metal softmax");
        let mut cpu_out = vec![0.0f32; rows * cols];
        cpu::softmax_f32(&input, &mut cpu_out, rows, cols).expect("cpu softmax");
        let d = max_abs_diff(&gpu, &cpu_out);
        eprintln!("softmax  rows={rows:<4} cols={cols:<5} max|Δ|={d:.3e}");
        assert!(d <= ATOL, "softmax rows={rows} cols={cols}: {d} > {ATOL}");
        worst = worst.max(d);
    }

    // Causal-masked row: the upper triangle is `-inf`, exactly as Whisper's
    // attention writes it. Both backends must map a masked score to a 0 weight
    // (exp(-inf) = 0) and renormalise over the visible keys.
    let t = 6usize;
    let base = rand_vec(0xCA05, t * t);
    let mut masked = base.clone();
    for i in 0..t {
        for j in (i + 1)..t {
            masked[i * t + j] = f32::NEG_INFINITY;
        }
    }
    let mut gpu = vec![f32::NAN; t * t];
    ctx.softmax_f32(&masked, &mut gpu, t, t)
        .expect("metal softmax masked");
    let mut cpu_out = vec![0.0f32; t * t];
    cpu::softmax_f32(&masked, &mut cpu_out, t, t).expect("cpu softmax masked");
    let d = max_abs_diff(&gpu, &cpu_out);
    eprintln!("softmax  causal-masked t={t} max|Δ|={d:.3e}");
    assert!(d <= ATOL, "causal-masked softmax: {d} > {ATOL}");
    // Masked entries must be exactly 0 on the GPU (not merely small).
    for i in 0..t {
        for j in (i + 1)..t {
            assert_eq!(gpu[i * t + j], 0.0, "masked weight [{i},{j}] not zero");
        }
    }
    worst = worst.max(d);
    eprintln!("softmax Metal vs CPU: global max|Δ| = {worst:.3e} (atol {ATOL})");
}

#[test]
fn layer_norm_metal_matches_cpu() {
    let ctx = ctx_or_skip!("layer_norm");
    let eps = cpu::LAYER_NORM_DEFAULT_EPS;
    // cols up to d_model = 512, rows up to a full 1500-position encoder window.
    let shapes = [
        (1usize, 2usize),
        (2, 8),
        (37, 41),
        (1, 512),
        (4, 512),
        (1500, 512),
    ];
    let mut worst = 0.0f32;
    for &(rows, cols) in &shapes {
        let input = rand_vec(0x1A11 ^ ((rows * 89 + cols) as u64), rows * cols);
        let gamma = rand_vec(0x6A11 ^ (cols as u64), cols);
        let beta = rand_vec(0xBE7A ^ (cols as u64), cols);
        let mut gpu = vec![f32::NAN; rows * cols];
        ctx.layer_norm_f32(&input, &mut gpu, rows, cols, &gamma, &beta, eps)
            .expect("metal layer_norm");
        let mut cpu_out = vec![0.0f32; rows * cols];
        cpu::layer_norm_f32(&input, &mut cpu_out, rows, cols, &gamma, &beta, eps)
            .expect("cpu layer_norm");
        let d = max_abs_diff(&gpu, &cpu_out);
        eprintln!("layer_norm  rows={rows:<5} cols={cols:<4} max|Δ|={d:.3e}");
        assert!(
            d <= ATOL,
            "layer_norm rows={rows} cols={cols}: {d} > {ATOL}"
        );
        worst = worst.max(d);
    }
    eprintln!("layer_norm Metal vs CPU: global max|Δ| = {worst:.3e} (atol {ATOL})");
}

#[test]
fn gelu_metal_matches_cpu() {
    let ctx = ctx_or_skip!("gelu");
    // Element-wise; a few sizes plus a wide-range input to stress erf.
    let lens = [1usize, 7, 63, 1000, 4 * 2048];
    let mut worst = 0.0f32;
    for &n in &lens {
        // Scale into roughly [-6, 6) so both tails of GELU/erf are exercised.
        let x: Vec<f32> = rand_vec(0x9E10 ^ (n as u64), n)
            .into_iter()
            .map(|v| v * 6.0)
            .collect();
        let mut gpu = vec![f32::NAN; n];
        ctx.gelu_f32(&x, &mut gpu).expect("metal gelu");
        let mut cpu_out = vec![0.0f32; n];
        cpu::gelu_f32(&x, &mut cpu_out).expect("cpu gelu");
        let d = max_abs_diff(&gpu, &cpu_out);
        eprintln!("gelu  n={n:<6} max|Δ|={d:.3e}");
        assert!(d <= ATOL, "gelu n={n}: {d} > {ATOL}");
        worst = worst.max(d);
    }
    eprintln!("gelu Metal vs CPU: global max|Δ| = {worst:.3e} (atol {ATOL})");
}

#[test]
fn conv1d_metal_matches_cpu() {
    let ctx = ctx_or_skip!("conv1d");
    // (in_ch, in_len, out_ch, kernel, stride, padding, bias). The last two are
    // the real Whisper encoder stem convs (80→512 k3 s1 p1, then 512→512 k3 s2 p1)
    // over a full 3000-frame mel window.
    let cases = [
        (1usize, 5usize, 1usize, 3usize, 1usize, 0usize, false),
        (1, 8, 1, 3, 1, 1, true),
        (2, 4, 1, 2, 2, 0, false),
        (3, 16, 4, 3, 1, 1, true),
        (8, 50, 16, 5, 2, 2, true),
        (80, 3000, 512, 3, 1, 1, true),
        (512, 1500, 512, 3, 2, 1, true),
    ];
    let mut worst = 0.0f32;
    for &(in_ch, in_len, out_ch, kernel, stride, padding, with_bias) in &cases {
        let out_len = (in_len + 2 * padding - kernel) / stride + 1;
        let input = rand_vec(0xC0 ^ ((in_ch * 131 + in_len) as u64), in_ch * in_len);
        let weight = rand_vec(
            0xF0 ^ ((out_ch * 17 + in_ch * kernel) as u64),
            out_ch * in_ch * kernel,
        );
        let bias_vec = with_bias.then(|| rand_vec(0xB1A5 ^ (out_ch as u64), out_ch));
        let bias = bias_vec.as_deref();

        let mut gpu = vec![f32::NAN; out_ch * out_len];
        ctx.conv1d_f32(
            &input, in_ch, in_len, &weight, out_ch, kernel, bias, stride, padding, &mut gpu,
        )
        .expect("metal conv1d");
        let mut cpu_out = vec![0.0f32; out_ch * out_len];
        cpu::conv1d_f32(
            &input,
            in_ch,
            in_len,
            &weight,
            out_ch,
            kernel,
            bias,
            stride,
            padding,
            &mut cpu_out,
        )
        .expect("cpu conv1d");
        let d = max_abs_diff(&gpu, &cpu_out);
        eprintln!(
            "conv1d  in_ch={in_ch:<4} in_len={in_len:<5} out_ch={out_ch:<4} k={kernel} s={stride} p={padding} bias={with_bias:<5} max|Δ|={d:.3e}"
        );
        assert!(
            d <= ATOL,
            "conv1d in_ch={in_ch} in_len={in_len} out_ch={out_ch} k={kernel} s={stride} p={padding}: {d} > {ATOL}"
        );
        worst = worst.max(d);
    }
    eprintln!("conv1d Metal vs CPU: global max|Δ| = {worst:.3e} (atol {ATOL})");
}

/// Shape mismatches are explicit `InvalidArgument`, not a GPU fault (mirrors the
/// GEMM shape-validation test).
#[test]
fn kernels_reject_bad_shapes_explicitly() {
    let ctx = ctx_or_skip!("shape-validation");
    // gemv: a length must be m*k.
    assert!(
        ctx.gemv_f32(2, 3, &[0.0; 5], &[0.0; 3], None, &mut [0.0; 2])
            .is_err()
    );
    // softmax: length must be rows*cols.
    assert!(ctx.softmax_f32(&[0.0; 6], &mut [0.0; 6], 2, 4).is_err());
    // layer_norm: gamma length must be cols.
    assert!(
        ctx.layer_norm_f32(&[0.0; 8], &mut [0.0; 8], 2, 4, &[1.0; 3], &[0.0; 4], 1e-5)
            .is_err()
    );
    // gelu: out length must equal x length.
    assert!(ctx.gelu_f32(&[0.0; 4], &mut [0.0; 3]).is_err());
    // conv1d: zero stride is rejected before any GPU work.
    assert!(
        ctx.conv1d_f32(&[1.0, 2.0], 1, 2, &[1.0], 1, 1, None, 0, 0, &mut [0.0; 1])
            .is_err()
    );
    // conv1d: kernel larger than the padded input.
    assert!(
        ctx.conv1d_f32(
            &[1.0, 2.0],
            1,
            2,
            &[1.0; 5],
            1,
            5,
            None,
            1,
            0,
            &mut [0.0; 1]
        )
        .is_err()
    );
}
