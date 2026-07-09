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
use vokra_core::{KvCache, PrenormLayer};

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

// ---- Phase-5: fused device-resident MLP -------------------------------------

/// The fused `MetalContext::mlp_f32` (fc1 GEMM → GELU → fc2 GEMM in ONE command
/// buffer, the `[t, ffn]` intermediates kept device-resident, one readback of
/// `out`) must be **bit-identical** to running the same three GPU kernels
/// per-op (`gemm_f32` → `gelu_f32` → `gemm_f32`, three readbacks) — same
/// kernels, same order, same launch geometry — and must match the CPU
/// three-kernel reference within the FP32 bound. `d` is the fc1-in / fc2-out
/// width, `ffn` the fc1-out / fc2-in width.
#[test]
fn mlp_fused_matches_sequential_and_cpu() {
    let ctx = ctx_or_skip!("mlp");
    // (t, d, ffn): tiny, ragged non-16 multiples, and a Whisper-tiny-ish block.
    let shapes = [
        (1usize, 2usize, 4usize),
        (3, 8, 16),
        (7, 5, 9),
        (16, 64, 128),
        (30, 40, 50),
    ];
    let mut worst_seq = 0.0f32;
    let mut worst_cpu = 0.0f32;
    for &(t, d, ffn) in &shapes {
        for with_bias in [false, true] {
            let x = rand_vec(1, t * d);
            let fc1_w = rand_vec(2, d * ffn);
            let fc2_w = rand_vec(3, ffn * d);
            let fc1_b = rand_vec(4, ffn);
            let fc2_b = rand_vec(5, d);
            let (b1, b2) = if with_bias {
                (Some(fc1_b.as_slice()), Some(fc2_b.as_slice()))
            } else {
                (None, None)
            };

            // Fused GPU (device-resident intermediates, one readback).
            let mut fused = vec![0.0f32; t * d];
            ctx.mlp_f32(t, d, ffn, &x, &fc1_w, b1, &fc2_w, b2, &mut fused)
                .expect("fused mlp");

            // Per-op GPU (same kernels, three readbacks) — the bit-identical ref.
            let mut h = vec![0.0f32; t * ffn];
            ctx.gemm_f32(t, ffn, d, &x, &fc1_w, b1, &mut h).unwrap();
            let mut a = vec![0.0f32; t * ffn];
            ctx.gelu_f32(&h, &mut a).unwrap();
            let mut seq = vec![0.0f32; t * d];
            ctx.gemm_f32(t, d, ffn, &a, &fc2_w, b2, &mut seq).unwrap();

            // CPU three-kernel reference.
            let mut hc = vec![0.0f32; t * ffn];
            cpu::gemm_f32(t, ffn, d, &x, &fc1_w, b1, &mut hc).unwrap();
            let mut ac = vec![0.0f32; t * ffn];
            cpu::gelu_f32(&hc, &mut ac).unwrap();
            let mut cpu_out = vec![0.0f32; t * d];
            cpu::gemm_f32(t, d, ffn, &ac, &fc2_w, b2, &mut cpu_out).unwrap();

            let d_seq = max_abs_diff(&fused, &seq);
            let d_cpu = max_abs_diff(&fused, &cpu_out);
            assert_eq!(
                d_seq, 0.0,
                "fused vs per-op GPU must be bit-identical (t={t} d={d} ffn={ffn} bias={with_bias}); max|Δ|={d_seq:.3e}"
            );
            assert!(
                d_cpu <= ATOL,
                "fused GPU vs CPU max|Δ| {d_cpu:.3e} exceeds atol {ATOL} (t={t} d={d} ffn={ffn} bias={with_bias})"
            );
            worst_seq = worst_seq.max(d_seq);
            worst_cpu = worst_cpu.max(d_cpu);
        }
    }
    eprintln!(
        "Metal fused MLP parity: vs per-op GPU max|Δ| = {worst_seq:.3e} (bit-identical), vs CPU max|Δ| = {worst_cpu:.3e} (atol = {ATOL})"
    );
}

/// Reports the readback / sync reduction the fusion buys at a realistic Whisper
/// MLP-block shape (`t = n_audio_ctx = 1500`, `d = 512`, `ffn = 2048` — base),
/// and re-checks bit-identical parity at that scale. Fused issues ONE
/// `waitUntilCompleted` + ONE readback per call; the per-op path issues THREE of
/// each and additionally round-trips the two `[t, ffn]` (= 1500×2048 = 12 MB)
/// intermediates through the shared-buffer boundary. Wall time is printed (not a
/// hard gate — it is environment-sensitive), so run with `-- --nocapture`.
#[test]
fn mlp_fused_reduces_readback_at_whisper_scale() {
    let ctx = ctx_or_skip!("mlp scale");
    let (t, d, ffn) = (1500usize, 512usize, 2048usize);
    let x = rand_vec(11, t * d);
    let fc1_w = rand_vec(12, d * ffn);
    let fc2_w = rand_vec(13, ffn * d);
    let fc1_b = rand_vec(14, ffn);
    let fc2_b = rand_vec(15, d);
    let iters: u32 = 20;

    // Correctness at scale: fused == per-op GPU, bit-for-bit.
    let mut fused = vec![0.0f32; t * d];
    ctx.mlp_f32(
        t,
        d,
        ffn,
        &x,
        &fc1_w,
        Some(&fc1_b),
        &fc2_w,
        Some(&fc2_b),
        &mut fused,
    )
    .expect("fused mlp (scale)");
    let mut h = vec![0.0f32; t * ffn];
    ctx.gemm_f32(t, ffn, d, &x, &fc1_w, Some(&fc1_b), &mut h)
        .unwrap();
    let mut a = vec![0.0f32; t * ffn];
    ctx.gelu_f32(&h, &mut a).unwrap();
    let mut seq = vec![0.0f32; t * d];
    ctx.gemm_f32(t, d, ffn, &a, &fc2_w, Some(&fc2_b), &mut seq)
        .unwrap();
    assert_eq!(
        max_abs_diff(&fused, &seq),
        0.0,
        "fused vs per-op GPU must be bit-identical at Whisper scale"
    );

    // Fused: 1 readback + 1 sync per call.
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let mut o = vec![0.0f32; t * d];
        ctx.mlp_f32(
            t,
            d,
            ffn,
            &x,
            &fc1_w,
            Some(&fc1_b),
            &fc2_w,
            Some(&fc2_b),
            &mut o,
        )
        .unwrap();
    }
    let fused_dt = t0.elapsed();

    // Per-op: 3 readbacks + 3 syncs, plus the two [t, ffn] intermediate crossings.
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        let mut h = vec![0.0f32; t * ffn];
        ctx.gemm_f32(t, ffn, d, &x, &fc1_w, Some(&fc1_b), &mut h)
            .unwrap();
        let mut a = vec![0.0f32; t * ffn];
        ctx.gelu_f32(&h, &mut a).unwrap();
        let mut o = vec![0.0f32; t * d];
        ctx.gemm_f32(t, d, ffn, &a, &fc2_w, Some(&fc2_b), &mut o)
            .unwrap();
    }
    let perop_dt = t1.elapsed();

    eprintln!(
        "Metal fused MLP (t={t} d={d} ffn={ffn}, {iters} iters): \
         fused {fused_dt:?} ({:?}/it, 1 readback + 1 sync) vs \
         per-op {perop_dt:?} ({:?}/it, 3 readbacks + 3 syncs)",
        fused_dt / iters,
        perop_dt / iters,
    );
}

// ---- Phase-5: fused device-resident non-causal attention --------------------

/// Per-op non-causal multi-head attention, replicating `whisper::nn::
/// attention_from_kv_into`'s head loop **exactly** (q-proj GEMM → scale the whole
/// `q` → per head {host gather qh/vh, host gather-transpose kh_t, scores GEMM,
/// softmax, context GEMM, host scatter} → out-proj GEMM), with the GEMM / softmax
/// supplied as closures. Passing the GPU (`ctx.*`) closures yields the
/// bit-identical per-op reference the fused `attn_f32` must equal; passing the
/// CPU (`cpu::*`) closures yields the FP32 oracle. The single query-scale
/// multiply is the only host arithmetic — a lone IEEE-754 f32 multiply, so it is
/// identical on the host and (for the normal-valued inputs here) on the GPU,
/// which is why the fused-vs-per-op comparison is bit-exact.
/// GEMM closure for [`attn_reference`] (the `gemm_f32` contract: `m, n, k, a, b,
/// bias, out`), factored out to keep the reference signature within
/// `clippy::type_complexity`.
type GemmFn<'a> = dyn Fn(usize, usize, usize, &[f32], &[f32], Option<&[f32]>, &mut [f32]) + 'a;
/// Softmax closure for [`attn_reference`] (`input, out, rows, cols`).
type SoftmaxFn<'a> = dyn Fn(&[f32], &mut [f32], usize, usize) + 'a;

#[allow(clippy::too_many_arguments)] // faithful replica of the nn.rs attention operands
fn attn_reference(
    gemm: &GemmFn<'_>,
    softmax: &SoftmaxFn<'_>,
    t_q: usize,
    t_kv: usize,
    d: usize,
    n_head: usize,
    xq: &[f32],
    q_w: &[f32],
    q_bias: Option<&[f32]>,
    k: &[f32],
    v: &[f32],
    out_w: &[f32],
    out_bias: Option<&[f32]>,
    scale: f32,
) -> Vec<f32> {
    let hd = d / n_head;
    let mut q = vec![0.0f32; t_q * d];
    gemm(t_q, d, d, xq, q_w, q_bias, &mut q);
    for val in &mut q {
        *val *= scale;
    }
    let mut context = vec![0.0f32; t_q * d];
    let mut qh = vec![0.0f32; t_q * hd];
    let mut vh = vec![0.0f32; t_kv * hd];
    let mut kh_t = vec![0.0f32; hd * t_kv];
    let mut scores = vec![0.0f32; t_q * t_kv];
    let mut probs = vec![0.0f32; t_q * t_kv];
    let mut ctx_h = vec![0.0f32; t_q * hd];
    for h in 0..n_head {
        let c0 = h * hd;
        for i in 0..t_q {
            qh[i * hd..i * hd + hd].copy_from_slice(&q[i * d + c0..i * d + c0 + hd]);
        }
        for j in 0..t_kv {
            vh[j * hd..j * hd + hd].copy_from_slice(&v[j * d + c0..j * d + c0 + hd]);
            for c in 0..hd {
                kh_t[c * t_kv + j] = k[j * d + c0 + c];
            }
        }
        gemm(t_q, t_kv, hd, &qh, &kh_t, None, &mut scores);
        softmax(&scores, &mut probs, t_q, t_kv);
        gemm(t_q, hd, t_kv, &probs, &vh, None, &mut ctx_h);
        for i in 0..t_q {
            context[i * d + c0..i * d + c0 + hd].copy_from_slice(&ctx_h[i * hd..i * hd + hd]);
        }
    }
    let mut out = vec![0.0f32; t_q * d];
    gemm(t_q, d, d, &context, out_w, out_bias, &mut out);
    out
}

/// The fused `MetalContext::attn_f32` (q-proj → per-head {gather, QKᵀ, softmax,
/// A·V, scatter} → out-proj, every intermediate device-resident, one readback)
/// must be **bit-identical** to the per-op path built from the same GPU kernels
/// (`attn_reference` with the `ctx.*` closures — same kernels, same order, same
/// launch geometry, the scale folded into the qh gather instead of a whole-`q`
/// pass), and must match the CPU reference within the FP32 bound. Shapes cover
/// single-head, ragged non-16 sizes and a Whisper-tiny-ish `(16,16,64,8)`.
#[test]
fn attn_fused_matches_sequential_and_cpu() {
    let ctx = ctx_or_skip!("attn");
    // (t_q, t_kv, d, n_head); every d is divisible by n_head.
    let shapes = [
        (1usize, 4usize, 2usize, 1usize),
        (3, 8, 16, 2),
        (7, 5, 24, 3),
        (16, 16, 64, 8),
        (30, 20, 40, 5),
    ];
    let mut worst_seq = 0.0f32;
    let mut worst_cpu = 0.0f32;
    for &(t_q, t_kv, d, n_head) in &shapes {
        let hd = d / n_head;
        let scale = (hd as f32).powf(-0.5);
        for with_bias in [false, true] {
            let xq = rand_vec(0x1A ^ ((t_q * 7 + d) as u64), t_q * d);
            let q_w = rand_vec(0x2B ^ (d as u64), d * d);
            let k = rand_vec(0x3C ^ (t_kv as u64), t_kv * d);
            let v = rand_vec(0x4D ^ ((t_kv * 3 + d) as u64), t_kv * d);
            let out_w = rand_vec(0x5E ^ (d as u64 + 1), d * d);
            let q_b = rand_vec(0x6F, d);
            let o_b = rand_vec(0x70, d);
            let (qb, ob) = if with_bias {
                (Some(q_b.as_slice()), Some(o_b.as_slice()))
            } else {
                (None, None)
            };

            // Fused GPU (device-resident intermediates, one readback).
            let mut fused = vec![0.0f32; t_q * d];
            ctx.attn_f32(
                t_q, t_kv, d, n_head, &xq, &q_w, qb, &k, &v, &out_w, ob, scale, &mut fused,
            )
            .expect("fused attn");

            // Per-op GPU (same kernels, host gather/scale/scatter) — bit-identical ref.
            let gpu_gemm = |m: usize,
                            n: usize,
                            kk: usize,
                            a: &[f32],
                            b: &[f32],
                            bias: Option<&[f32]>,
                            o: &mut [f32]| {
                ctx.gemm_f32(m, n, kk, a, b, bias, o).expect("gpu gemm");
            };
            let gpu_softmax = |i: &[f32], o: &mut [f32], r: usize, c: usize| {
                ctx.softmax_f32(i, o, r, c).expect("gpu softmax");
            };
            let seq = attn_reference(
                &gpu_gemm,
                &gpu_softmax,
                t_q,
                t_kv,
                d,
                n_head,
                &xq,
                &q_w,
                qb,
                &k,
                &v,
                &out_w,
                ob,
                scale,
            );

            // CPU reference.
            let cpu_gemm = |m: usize,
                            n: usize,
                            kk: usize,
                            a: &[f32],
                            b: &[f32],
                            bias: Option<&[f32]>,
                            o: &mut [f32]| {
                cpu::gemm_f32(m, n, kk, a, b, bias, o).expect("cpu gemm");
            };
            let cpu_softmax = |i: &[f32], o: &mut [f32], r: usize, c: usize| {
                cpu::softmax_f32(i, o, r, c).expect("cpu softmax");
            };
            let cpu_out = attn_reference(
                &cpu_gemm,
                &cpu_softmax,
                t_q,
                t_kv,
                d,
                n_head,
                &xq,
                &q_w,
                qb,
                &k,
                &v,
                &out_w,
                ob,
                scale,
            );

            let d_seq = max_abs_diff(&fused, &seq);
            let d_cpu = max_abs_diff(&fused, &cpu_out);
            assert_eq!(
                d_seq, 0.0,
                "fused vs per-op GPU must be bit-identical (t_q={t_q} t_kv={t_kv} d={d} n_head={n_head} bias={with_bias}); max|Δ|={d_seq:.3e}"
            );
            assert!(
                d_cpu <= ATOL,
                "fused GPU vs CPU max|Δ| {d_cpu:.3e} exceeds atol {ATOL} (t_q={t_q} t_kv={t_kv} d={d} n_head={n_head} bias={with_bias})"
            );
            worst_seq = worst_seq.max(d_seq);
            worst_cpu = worst_cpu.max(d_cpu);
        }
    }
    eprintln!(
        "Metal fused attn parity: vs per-op GPU max|Δ| = {worst_seq:.3e} (bit-identical), vs CPU max|Δ| = {worst_cpu:.3e} (atol = {ATOL})"
    );
}

/// Reports the submission / readback reduction the attention fusion buys at a
/// realistic Whisper-base encoder self-attention shape (`t_q = t_kv =
/// n_audio_ctx = 1500`, `d = 512`, `n_head = 8`), and re-checks bit-identical
/// parity at that scale. Fused issues ONE `waitUntilCompleted` + ONE readback per
/// call; the per-op path issues `2 + 3·n_head = 26` command buffers each with its
/// own sync + readback, plus the host qh/kh_t/vh/scores/probs/ctx_h round-trips.
/// Wall time is printed (not a hard gate — it is environment-sensitive), so run
/// with `-- --nocapture`.
#[test]
fn attn_fused_reduces_readback_at_whisper_scale() {
    let ctx = ctx_or_skip!("attn scale");
    let (t_q, t_kv, d, n_head) = (1500usize, 1500usize, 512usize, 8usize);
    let hd = d / n_head;
    let scale = (hd as f32).powf(-0.5);
    let xq = rand_vec(21, t_q * d);
    let q_w = rand_vec(22, d * d);
    let k = rand_vec(23, t_kv * d);
    let v = rand_vec(24, t_kv * d);
    let out_w = rand_vec(25, d * d);
    let q_b = rand_vec(26, d);
    let o_b = rand_vec(27, d);
    let iters: u32 = 10;

    // Correctness at scale: fused == per-op GPU, bit-for-bit.
    let mut fused = vec![0.0f32; t_q * d];
    ctx.attn_f32(
        t_q,
        t_kv,
        d,
        n_head,
        &xq,
        &q_w,
        Some(&q_b),
        &k,
        &v,
        &out_w,
        Some(&o_b),
        scale,
        &mut fused,
    )
    .expect("fused attn (scale)");
    let gpu_gemm = |m: usize,
                    n: usize,
                    kk: usize,
                    a: &[f32],
                    b: &[f32],
                    bias: Option<&[f32]>,
                    o: &mut [f32]| {
        ctx.gemm_f32(m, n, kk, a, b, bias, o).expect("gpu gemm");
    };
    let gpu_softmax = |i: &[f32], o: &mut [f32], r: usize, c: usize| {
        ctx.softmax_f32(i, o, r, c).expect("gpu softmax");
    };
    let seq = attn_reference(
        &gpu_gemm,
        &gpu_softmax,
        t_q,
        t_kv,
        d,
        n_head,
        &xq,
        &q_w,
        Some(&q_b),
        &k,
        &v,
        &out_w,
        Some(&o_b),
        scale,
    );
    assert_eq!(
        max_abs_diff(&fused, &seq),
        0.0,
        "fused vs per-op GPU must be bit-identical at Whisper scale"
    );

    // Fused: 1 readback + 1 sync per call.
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let mut o = vec![0.0f32; t_q * d];
        ctx.attn_f32(
            t_q,
            t_kv,
            d,
            n_head,
            &xq,
            &q_w,
            Some(&q_b),
            &k,
            &v,
            &out_w,
            Some(&o_b),
            scale,
            &mut o,
        )
        .unwrap();
    }
    let fused_dt = t0.elapsed();

    // Per-op: 26 command buffers (each its own sync + readback) + host round-trips.
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = attn_reference(
            &gpu_gemm,
            &gpu_softmax,
            t_q,
            t_kv,
            d,
            n_head,
            &xq,
            &q_w,
            Some(&q_b),
            &k,
            &v,
            &out_w,
            Some(&o_b),
            scale,
        );
    }
    let perop_dt = t1.elapsed();

    eprintln!(
        "Metal fused attn (t_q={t_q} t_kv={t_kv} d={d} n_head={n_head}, {iters} iters): \
         fused {fused_dt:?} ({:?}/it, 1 readback + 1 sync) vs \
         per-op {perop_dt:?} ({:?}/it, {} readbacks + {} syncs + host round-trips)",
        fused_dt / iters,
        perop_dt / iters,
        2 + 3 * n_head,
        2 + 3 * n_head,
    );
}

/// `attn_f32` rejects mis-sized / mis-configured operands with an explicit
/// `InvalidArgument` (never a GPU fault): here `d` (6) is not divisible by
/// `n_head` (4).
#[test]
fn attn_f32_rejects_missized() {
    let ctx = ctx_or_skip!("attn reject");
    let err = ctx.attn_f32(
        1,
        1,
        6,
        4, // d=6 not divisible by n_head=4
        &[0.0; 6],
        &[0.0; 36],
        None,
        &[0.0; 6],
        &[0.0; 6],
        &[0.0; 36],
        None,
        1.0,
        &mut [0.0; 6],
    );
    assert!(err.is_err(), "d % n_head != 0 must be rejected");
}

// ---- Phase-5 follow-on: device-resident whole-encoder stack -----------------

/// LayerNorm epsilon (Whisper / the CPU kernel default), shared by the prenorm
/// tests below.
const PRENORM_EPS: f32 = 1e-5;

/// Owned random weights for one pre-norm block (kept alive so the borrowed
/// [`PrenormLayer`] views below outlive the encode). Biases match Whisper: `q` /
/// `v` / `out` / `fc1` / `fc2` carry one, `k` does not.
struct LayerData {
    attn_ln_g: Vec<f32>,
    attn_ln_b: Vec<f32>,
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    mlp_ln_g: Vec<f32>,
    mlp_ln_b: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
}

fn make_layer(seed: u64, d: usize, ff: usize) -> LayerData {
    LayerData {
        attn_ln_g: rand_vec(seed ^ 0x01, d),
        attn_ln_b: rand_vec(seed ^ 0x02, d),
        q_w: rand_vec(seed ^ 0x03, d * d),
        q_b: rand_vec(seed ^ 0x04, d),
        k_w: rand_vec(seed ^ 0x05, d * d),
        v_w: rand_vec(seed ^ 0x06, d * d),
        v_b: rand_vec(seed ^ 0x07, d),
        out_w: rand_vec(seed ^ 0x08, d * d),
        out_b: rand_vec(seed ^ 0x09, d),
        mlp_ln_g: rand_vec(seed ^ 0x0A, d),
        mlp_ln_b: rand_vec(seed ^ 0x0B, d),
        fc1_w: rand_vec(seed ^ 0x0C, d * ff),
        fc1_b: rand_vec(seed ^ 0x0D, ff),
        fc2_w: rand_vec(seed ^ 0x0E, ff * d),
        fc2_b: rand_vec(seed ^ 0x0F, d),
    }
}

fn layer_view(l: &LayerData) -> PrenormLayer<'_> {
    PrenormLayer {
        attn_ln_gamma: &l.attn_ln_g,
        attn_ln_beta: &l.attn_ln_b,
        q_w: &l.q_w,
        q_bias: Some(&l.q_b),
        k_w: &l.k_w,
        k_bias: None, // Whisper's k_proj has no bias
        v_w: &l.v_w,
        v_bias: Some(&l.v_b),
        out_w: &l.out_w,
        out_bias: Some(&l.out_b),
        mlp_ln_gamma: &l.mlp_ln_g,
        mlp_ln_beta: &l.mlp_ln_b,
        fc1_w: &l.fc1_w,
        fc1_bias: Some(&l.fc1_b),
        fc2_w: &l.fc2_w,
        fc2_bias: Some(&l.fc2_b),
    }
}

/// The **current** per-op GPU encoder path (what `whisper::encoder` runs today on
/// a GPU backend): per block `layer_norm → k/v GEMM → fused attn_f32 → host
/// residual add → layer_norm → fused mlp_f32 → host residual add`, then a final
/// LayerNorm. Because the fused stack encodes the *same* kernels in the same order
/// (its attn passes == `attn_f32`, its mlp passes == `mlp_f32`, ln/GEMM/add
/// identical), this is the **bit-identical** reference — and it issues exactly
/// `6·N + 1` submissions, the count the fused stack collapses to one.
#[allow(clippy::too_many_arguments)]
fn prenorm_reference_current(
    ctx: &MetalContext,
    t: usize,
    d: usize,
    ff: usize,
    n_head: usize,
    hidden: &[f32],
    layers: &[PrenormLayer<'_>],
    fg: &[f32],
    fb: &[f32],
) -> Vec<f32> {
    let hd = d / n_head;
    let scale = (hd as f32).powf(-0.5);
    let mut h = hidden.to_vec();
    let mut ln = vec![0.0f32; t * d];
    for l in layers {
        ctx.layer_norm_f32(
            &h,
            &mut ln,
            t,
            d,
            l.attn_ln_gamma,
            l.attn_ln_beta,
            PRENORM_EPS,
        )
        .unwrap();
        let mut k = vec![0.0f32; t * d];
        ctx.gemm_f32(t, d, d, &ln, l.k_w, l.k_bias, &mut k).unwrap();
        let mut v = vec![0.0f32; t * d];
        ctx.gemm_f32(t, d, d, &ln, l.v_w, l.v_bias, &mut v).unwrap();
        let mut bo = vec![0.0f32; t * d];
        ctx.attn_f32(
            t, t, d, n_head, &ln, l.q_w, l.q_bias, &k, &v, l.out_w, l.out_bias, scale, &mut bo,
        )
        .unwrap();
        for (dst, &src) in h.iter_mut().zip(&bo) {
            *dst += src;
        }
        ctx.layer_norm_f32(
            &h,
            &mut ln,
            t,
            d,
            l.mlp_ln_gamma,
            l.mlp_ln_beta,
            PRENORM_EPS,
        )
        .unwrap();
        let mut bo2 = vec![0.0f32; t * d];
        ctx.mlp_f32(
            t, d, ff, &ln, l.fc1_w, l.fc1_bias, l.fc2_w, l.fc2_bias, &mut bo2,
        )
        .unwrap();
        for (dst, &src) in h.iter_mut().zip(&bo2) {
            *dst += src;
        }
    }
    let mut normed = vec![0.0f32; t * d];
    ctx.layer_norm_f32(&h, &mut normed, t, d, fg, fb, PRENORM_EPS)
        .unwrap();
    normed
}

/// The per-op **CPU** encoder path (the onnxruntime-agreeing reference the whole
/// Whisper CPU forward runs): same block structure via the `vokra-backend-cpu`
/// kernels + the CPU head-loop attention (`attn_reference` with CPU closures). The
/// fused GPU stack must match this within the FP32 bound (NFR-QL-01).
#[allow(clippy::too_many_arguments)]
fn prenorm_reference_cpu(
    t: usize,
    d: usize,
    ff: usize,
    n_head: usize,
    hidden: &[f32],
    layers: &[PrenormLayer<'_>],
    fg: &[f32],
    fb: &[f32],
) -> Vec<f32> {
    let hd = d / n_head;
    let scale = (hd as f32).powf(-0.5);
    let cpu_gemm = |m: usize,
                    n: usize,
                    kk: usize,
                    a: &[f32],
                    b: &[f32],
                    bias: Option<&[f32]>,
                    o: &mut [f32]| {
        cpu::gemm_f32(m, n, kk, a, b, bias, o).expect("cpu gemm");
    };
    let cpu_softmax = |i: &[f32], o: &mut [f32], r: usize, c: usize| {
        cpu::softmax_f32(i, o, r, c).expect("cpu softmax");
    };
    let mut h = hidden.to_vec();
    let mut ln = vec![0.0f32; t * d];
    for l in layers {
        cpu::layer_norm_f32(
            &h,
            &mut ln,
            t,
            d,
            l.attn_ln_gamma,
            l.attn_ln_beta,
            PRENORM_EPS,
        )
        .unwrap();
        let mut k = vec![0.0f32; t * d];
        cpu::gemm_f32(t, d, d, &ln, l.k_w, l.k_bias, &mut k).unwrap();
        let mut v = vec![0.0f32; t * d];
        cpu::gemm_f32(t, d, d, &ln, l.v_w, l.v_bias, &mut v).unwrap();
        let bo = attn_reference(
            &cpu_gemm,
            &cpu_softmax,
            t,
            t,
            d,
            n_head,
            &ln,
            l.q_w,
            l.q_bias,
            &k,
            &v,
            l.out_w,
            l.out_bias,
            scale,
        );
        for (dst, &src) in h.iter_mut().zip(&bo) {
            *dst += src;
        }
        cpu::layer_norm_f32(
            &h,
            &mut ln,
            t,
            d,
            l.mlp_ln_gamma,
            l.mlp_ln_beta,
            PRENORM_EPS,
        )
        .unwrap();
        let mut mh = vec![0.0f32; t * ff];
        cpu::gemm_f32(t, ff, d, &ln, l.fc1_w, l.fc1_bias, &mut mh).unwrap();
        let mut ma = vec![0.0f32; t * ff];
        cpu::gelu_f32(&mh, &mut ma).unwrap();
        let mut mo = vec![0.0f32; t * d];
        cpu::gemm_f32(t, d, ff, &ma, l.fc2_w, l.fc2_bias, &mut mo).unwrap();
        for (dst, &src) in h.iter_mut().zip(&mo) {
            *dst += src;
        }
    }
    let mut normed = vec![0.0f32; t * d];
    cpu::layer_norm_f32(&h, &mut normed, t, d, fg, fb, PRENORM_EPS).unwrap();
    normed
}

/// The device-resident whole-encoder `encode_prenorm_stack` must be
/// **bit-identical** to the current per-op GPU path (same kernels, order, launch
/// geometry — one submission vs `6·N + 1`) and match the CPU within the FP32
/// bound. Shapes cover single-block, ragged non-16 dims and a Whisper-tiny-ish
/// block, over 1-2 layers.
#[test]
fn prenorm_stack_matches_sequential_and_cpu() {
    let ctx = ctx_or_skip!("prenorm stack");
    // (t, d, ff, n_head, n_layers); every d divisible by n_head.
    let shapes = [
        (1usize, 2usize, 4usize, 1usize, 1usize),
        (3, 8, 16, 2, 2),
        (7, 24, 40, 3, 2),
        (16, 64, 128, 8, 2),
        (30, 40, 80, 5, 3),
    ];
    let mut worst_seq = 0.0f32;
    let mut worst_cpu = 0.0f32;
    for &(t, d, ff, n_head, n_layers) in &shapes {
        let data: Vec<LayerData> = (0..n_layers)
            .map(|i| make_layer((i * 97 + d) as u64, d, ff))
            .collect();
        let layers: Vec<PrenormLayer<'_>> = data.iter().map(layer_view).collect();
        let hidden = rand_vec(0xABCD ^ ((t * 13 + d) as u64), t * d);
        let fg = rand_vec(0x11, d);
        let fb = rand_vec(0x22, d);

        let mut fused = vec![0.0f32; t * d];
        ctx.encode_prenorm_stack(
            t,
            d,
            ff,
            n_head,
            PRENORM_EPS,
            &hidden,
            &layers,
            &fg,
            &fb,
            &mut fused,
        )
        .expect("fused prenorm stack");

        let seq = prenorm_reference_current(&ctx, t, d, ff, n_head, &hidden, &layers, &fg, &fb);
        let cpuref = prenorm_reference_cpu(t, d, ff, n_head, &hidden, &layers, &fg, &fb);

        let d_seq = max_abs_diff(&fused, &seq);
        let d_cpu = max_abs_diff(&fused, &cpuref);
        assert_eq!(
            d_seq, 0.0,
            "fused stack vs per-op GPU must be bit-identical (t={t} d={d} ff={ff} n_head={n_head} L={n_layers}); max|Δ|={d_seq:.3e}"
        );
        assert!(
            d_cpu <= ATOL,
            "fused stack vs CPU max|Δ| {d_cpu:.3e} exceeds atol {ATOL} (t={t} d={d} ff={ff} n_head={n_head} L={n_layers})"
        );
        worst_seq = worst_seq.max(d_seq);
        worst_cpu = worst_cpu.max(d_cpu);
    }
    eprintln!(
        "Metal prenorm-stack parity: vs per-op GPU max|Δ| = {worst_seq:.3e} (bit-identical), vs CPU max|Δ| = {worst_cpu:.3e} (atol = {ATOL})"
    );
}

/// The whole-encoder residency's payoff: the fused stack issues **exactly ONE**
/// submission (commit + waitUntilCompleted) for the whole encoder, versus the
/// current per-op path's `6·N + 1`. The count is measured with the context's
/// submission counter (env-independent, unlike wall time), and re-checked
/// bit-identical at this scale; wall time is printed (run with `-- --nocapture`).
#[test]
fn prenorm_stack_reduces_readback() {
    let ctx = ctx_or_skip!("prenorm stack readback");
    // A few blocks at a moderate width — the submission count is shape-independent,
    // so this stays quick while still exercising several layers / heads.
    let (t, d, ff, n_head, n_layers) = (256usize, 128usize, 512usize, 8usize, 4usize);
    let data: Vec<LayerData> = (0..n_layers)
        .map(|i| make_layer((i * 31 + 7) as u64, d, ff))
        .collect();
    let layers: Vec<PrenormLayer<'_>> = data.iter().map(layer_view).collect();
    let hidden = rand_vec(0x5151, t * d);
    let fg = rand_vec(0x61, d);
    let fb = rand_vec(0x62, d);

    // Fused: exactly ONE submission for the whole encoder.
    let s0 = ctx.submission_count();
    let t0 = std::time::Instant::now();
    let mut fused = vec![0.0f32; t * d];
    ctx.encode_prenorm_stack(
        t,
        d,
        ff,
        n_head,
        PRENORM_EPS,
        &hidden,
        &layers,
        &fg,
        &fb,
        &mut fused,
    )
    .expect("fused prenorm stack (readback)");
    let fused_dt = t0.elapsed();
    let d_fused = ctx.submission_count() - s0;

    // Current per-op path: 6·N + 1 submissions.
    let s1 = ctx.submission_count();
    let t1 = std::time::Instant::now();
    let seq = prenorm_reference_current(&ctx, t, d, ff, n_head, &hidden, &layers, &fg, &fb);
    let perop_dt = t1.elapsed();
    let d_perop = ctx.submission_count() - s1;

    assert_eq!(
        max_abs_diff(&fused, &seq),
        0.0,
        "fused stack vs per-op GPU must be bit-identical at readback scale"
    );
    assert_eq!(d_fused, 1, "the fused encoder must be ONE submission");
    assert_eq!(
        d_perop,
        (6 * n_layers + 1) as u64,
        "the per-op path must be 6·N + 1 submissions"
    );
    eprintln!(
        "Metal prenorm stack ({n_layers} layers, t={t} d={d} ff={ff} n_head={n_head}): \
         fused {fused_dt:?} ({d_fused} submission) vs \
         per-op {perop_dt:?} ({d_perop} submissions)"
    );
}

/// Each public device-in/out op (`layer_norm_dev` / `residual_add_dev` /
/// `mlp_dev` / `attn_dev`) is **bit-identical** to its host-in/out sibling
/// (`layer_norm_f32` / host add / `mlp_f32` / `attn_f32`) — same kernels, only the
/// buffer residency differs — and `download(upload(x)) == x` round-trips exactly.
#[test]
fn device_ops_match_host_in_out() {
    let ctx = ctx_or_skip!("device ops");

    // upload/download round-trip.
    let x = rand_vec(1, 37);
    let xt = ctx.upload(&x).expect("upload");
    let mut back = vec![0.0f32; 37];
    ctx.download(&xt, &mut back).expect("download");
    assert_eq!(back, x, "download(upload(x)) must equal x");

    // layer_norm_dev == layer_norm_f32.
    let (rows, cols) = (5usize, 8usize);
    let inp = rand_vec(2, rows * cols);
    let g = rand_vec(3, cols);
    let b = rand_vec(4, cols);
    let it = ctx.upload(&inp).unwrap();
    let gt = ctx.upload(&g).unwrap();
    let bt = ctx.upload(&b).unwrap();
    let mut ot = ctx.alloc_dev(rows * cols).unwrap();
    ctx.layer_norm_dev(&mut ot, &it, &gt, &bt, rows, cols, PRENORM_EPS)
        .expect("layer_norm_dev");
    let mut dev = vec![0.0f32; rows * cols];
    ctx.download(&ot, &mut dev).unwrap();
    let mut host = vec![0.0f32; rows * cols];
    ctx.layer_norm_f32(&inp, &mut host, rows, cols, &g, &b, PRENORM_EPS)
        .unwrap();
    assert_eq!(dev, host, "layer_norm_dev must equal layer_norm_f32");

    // residual_add_dev == host add.
    let a1 = rand_vec(5, 20);
    let a2 = rand_vec(6, 20);
    let mut dt = ctx.upload(&a1).unwrap();
    let st = ctx.upload(&a2).unwrap();
    ctx.residual_add_dev(&mut dt, &st)
        .expect("residual_add_dev");
    let mut dev = vec![0.0f32; 20];
    ctx.download(&dt, &mut dev).unwrap();
    let host: Vec<f32> = a1.iter().zip(&a2).map(|(x, y)| x + y).collect();
    assert_eq!(dev, host, "residual_add_dev must equal a host += add");

    // mlp_dev == mlp_f32.
    let (t, d, ff) = (4usize, 6usize, 12usize);
    let mx = rand_vec(7, t * d);
    let f1 = rand_vec(8, d * ff);
    let f2 = rand_vec(9, ff * d);
    let b1 = rand_vec(10, ff);
    let b2 = rand_vec(11, d);
    let mxt = ctx.upload(&mx).unwrap();
    let f1t = ctx.upload(&f1).unwrap();
    let f2t = ctx.upload(&f2).unwrap();
    let b1t = ctx.upload(&b1).unwrap();
    let b2t = ctx.upload(&b2).unwrap();
    let mut mot = ctx.alloc_dev(t * d).unwrap();
    ctx.mlp_dev(t, d, ff, &mxt, &f1t, Some(&b1t), &f2t, Some(&b2t), &mut mot)
        .expect("mlp_dev");
    let mut dev = vec![0.0f32; t * d];
    ctx.download(&mot, &mut dev).unwrap();
    let mut host = vec![0.0f32; t * d];
    ctx.mlp_f32(t, d, ff, &mx, &f1, Some(&b1), &f2, Some(&b2), &mut host)
        .unwrap();
    assert_eq!(dev, host, "mlp_dev must equal mlp_f32");

    // attn_dev == attn_f32.
    let (tq, tkv, dd, nh) = (7usize, 5usize, 24usize, 3usize);
    let hd = dd / nh;
    let scale = (hd as f32).powf(-0.5);
    let xq = rand_vec(12, tq * dd);
    let qw = rand_vec(13, dd * dd);
    let kk = rand_vec(14, tkv * dd);
    let vv = rand_vec(15, tkv * dd);
    let ow = rand_vec(16, dd * dd);
    let qb = rand_vec(17, dd);
    let ob = rand_vec(18, dd);
    let xqt = ctx.upload(&xq).unwrap();
    let qwt = ctx.upload(&qw).unwrap();
    let kt = ctx.upload(&kk).unwrap();
    let vt = ctx.upload(&vv).unwrap();
    let owt = ctx.upload(&ow).unwrap();
    let qbt = ctx.upload(&qb).unwrap();
    let obt = ctx.upload(&ob).unwrap();
    let mut aot = ctx.alloc_dev(tq * dd).unwrap();
    ctx.attn_dev(
        tq,
        tkv,
        dd,
        nh,
        &xqt,
        &qwt,
        Some(&qbt),
        &kt,
        &vt,
        &owt,
        Some(&obt),
        scale,
        &mut aot,
    )
    .expect("attn_dev");
    let mut dev = vec![0.0f32; tq * dd];
    ctx.download(&aot, &mut dev).unwrap();
    let mut host = vec![0.0f32; tq * dd];
    ctx.attn_f32(
        tq,
        tkv,
        dd,
        nh,
        &xq,
        &qw,
        Some(&qb),
        &kk,
        &vv,
        &ow,
        Some(&ob),
        scale,
        &mut host,
    )
    .unwrap();
    assert_eq!(dev, host, "attn_dev must equal attn_f32");

    eprintln!("Metal device-in/out ops all bit-identical to their host-in/out siblings");
}

// ---- Decoder-step Phase 2: device-resident self-attention K/V cache ---------

/// The device [`MetalContext::kv_append`] primitive must reproduce a host
/// [`KvCache`] project-and-concatenate: each step's `k = x·Wk (+b)` /
/// `v = x·Wv (+b)` is written in place at the running row `len`. The multi-step
/// result must equal
/// (a) a single **monolithic** projection of all rows at once on the GPU — proving
///     the offset write neither shifts nor corrupts rows (`max|Δ| == 0`), and
/// (b) the host `KvCache` fed the same weights through the CPU GEMM oracle
///     (within the FP32 `ATOL`).
///
/// Covers a forced multi-row prefix followed by single-token steps, bias on/off,
/// and asserts the reserve is a true capacity the append never reallocates.
#[test]
fn kv_cache_append_matches_host_project_concat() {
    let ctx = ctx_or_skip!("kv cache append");
    // (d, width, step sizes). `width` is the projected K/V hidden size (== d for
    // whisper self-attention, kept distinct here to exercise a general [d,width]
    // projection).
    let cases: [(usize, usize, &[usize]); 3] = [
        (4, 4, &[1, 1, 1, 1]),
        (16, 16, &[3, 1, 1, 2, 1]), // forced prefix (3) then single-token steps
        (24, 24, &[5, 2, 1]),
    ];
    let mut worst_cpu = 0.0f32;
    for (ci, &(d, width, steps)) in cases.iter().enumerate() {
        let total: usize = steps.iter().sum();
        let cap_rows = total + 5; // reserve strictly more than the decode needs
        for with_bias in [false, true] {
            let seed = 0x4B56_0001u64 ^ ((ci * 131 + usize::from(with_bias)) as u64);
            let x_full = rand_vec(seed, total * d);
            let k_w = rand_vec(seed ^ 0xA1, d * width);
            let v_w = rand_vec(seed ^ 0xB2, d * width);
            let k_b = rand_vec(seed ^ 0xC3, width);
            let v_b = rand_vec(seed ^ 0xD4, width);
            let (kb, vb) = if with_bias {
                (Some(k_b.as_slice()), Some(v_b.as_slice()))
            } else {
                (None, None)
            };

            // Weights + biases uploaded once (constant across steps).
            let kw_dev = ctx.upload(&k_w).unwrap();
            let vw_dev = ctx.upload(&v_w).unwrap();
            let kb_dev = kb.map(|b| ctx.upload(b).unwrap());
            let vb_dev = vb.map(|b| ctx.upload(b).unwrap());

            // (a) Device multi-step append.
            let mut cache = ctx.new_kv_cache(cap_rows, width).expect("new_kv_cache");
            assert!(cache.is_empty(), "a fresh cache is empty");
            assert_eq!(cache.capacity_rows(), cap_rows, "reserve is the hard cap");
            assert_eq!(cache.width(), width);
            let mut row = 0usize;
            for &t in steps {
                let x_dev = ctx.upload(&x_full[row * d..(row + t) * d]).unwrap();
                ctx.kv_append(
                    &mut cache,
                    t,
                    d,
                    &x_dev,
                    &kw_dev,
                    kb_dev.as_ref(),
                    &vw_dev,
                    vb_dev.as_ref(),
                )
                .expect("kv_append");
                row += t;
                assert_eq!(cache.len(), row, "len advances by each step's rows");
            }
            assert_eq!(cache.len(), total);
            let mut dev_k = vec![0.0f32; total * width];
            let mut dev_v = vec![0.0f32; total * width];
            ctx.kv_download(&cache, &mut dev_k, &mut dev_v)
                .expect("kv_download");

            // (b) Device single-shot append of ALL rows at offset 0 (monolithic).
            let mut mono = ctx.new_kv_cache(cap_rows, width).unwrap();
            let x_all = ctx.upload(&x_full).unwrap();
            ctx.kv_append(
                &mut mono,
                total,
                d,
                &x_all,
                &kw_dev,
                kb_dev.as_ref(),
                &vw_dev,
                vb_dev.as_ref(),
            )
            .unwrap();
            let mut mono_k = vec![0.0f32; total * width];
            let mut mono_v = vec![0.0f32; total * width];
            ctx.kv_download(&mono, &mut mono_k, &mut mono_v).unwrap();
            assert_eq!(
                max_abs_diff(&dev_k, &mono_k),
                0.0,
                "K offset-append must be bit-identical to a monolithic projection (case {ci} bias={with_bias})"
            );
            assert_eq!(
                max_abs_diff(&dev_v, &mono_v),
                0.0,
                "V offset-append must be bit-identical to a monolithic projection (case {ci} bias={with_bias})"
            );

            // (c) Host KvCache project+concat through the CPU GEMM oracle.
            let mut host = KvCache::with_reserve(1, width, cap_rows);
            let mut row = 0usize;
            for &t in steps {
                let x_step = &x_full[row * d..(row + t) * d];
                let mut k_row = vec![0.0f32; t * width];
                let mut v_row = vec![0.0f32; t * width];
                cpu::gemm_f32(t, width, d, x_step, &k_w, kb, &mut k_row).unwrap();
                cpu::gemm_f32(t, width, d, x_step, &v_w, vb, &mut v_row).unwrap();
                host.append(0, &k_row, &v_row);
                host.advance(t);
                row += t;
            }
            let dk = max_abs_diff(&dev_k, host.k(0));
            let dv = max_abs_diff(&dev_v, host.v(0));
            assert!(
                dk <= ATOL,
                "device K vs host KvCache max|Δ| {dk:.3e} > {ATOL} (case {ci} bias={with_bias})"
            );
            assert!(
                dv <= ATOL,
                "device V vs host KvCache max|Δ| {dv:.3e} > {ATOL} (case {ci} bias={with_bias})"
            );
            worst_cpu = worst_cpu.max(dk).max(dv);
        }
    }
    eprintln!(
        "Metal KV cache append: offset-write bit-identical (Δ=0) to monolithic projection; vs host KvCache max|Δ| = {worst_cpu:.3e} (atol {ATOL})"
    );
}

/// Feeding a device K/V cache (built by the [`MetalContext::kv_append`]
/// primitive) into the Phase-1 causal fused attention
/// ([`MetalContext::attn_causal_f32`]) must match the host path — a [`KvCache`]
/// filled by the CPU projection, then the same causal multi-head attention on the
/// CPU — within the FP32 `ATOL`. This closes the loop the primitive exists for:
/// append K/V on the device across steps, then attend over the whole cache
/// causally. The queries are the last `t_q` positions (`q_offset = t_kv - t_q`),
/// so the final query attends every cached key.
#[test]
fn kv_cache_feeds_causal_attention_matches_cpu() {
    let ctx = ctx_or_skip!("kv cache causal attn");
    // (d, n_head, key-append steps, t_q); every d is divisible by n_head.
    let cases: [(usize, usize, &[usize], usize); 3] = [
        (16, 2, &[3, 1, 1], 2),
        (24, 3, &[4, 2, 1, 1], 1),
        (32, 8, &[6, 1, 1], 3),
    ];
    let mut worst = 0.0f32;
    for (ci, &(d, n_head, steps, t_q)) in cases.iter().enumerate() {
        let t_kv: usize = steps.iter().sum();
        assert!(t_q <= t_kv);
        let q_offset = t_kv - t_q;
        let hd = d / n_head;
        let scale = (hd as f32).powf(-0.5);
        let seed = 0x4B56_1000u64 ^ (ci as u64 * 17);

        // Self-attention K/V projection weights (width == d) + query operands.
        let x_full = rand_vec(seed, t_kv * d);
        let k_w = rand_vec(seed ^ 0x11, d * d);
        let v_w = rand_vec(seed ^ 0x22, d * d);
        let xq = rand_vec(seed ^ 0x33, t_q * d);
        let q_w = rand_vec(seed ^ 0x44, d * d);
        let out_w = rand_vec(seed ^ 0x55, d * d);

        let kw_dev = ctx.upload(&k_w).unwrap();
        let vw_dev = ctx.upload(&v_w).unwrap();

        // Device cache: append the keys/values step by step.
        let mut cache = ctx.new_kv_cache(t_kv, d).unwrap();
        let mut row = 0usize;
        for &t in steps {
            let x_dev = ctx.upload(&x_full[row * d..(row + t) * d]).unwrap();
            ctx.kv_append(&mut cache, t, d, &x_dev, &kw_dev, None, &vw_dev, None)
                .unwrap();
            row += t;
        }
        let mut dev_k = vec![0.0f32; t_kv * d];
        let mut dev_v = vec![0.0f32; t_kv * d];
        ctx.kv_download(&cache, &mut dev_k, &mut dev_v).unwrap();

        // Device causal attention over the downloaded cache.
        let mut dev_out = vec![0.0f32; t_q * d];
        ctx.attn_causal_f32(
            t_q,
            t_kv,
            d,
            n_head,
            &xq,
            &q_w,
            None,
            &dev_k,
            &dev_v,
            &out_w,
            None,
            scale,
            q_offset,
            &mut dev_out,
        )
        .expect("attn_causal_f32");

        // Host reference: KvCache (CPU projection) + CPU causal attention.
        let mut host = KvCache::with_reserve(1, d, t_kv);
        let mut row = 0usize;
        for &t in steps {
            let x_step = &x_full[row * d..(row + t) * d];
            let mut k_row = vec![0.0f32; t * d];
            let mut v_row = vec![0.0f32; t * d];
            cpu::gemm_f32(t, d, d, x_step, &k_w, None, &mut k_row).unwrap();
            cpu::gemm_f32(t, d, d, x_step, &v_w, None, &mut v_row).unwrap();
            host.append(0, &k_row, &v_row);
            host.advance(t);
            row += t;
        }
        let cpu_gemm = |m: usize,
                        n: usize,
                        kk: usize,
                        a: &[f32],
                        b: &[f32],
                        bias: Option<&[f32]>,
                        o: &mut [f32]| {
            cpu::gemm_f32(m, n, kk, a, b, bias, o).expect("cpu gemm");
        };
        // Causal softmax: mask keys `j > q_offset + i` before the plain softmax —
        // exactly the host `-inf` mask the GPU causal kernel replicates.
        let causal_softmax = |inp: &[f32], o: &mut [f32], rows: usize, cols: usize| {
            let mut masked = inp.to_vec();
            for i in 0..rows {
                for j in (q_offset + i + 1)..cols {
                    masked[i * cols + j] = f32::NEG_INFINITY;
                }
            }
            cpu::softmax_f32(&masked, o, rows, cols).expect("cpu softmax");
        };
        let cpu_out = attn_reference(
            &cpu_gemm,
            &causal_softmax,
            t_q,
            t_kv,
            d,
            n_head,
            &xq,
            &q_w,
            None,
            host.k(0),
            host.v(0),
            &out_w,
            None,
            scale,
        );

        let diff = max_abs_diff(&dev_out, &cpu_out);
        assert!(
            diff <= ATOL,
            "device KV → causal attn vs host KvCache + CPU causal attn max|Δ| {diff:.3e} > {ATOL} (case {ci})"
        );
        worst = worst.max(diff);
    }
    eprintln!(
        "Metal KV cache → causal attention vs host KvCache + CPU: global max|Δ| = {worst:.3e} (atol {ATOL})"
    );
}

// =====================================================================
// M3-04 fused KV-cache dequant + GEMV parity
// =====================================================================
//
// The three MSL kernels (`vokra_dequant_gemv_q{4,5,8}_0_f32`) are the GPU
// implementation of the [`vokra_core::KvQuantDequantGemvOps`] seam. Their CPU
// differential oracle is
// [`vokra_core::kv_quant::dequant_gemm::dequant_gemv_scalar`] — a scalar
// row-major `y = A · x` GEMV with per-block dequant in the reduction. Because
// both paths consume the identical on-wire byte layout produced by
// [`vokra_core::kv_quant::dequant_gemm::pack_matrix_to_bytes`], the parity is
// exact up to FP32 GEMV rounding.

/// Shape parameters covering the two attention-head widths the current model
/// zoo actually uses: Whisper `d_head = 64` (2 blocks / row) and Kokoro
/// `d_head = 128` (4 blocks / row). The `1 × 1` and `1 × 8` shapes stress the
/// tail-guard branch of the grid launch.
const M3_04_SHAPES: &[(usize, usize)] = &[
    (1, 1),
    (1, 8),
    (2, 2),
    (4, 2),
    (16, 2),
    (32, 4),
    (128, 2),
    (256, 4),
];

#[test]
fn dequant_gemv_metal_matches_cpu_q8_0() {
    let ctx = ctx_or_skip!("dequant_gemv Q8_0");
    dequant_gemv_parity(&ctx, vokra_core::KvQuant::Q8_0);
}

#[test]
fn dequant_gemv_metal_matches_cpu_q5_0() {
    let ctx = ctx_or_skip!("dequant_gemv Q5_0");
    dequant_gemv_parity(&ctx, vokra_core::KvQuant::Q5_0);
}

#[test]
fn dequant_gemv_metal_matches_cpu_q4_0() {
    let ctx = ctx_or_skip!("dequant_gemv Q4_0");
    dequant_gemv_parity(&ctx, vokra_core::KvQuant::Q4_0);
}

fn dequant_gemv_parity(ctx: &MetalContext, mode: vokra_core::KvQuant) {
    let mut worst = 0.0f32;
    for &(n_rows, n_bpr) in M3_04_SHAPES {
        let per_row_len = n_bpr * 32;
        // Deterministic FP32 matrix + x vector.
        let a = rand_vec(0xD3 ^ ((n_rows * 131 + n_bpr) as u64), n_rows * per_row_len);
        let x = rand_vec(0xE5 ^ (per_row_len as u64), per_row_len);

        // Pack the matrix into on-wire bytes using the CPU packer — the exact
        // same byte payload feeds the GPU kernel and the CPU differential
        // oracle. Bit-identical bytes = bit-identical dequant → any difference
        // is FP32 GEMV rounding only.
        let bytes =
            vokra_core::kv_quant::dequant_gemm::pack_matrix_to_bytes(mode, &a, n_rows, n_bpr)
                .expect("pack");

        // CPU oracle.
        let cpu_y = vokra_core::kv_quant::dequant_gemm::dequant_gemv_scalar(
            mode, &bytes, n_rows, n_bpr, &x,
        )
        .expect("cpu dequant_gemv_scalar");

        // GPU kernel through the direct method, then the trait entry point;
        // both must produce byte-identical outputs (same launcher).
        let gpu_direct = ctx
            .dequant_gemv_f32(mode, &bytes, n_rows, n_bpr, &x)
            .expect("metal dequant_gemv_f32 (direct)");
        let gpu_trait = {
            use vokra_core::KvQuantDequantGemvOps;
            ctx.fused_dequant_gemv(mode, &bytes, n_rows, n_bpr, &x)
                .expect("metal fused_dequant_gemv (trait)")
        };
        assert_eq!(
            gpu_direct, gpu_trait,
            "trait and direct entry points must be identical for {mode:?}"
        );

        let d = max_abs_diff(&gpu_direct, &cpu_y);
        eprintln!("dequant_gemv {mode:?} n_rows={n_rows:<4} n_bpr={n_bpr:<3} max|Δ|={d:.3e}");
        assert!(
            d <= 1e-4,
            "{mode:?} n_rows={n_rows} n_bpr={n_bpr}: {d} > 1e-4 (fused kernel drift too large)"
        );
        worst = worst.max(d);
    }
    eprintln!("Metal fused dequant_gemv {mode:?} vs CPU: global max|Δ| = {worst:.3e} (bound 1e-4)");
}

#[test]
fn dequant_gemv_metal_rejects_fp32_mode() {
    let ctx = ctx_or_skip!("dequant_gemv Fp32 rejection");
    // A well-formed byte payload for Q8_0 shape, but with mode=Fp32 — must
    // surface as an explicit error (never a silent GPU fallback, FR-EX-08).
    let bytes = vec![0u8; 4 * 2 * 34];
    let x = vec![0.0f32; 64];
    let err = ctx
        .dequant_gemv_f32(vokra_core::KvQuant::Fp32, &bytes, 4, 2, &x)
        .unwrap_err();
    match err {
        vokra_core::VokraError::InvalidArgument(msg) => {
            assert!(msg.contains("Fp32"), "unexpected message: {msg}");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn dequant_gemv_metal_rejects_shape_mismatch() {
    let ctx = ctx_or_skip!("dequant_gemv shape mismatch");
    // Q8_0 with 4 rows × 2 bpr expects 272 bytes; feed 100 to trigger the
    // shared validate.
    let bytes = vec![0u8; 100];
    let x = vec![0.0f32; 64];
    let err = ctx
        .dequant_gemv_f32(vokra_core::KvQuant::Q8_0, &bytes, 4, 2, &x)
        .unwrap_err();
    assert!(matches!(err, vokra_core::VokraError::InvalidArgument(_)));
}
