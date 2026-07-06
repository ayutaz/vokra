//! CUDA numerical parity for the Phase-4 kernels (M2-03 T10-T14): the FP32 GPU
//! `gemv` / `softmax` / `layer_norm` / `gelu` / `conv1d` vs the `vokra-backend-cpu`
//! kernels (M0-08) that are the same differential oracle the scalar⇔SIMD harness
//! and the Metal port use. Ceiling is the NFR-QL-01 FP32 bound `atol = 0.01` (the
//! observed error is far smaller and logged per shape).
//!
//! Like the GEMM parity suite (`parity_cuda.rs`), every test is **device-gated**:
//! [`CudaContext::new`] gates each one, so a CUDA-less host (e.g. the Apple Mac
//! this crate is authored on) skips rather than fails. The real GPU comparison is
//! meant to run on the **vast.ai RTX 4090** runner (M2-03-T25), NOT on this
//! machine — mirroring `parity_kernels_metal.rs` exactly so the two GPU backends
//! are checked against the same oracle and shapes.

use vokra_backend_cpu::kernels as cpu;
use vokra_backend_cuda::CudaContext;
use vokra_core::{KvCache, PrenormLayer};

/// NFR-QL-01 FP32 parity ceiling.
const ATOL: f32 = 0.01;

/// Deterministic pseudo-random f32 in roughly [-1, 1) (xorshift64*), matching the
/// Metal / GEMM parity suites so inputs are reproducible across backends.
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

/// Builds a context or prints a skip and returns from the test (no CUDA device).
macro_rules! ctx_or_skip {
    ($what:literal) => {
        match CudaContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    concat!(
                        "no CUDA device (",
                        $what,
                        "); skipping (run on vast.ai RTX 4090): {}"
                    ),
                    e
                );
                return;
            }
        }
    };
}

#[test]
fn gemv_cuda_matches_cpu() {
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
                .expect("cuda gemv");
            let mut cpu_out = vec![0.0f32; m];
            cpu::gemv_f32(m, k, &a, &x, bias, &mut cpu_out).expect("cpu gemv");

            let d = max_abs_diff(&gpu, &cpu_out);
            eprintln!("gemv  m={m:<5} k={k:<5} bias={with_bias:<5} max|Δ|={d:.3e}");
            assert!(d <= ATOL, "gemv m={m} k={k} bias={with_bias}: {d} > {ATOL}");
            worst = worst.max(d);
        }
    }
    eprintln!("gemv CUDA vs CPU: global max|Δ| = {worst:.3e} (atol {ATOL})");
}

#[test]
fn softmax_cuda_matches_cpu() {
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
            .expect("cuda softmax");
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
        .expect("cuda softmax masked");
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
    eprintln!("softmax CUDA vs CPU: global max|Δ| = {worst:.3e} (atol {ATOL})");
}

#[test]
fn layer_norm_cuda_matches_cpu() {
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
            .expect("cuda layer_norm");
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
    eprintln!("layer_norm CUDA vs CPU: global max|Δ| = {worst:.3e} (atol {ATOL})");
}

#[test]
fn gelu_cuda_matches_cpu() {
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
        ctx.gelu_f32(&x, &mut gpu).expect("cuda gelu");
        let mut cpu_out = vec![0.0f32; n];
        cpu::gelu_f32(&x, &mut cpu_out).expect("cpu gelu");
        let d = max_abs_diff(&gpu, &cpu_out);
        eprintln!("gelu  n={n:<6} max|Δ|={d:.3e}");
        assert!(d <= ATOL, "gelu n={n}: {d} > {ATOL}");
        worst = worst.max(d);
    }
    eprintln!("gelu CUDA vs CPU: global max|Δ| = {worst:.3e} (atol {ATOL})");
}

#[test]
fn conv1d_cuda_matches_cpu() {
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
        .expect("cuda conv1d");
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
    eprintln!("conv1d CUDA vs CPU: global max|Δ| = {worst:.3e} (atol {ATOL})");
}

/// Shape mismatches are explicit `InvalidArgument`, not a GPU fault (mirrors the
/// GEMM shape-validation test and the Metal kernels suite).
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

/// The fused `CudaContext::mlp_f32` (fc1 GEMM → GELU → fc2 GEMM on one stream,
/// the `[t, ffn]` intermediates kept device-resident, one D2H of `out`) must be
/// **bit-identical** to running the same three GPU kernels per-op
/// (`gemm_f32` → `gelu_f32` → `gemm_f32`, three D2Hs) — same kernels, same
/// order, same launch geometry — and must match the CPU three-kernel reference
/// within the FP32 bound. Device-gated: skips on a CUDA-less host (this Mac);
/// runs for real on the vast.ai RTX 4090 (M2-03-T25). `d` is the fc1-in /
/// fc2-out width, `ffn` the fc1-out / fc2-in width.
#[test]
fn mlp_fused_matches_sequential_and_cpu() {
    let ctx = ctx_or_skip!("mlp");
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

            // Fused GPU (device-resident intermediates, one D2H).
            let mut fused = vec![0.0f32; t * d];
            ctx.mlp_f32(t, d, ffn, &x, &fc1_w, b1, &fc2_w, b2, &mut fused)
                .expect("fused mlp");

            // Per-op GPU (same kernels, three D2Hs) — the bit-identical reference.
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
        "CUDA fused MLP parity: vs per-op GPU max|Δ| = {worst_seq:.3e} (bit-identical), vs CPU max|Δ| = {worst_cpu:.3e} (atol = {ATOL})"
    );
}

// ---- Phase-5: fused device-resident non-causal attention --------------------

/// Per-op non-causal multi-head attention, replicating `whisper::nn::
/// attention_from_kv_into`'s head loop **exactly** (q-proj GEMM → scale the whole
/// `q` → per head {host gather qh/vh, host gather-transpose kh_t, scores GEMM,
/// softmax, context GEMM, host scatter} → out-proj GEMM), with the GEMM / softmax
/// supplied as closures — the CUDA twin of the Metal parity suite's helper.
/// Passing the GPU (`ctx.*`) closures yields the bit-identical per-op reference
/// the fused `attn_f32` must equal; passing the CPU (`cpu::*`) closures yields the
/// FP32 oracle.
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

/// The fused `CudaContext::attn_f32` (q-proj → per-head {gather, QKᵀ, softmax,
/// A·V, scatter} → out-proj on one stream, every intermediate device-resident,
/// one D2H) must be **bit-identical** to the per-op path built from the same GPU
/// kernels (`attn_reference` with the `ctx.*` closures — same kernels, order and
/// launch geometry, the scale folded into the qh gather) and match the CPU
/// reference within the FP32 bound. Device-gated: skips on this CUDA-less Mac;
/// runs for real on the vast.ai RTX 4090 (M2-03-T25).
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

            // Fused GPU (device-resident intermediates, one D2H).
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
        "CUDA fused attn parity: vs per-op GPU max|Δ| = {worst_seq:.3e} (bit-identical), vs CPU max|Δ| = {worst_cpu:.3e} (atol = {ATOL})"
    );
}

/// Reports the submission / readback reduction the attention fusion buys at a
/// realistic Whisper-base encoder self-attention shape (`t_q = t_kv = 1500`,
/// `d = 512`, `n_head = 8`) and re-checks bit-identical parity at that scale.
/// Fused issues ONE `cuStreamSynchronize` + ONE D2H per call; the per-op path
/// issues `2 + 3·n_head = 26` synchronised launches each with its own D2H, plus
/// the host round-trips. Device-gated (runs on the vast.ai RTX 4090).
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
        "CUDA fused attn (t_q={t_q} t_kv={t_kv} d={d} n_head={n_head}, {iters} iters): \
         fused {fused_dt:?} ({:?}/it, 1 D2H + 1 sync) vs \
         per-op {perop_dt:?} ({:?}/it, {} D2Hs + {} syncs + host round-trips)",
        fused_dt / iters,
        perop_dt / iters,
        2 + 3 * n_head,
        2 + 3 * n_head,
    );
}

/// `attn_f32` rejects mis-sized / mis-configured operands with an explicit
/// `InvalidArgument` (never a device fault): here `d` (6) is not divisible by
/// `n_head` (4). Validation runs before any device call, so this fires even on a
/// CUDA-less host — hence it is NOT device-gated.
#[test]
fn attn_f32_rejects_missized() {
    let ctx = match CudaContext::new() {
        Ok(c) => c,
        Err(_) => return, // no device: the shape guard is exercised on the 4090 run
    };
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
// Symmetric with the Metal suite (`parity_kernels_metal.rs`). Device-gated via
// `ctx_or_skip!`, so this crate — authored on a CUDA-less Apple Mac — compiles
// and skips here; it runs for real on the vast.ai RTX 4090 (M2-03-T25).

/// LayerNorm epsilon (Whisper / the CPU kernel default).
const PRENORM_EPS: f32 = 1e-5;

/// Owned random weights for one pre-norm block (biases match Whisper: `q` / `v` /
/// `out` / `fc1` / `fc2` carry one, `k` does not).
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

/// The **current** per-op GPU encoder path (per block `layer_norm → k/v GEMM →
/// fused attn_f32 → host residual add → layer_norm → fused mlp_f32 → host residual
/// add`, then a final LayerNorm). The fused stack encodes the same kernels in the
/// same order, so this is the bit-identical reference and issues `6·N + 1`
/// stream synchronisations, the count the fused stack collapses to one.
#[allow(clippy::too_many_arguments)]
fn prenorm_reference_current(
    ctx: &CudaContext,
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

/// The per-op **CPU** encoder path (the onnxruntime-agreeing reference), for the
/// FP32-bound comparison.
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

/// `CudaContext::encode_prenorm_stack` must be **bit-identical** to the current
/// per-op GPU path (same kernels — one synchronise vs `6·N + 1`) and match the CPU
/// within the FP32 bound. Runs on the vast.ai RTX 4090; skips on a CUDA-less host.
#[test]
fn prenorm_stack_matches_sequential_and_cpu() {
    let ctx = ctx_or_skip!("prenorm stack");
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
        "CUDA prenorm-stack parity: vs per-op GPU max|Δ| = {worst_seq:.3e} (bit-identical), vs CPU max|Δ| = {worst_cpu:.3e} (atol = {ATOL})"
    );
}

/// The whole-encoder residency's payoff: the fused stack issues **exactly ONE**
/// stream synchronise for the whole encoder, versus the per-op path's `6·N + 1`
/// (measured with the context's submission counter, env-independent). Re-checks
/// bit-identical; prints wall time.
#[test]
fn prenorm_stack_reduces_readback() {
    let ctx = ctx_or_skip!("prenorm stack readback");
    let (t, d, ff, n_head, n_layers) = (256usize, 128usize, 512usize, 8usize, 4usize);
    let data: Vec<LayerData> = (0..n_layers)
        .map(|i| make_layer((i * 31 + 7) as u64, d, ff))
        .collect();
    let layers: Vec<PrenormLayer<'_>> = data.iter().map(layer_view).collect();
    let hidden = rand_vec(0x5151, t * d);
    let fg = rand_vec(0x61, d);
    let fb = rand_vec(0x62, d);

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
    assert_eq!(d_fused, 1, "the fused encoder must be ONE synchronise");
    assert_eq!(
        d_perop,
        (6 * n_layers + 1) as u64,
        "the per-op path must be 6·N + 1 synchronises"
    );
    eprintln!(
        "CUDA prenorm stack ({n_layers} layers, t={t} d={d} ff={ff} n_head={n_head}): \
         fused {fused_dt:?} ({d_fused} sync) vs per-op {perop_dt:?} ({d_perop} syncs)"
    );
}

/// Each public device-in/out op is **bit-identical** to its host-in/out sibling,
/// and `download(upload(x)) == x`.
#[test]
fn device_ops_match_host_in_out() {
    let ctx = ctx_or_skip!("device ops");

    let x = rand_vec(1, 37);
    let xt = ctx.upload(&x).expect("upload");
    let mut back = vec![0.0f32; 37];
    ctx.download(&xt, &mut back).expect("download");
    assert_eq!(back, x, "download(upload(x)) must equal x");

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

    eprintln!("CUDA device-in/out ops all bit-identical to their host-in/out siblings");
}

// ---- Decoder-step Phase 2: device-resident self-attention K/V cache ---------

/// The device [`CudaContext::kv_append`] primitive must reproduce a host
/// [`KvCache`] project-and-concatenate: each step's `k = x·Wk (+b)` /
/// `v = x·Wv (+b)` is written in place at the running row `len`. The multi-step
/// result must equal
/// (a) a single **monolithic** projection of all rows at once on the GPU — proving
///     the offset write neither shifts nor corrupts rows (`max|Δ| == 0`), and
/// (b) the host `KvCache` fed the same weights through the CPU GEMM oracle (within
///     the FP32 `ATOL`).
///
/// Mirrors `parity_kernels_metal.rs::kv_cache_append_matches_host_project_concat`
/// exactly so the two GPU backends are checked against the same oracle and shapes.
/// Device-gated: runs on the vast.ai RTX 4090, skips on this Mac. (The Metal suite
/// additionally checks causal attention over the cache; CUDA has no causal fused
/// attention yet, so that case is Metal-only.)
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
        "CUDA KV cache append: offset-write bit-identical (Δ=0) to monolithic projection; vs host KvCache max|Δ| = {worst_cpu:.3e} (atol {ATOL})"
    );
}
