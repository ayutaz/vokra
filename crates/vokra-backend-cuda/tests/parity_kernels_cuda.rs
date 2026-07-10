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
use vokra_backend_cuda::{CudaContext, CudaDecodeSession};
use vokra_core::{DecoderLayerView, KvCache, PrenormLayer};

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

// ---- M2-03 follow-up (c04): FlashAttention v2 causal parity vs decomposed ----

/// FA v2 (`launch_flash_attn_v2`, T-follow-02/03) must match the decomposed
/// `launch_attn_chain` (`causal = true` with matching `q_offset`) elementwise
/// within the NFR-QL-01 FP32 bound `atol = 0.01` (with a `rtol = 1e-5`
/// tie-break for large magnitudes).
///
/// This is the **primitive parity** step of the M2-03 RTF<0.1 follow-up plan
/// (T-follow-04). Both arms live on the GPU:
///   * Candidate: the fused Flash-Attention v2 kernel, dispatched through the
///     public [`CudaContext::flash_attn_dev`] wrapper (which bypasses the
///     internal `FA_V2_MIN_TQ = 16` gate — the sweep below stays at
///     `t_q ≥ 16` anyway, so the FA v2 tile is never wasted).
///   * Reference: the decomposed `2 + 7·n_head` causal chain, dispatched
///     through the public [`CudaContext::attn_causal_dev`] wrapper — the
///     SAME internal oracle Metal / CPU parity uses (no fabricated numbers,
///     no PyTorch import at test time).
///
/// The `t_q` sweep — `[16, 24, 32, 64, 448, 1500]` — covers:
///   * `16` — one full FA v2 query tile (`Br = 16`).
///   * `24` / `32` — 2 tiles, exercising the ragged-tail branch (`br_eff`).
///   * `64` — 4 tiles, still small enough that KV fits in a couple of
///     `Bc = 64` iterations.
///   * `448` — Whisper `n_text_ctx` (decoder-max full-prefix step).
///   * `1500` — Whisper `n_audio_ctx` (encoder-max, 30 s audio).
///
/// Each shape uses `t_kv = t_q` (prefix-step semantics, `q_offset = 0`; row
/// `i` attends keys `[0, i]` — the causal mask is meaningful for every row
/// except the last).
///
/// The RNG is `rand_vec` (xorshift, deterministic, seed 42) — the same
/// zero-dep PRNG the rest of this file uses. The spec calls for
/// `SmallRng::seed_from_u64(42)`; we honor the *seed* (`42`) and the
/// *determinism* contract but stay on the zero-dep xorshift because
/// `vokra-backend-cuda` deliberately has no `rand` dev-dependency
/// (`Cargo.toml` L21-33, NFR-DS-02).
///
/// Gated by the existing probe-skip macro: on a CUDA-less host (this M1 iMac)
/// the test prints the skip line and returns green; the real parity is
/// exercised on the **vast.ai RTX 4090** runner (M2-03-T25, T-follow-09).
#[test]
fn flash_attn_v2_causal_vs_decomposed_f32() {
    let ctx = ctx_or_skip!("flash_attn_v2_causal");

    // Whisper `d_head = 64` (base=512/8, medium=1024/16, large-v3=1280/20 all
    // land at 64). Fix `d_head = 64` and pick `n_head = 4` → `d = 256` so the
    // sweep stays within a single-GPU shared-memory budget while still
    // exercising the multi-head dispatch shape FA v2 will use in production.
    // `flash_attn_dev` also validates `d/n_head == 64` — this shape is the
    // only one it accepts (kernel's tile budget is `d_head`-fixed).
    let n_head = 4usize;
    let d_head = 64usize;
    let d = n_head * d_head;
    let scale = (d_head as f32).powf(-0.5);

    // The seed-42 contract (spec) — every shape derives its own sub-seed from
    // this base so the RNG streams don't collide across `t_q` cases.
    const SEED: u64 = 42;
    // FA v2 tile is `Br = 16`; the sweep stays at `t_q ≥ 16` so the kernel
    // is dispatched with a non-wasted tile (task M2-03-followup-rtf T04).
    //   * 16 — 1 query tile (br_eff == Br).
    //   * 24 / 32 — 2 tiles (24 exercises br_eff == 8 ragged tail).
    //   * 64 — 4 tiles.
    //   * 448 — Whisper `n_text_ctx` (decoder-max prefix step).
    //   * 1500 — Whisper `n_audio_ctx` (encoder-max, 30 s audio window).
    let t_q_sweep = [16usize, 24, 32, 64, 448, 1500];
    let mut worst_abs = 0.0f32;
    let mut worst_rel = 0.0f32;

    for &t_q in &t_q_sweep {
        // Prefix step: `t_kv = t_q`, `q_offset = 0` — row `i` attends keys
        // `[0, i]`. This is Whisper's decoder prefix decode (all `t_q` queries
        // seeded at once) and the tightest causal shape the FA v2 kernel
        // needs to get right (every row's mask is meaningful except row
        // `t_q - 1`).
        let t_kv = t_q;
        let q_offset = 0usize;

        // Deterministic sub-seeds: mix `SEED` and `t_q` so each shape gets
        // its own reproducible input tensors (the same recipe existing tests
        // in this file use for their per-shape sub-seeds).
        let s = SEED ^ ((t_q as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let xq = rand_vec(s ^ 0x01, t_q * d);
        let qw = rand_vec(s ^ 0x02, d * d);
        let kk = rand_vec(s ^ 0x03, t_kv * d);
        let vv = rand_vec(s ^ 0x04, t_kv * d);
        let ow = rand_vec(s ^ 0x05, d * d);
        let qb = rand_vec(s ^ 0x06, d);
        let ob = rand_vec(s ^ 0x07, d);

        // Upload the inputs once — the reference and the candidate share the
        // same device buffers so the parity Δ is *only* kernel-vs-kernel.
        let xqt = ctx.upload(&xq).unwrap();
        let qwt = ctx.upload(&qw).unwrap();
        let kt = ctx.upload(&kk).unwrap();
        let vt = ctx.upload(&vv).unwrap();
        let owt = ctx.upload(&ow).unwrap();
        let qbt = ctx.upload(&qb).unwrap();
        let obt = ctx.upload(&ob).unwrap();

        // ---- (a) Reference: decomposed `launch_attn_chain` (causal=true) ----
        //
        // The public [`CudaContext::attn_causal_dev`] wrapper forces
        // `use_flash_attn: false`, so this is the byte-for-byte
        // `2 + 7·n_head` chain (M2-03 parity oracle). Same math the CPU
        // sees — this preserves the internal-oracle rule (no PyTorch at
        // test time, no fabricated numbers).
        let mut ref_dev = ctx.alloc_dev(t_q * d).unwrap();
        ctx.attn_causal_dev(
            t_q,
            t_kv,
            d,
            n_head,
            &xqt,
            &qwt,
            Some(&qbt),
            &kt,
            &vt,
            &owt,
            Some(&obt),
            scale,
            q_offset,
            &mut ref_dev,
        )
        .expect("reference decomposed attn_causal_dev");
        let mut reference = vec![0.0f32; t_q * d];
        ctx.download(&ref_dev, &mut reference).unwrap();

        // ---- (b) Candidate: FA v2 fused kernel (causal=true) ----------------
        //
        // The public [`CudaContext::flash_attn_dev`] wrapper unconditionally
        // dispatches `launch_flash_attn_v2`, bypassing the internal
        // `FA_V2_MIN_TQ = 16` runtime gate (the sweep stays at `t_q ≥ 16`
        // anyway so no tile is wasted — the bypass matters only for
        // testing / diagnostic entrypoints).
        let mut cand_dev = ctx.alloc_dev(t_q * d).unwrap();
        ctx.flash_attn_dev(
            t_q,
            t_kv,
            d,
            n_head,
            &xqt,
            &qwt,
            Some(&qbt),
            &kt,
            &vt,
            &owt,
            Some(&obt),
            scale,
            /* causal = */ true,
            q_offset,
            &mut cand_dev,
        )
        .expect("candidate FA v2 flash_attn_dev");
        let mut candidate = vec![0.0f32; t_q * d];
        ctx.download(&cand_dev, &mut candidate).unwrap();

        // ---- (c) Parity: elementwise |Δ| ≤ max(atol, rtol · |ref|) ----------
        //
        // NFR-QL-01 FP32 bound `atol = 0.01`; `rtol = 1e-5` per spec covers
        // the large-magnitude out-proj output range without inflating the
        // pass condition on small values.
        const RTOL: f32 = 1e-5;
        for (i, (&r, &c)) in reference.iter().zip(&candidate).enumerate() {
            let abs = (r - c).abs();
            let tol = ATOL.max(RTOL * r.abs());
            assert!(
                abs <= tol,
                "flash_attn_v2 causal candidate diverges at t_q={t_q}, i={i}: |ref-cand| = {abs:.3e} > tol {tol:.3e} (ref={r:.4}, cand={c:.4})"
            );
            worst_abs = worst_abs.max(abs);
            worst_rel = worst_rel.max(if r.abs() > 0.0 { abs / r.abs() } else { 0.0 });
        }
    }

    eprintln!(
        "CUDA flash_attn_v2 causal vs decomposed: worst |Δ| = {worst_abs:.3e} (atol {ATOL}), worst rel = {worst_rel:.3e} (rtol 1e-5) over t_q ∈ {{16,24,32,64,448,1500}} (t_kv=t_q, q_offset=0)"
    );
}

/// Non-causal companion of [`flash_attn_v2_causal_vs_decomposed_f32`]: the FA
/// v2 kernel's `causal = false` branch must match the decomposed non-causal
/// chain ([`CudaContext::attn_dev`]) elementwise within the same FP32 bound
/// (NFR-QL-01 `atol = 0.01`, `rtol = 1e-5`). Same sweep as the causal test,
/// but `q_offset` is unused and every row attends every key.
///
/// This is not redundant with the causal test — it exercises the FA v2
/// kernel's dominant fast path (cross-attention: the encoder-side attention
/// [`CudaDecodeSession`] runs on every decoded token) with zero masked
/// positions, so any divergence between the reference and the candidate is
/// pure tile-arithmetic drift (no interaction with the `-INFINITY` mask
/// write inside the S_tile). The causal test above covers the mask
/// interaction; together they gate the full FA v2 kernel surface.
///
/// Device-gated: skips on this M1 iMac, runs on vast.ai RTX 4090.
#[test]
fn flash_attn_v2_noncausal_vs_decomposed_f32() {
    let ctx = ctx_or_skip!("flash_attn_v2_noncausal");

    let n_head = 4usize;
    let d_head = 64usize;
    let d = n_head * d_head;
    let scale = (d_head as f32).powf(-0.5);

    const SEED: u64 = 42;
    // Same `t_q ≥ 16` sweep as the causal test — cross-attention shapes in
    // production are also `t_q ≥ 1` (the encoder-max case `t_q = 1500` is
    // representative of the encoder self-attention Whisper does).
    let t_q_sweep = [16usize, 24, 32, 64, 448, 1500];
    let mut worst_abs = 0.0f32;
    let mut worst_rel = 0.0f32;

    for &t_q in &t_q_sweep {
        // Non-causal parity: match `t_kv = t_q` so the shapes are identical
        // to the causal test — same allocations, same launches, only the
        // `causal` flag toggles between the two tests.
        let t_kv = t_q;

        // Sub-seed distinct from the causal test's stream so we exercise a
        // different input distribution (guards against a lucky-input regression).
        let s = SEED ^ 0xBEEF ^ ((t_q as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let xq = rand_vec(s ^ 0x01, t_q * d);
        let qw = rand_vec(s ^ 0x02, d * d);
        let kk = rand_vec(s ^ 0x03, t_kv * d);
        let vv = rand_vec(s ^ 0x04, t_kv * d);
        let ow = rand_vec(s ^ 0x05, d * d);
        let qb = rand_vec(s ^ 0x06, d);
        let ob = rand_vec(s ^ 0x07, d);

        let xqt = ctx.upload(&xq).unwrap();
        let qwt = ctx.upload(&qw).unwrap();
        let kt = ctx.upload(&kk).unwrap();
        let vt = ctx.upload(&vv).unwrap();
        let owt = ctx.upload(&ow).unwrap();
        let qbt = ctx.upload(&qb).unwrap();
        let obt = ctx.upload(&ob).unwrap();

        // Reference: decomposed non-causal (`attn_dev`). Same math the M2-03
        // parity oracle uses — the byte-for-byte `2 + 7·n_head` chain with
        // the plain `vokra_softmax_f32` (no causal mask).
        let mut ref_dev = ctx.alloc_dev(t_q * d).unwrap();
        ctx.attn_dev(
            t_q,
            t_kv,
            d,
            n_head,
            &xqt,
            &qwt,
            Some(&qbt),
            &kt,
            &vt,
            &owt,
            Some(&obt),
            scale,
            &mut ref_dev,
        )
        .expect("reference decomposed attn_dev");
        let mut reference = vec![0.0f32; t_q * d];
        ctx.download(&ref_dev, &mut reference).unwrap();

        // Candidate: FA v2 non-causal. `q_offset = 0` is unused when
        // `causal = false` (the kernel branches on `causal` before reading
        // `q_offset`), but pass it as `0` for explicit intent.
        let mut cand_dev = ctx.alloc_dev(t_q * d).unwrap();
        ctx.flash_attn_dev(
            t_q,
            t_kv,
            d,
            n_head,
            &xqt,
            &qwt,
            Some(&qbt),
            &kt,
            &vt,
            &owt,
            Some(&obt),
            scale,
            /* causal = */ false,
            /* q_offset (ignored) = */ 0,
            &mut cand_dev,
        )
        .expect("candidate FA v2 flash_attn_dev (non-causal)");
        let mut candidate = vec![0.0f32; t_q * d];
        ctx.download(&cand_dev, &mut candidate).unwrap();

        const RTOL: f32 = 1e-5;
        for (i, (&r, &c)) in reference.iter().zip(&candidate).enumerate() {
            let abs = (r - c).abs();
            let tol = ATOL.max(RTOL * r.abs());
            assert!(
                abs <= tol,
                "flash_attn_v2 non-causal candidate diverges at t_q={t_q}, i={i}: |ref-cand| = {abs:.3e} > tol {tol:.3e} (ref={r:.4}, cand={c:.4})"
            );
            worst_abs = worst_abs.max(abs);
            worst_rel = worst_rel.max(if r.abs() > 0.0 { abs / r.abs() } else { 0.0 });
        }
    }

    eprintln!(
        "CUDA flash_attn_v2 non-causal vs decomposed: worst |Δ| = {worst_abs:.3e} (atol {ATOL}), worst rel = {worst_rel:.3e} (rtol 1e-5) over t_q ∈ {{16,24,32,64,448,1500}} (t_kv=t_q)"
    );
}

/// Validation guards on the public [`CudaContext::flash_attn_dev`] wrapper —
/// the FA v2 kernel's tile budget is hard-tuned for Whisper's `d_head = 64`,
/// and the causal contract requires the mask window fits inside `t_kv`. This
/// test validates both guards fire *before* any device launch, so a caller
/// on a non-Whisper shape or an off-by-one causal window gets an explicit
/// [`VokraError::InvalidArgument`] (FR-EX-08) instead of a silent
/// no-op / crash inside the kernel.
///
/// Device-gated (needs a `CudaContext` even to allocate the input tensors),
/// so this skips on the M1 iMac. Runs on vast.ai RTX 4090.
#[test]
fn flash_attn_dev_input_validation() {
    let ctx = ctx_or_skip!("flash_attn_dev validation");

    let n_head = 4usize;
    let d_head = 64usize;
    let d = n_head * d_head;
    let scale = (d_head as f32).powf(-0.5);
    let t_q = 16usize;
    let t_kv = 16usize;
    let xq = ctx.upload(&rand_vec(1, t_q * d)).unwrap();
    let qw = ctx.upload(&rand_vec(2, d * d)).unwrap();
    let k = ctx.upload(&rand_vec(3, t_kv * d)).unwrap();
    let v = ctx.upload(&rand_vec(4, t_kv * d)).unwrap();
    let ow = ctx.upload(&rand_vec(5, d * d)).unwrap();
    let mut out = ctx.alloc_dev(t_q * d).unwrap();

    // ---- Guard 1: d_head != 64 must reject (kernel's tile budget is
    // d_head-fixed at compile time). Try n_head = 2 → d_head = 128.
    let n_head_bad = 2usize;
    let res_dhead = ctx.flash_attn_dev(
        t_q, t_kv, d, n_head_bad, &xq, &qw, None, &k, &v, &ow, None, scale, false, 0, &mut out,
    );
    match res_dhead {
        Err(vokra_core::VokraError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("d/n_head") && msg.contains("64"),
                "expected d_head guard message, got: {msg}"
            );
        }
        other => panic!("expected InvalidArgument on d_head != 64, got: {other:?}"),
    }

    // ---- Guard 2: causal q_offset + t_q > t_kv must reject (row t_q-1 would
    // attend past the K/V window).
    let res_qoff = ctx.flash_attn_dev(
        t_q, t_kv, d, n_head, &xq, &qw, None, &k, &v, &ow, None, scale, /* causal = */ true,
        /* q_offset = */ 1, // 1 + 16 = 17 > t_kv=16
        &mut out,
    );
    match res_qoff {
        Err(vokra_core::VokraError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("q_offset") && msg.contains("t_q"),
                "expected q_offset guard message, got: {msg}"
            );
        }
        other => panic!("expected InvalidArgument on q_offset + t_q > t_kv, got: {other:?}"),
    }

    // ---- Guard 3: causal with q_offset = 0 and t_q == t_kv must succeed
    // (mask window is exactly t_kv, row t_q-1 attends every key).
    let res_ok = ctx.flash_attn_dev(
        t_q, t_kv, d, n_head, &xq, &qw, None, &k, &v, &ow, None, scale, /* causal = */ true,
        /* q_offset = */ 0, // 0 + 16 = 16 == t_kv=16, OK
        &mut out,
    );
    assert!(
        res_ok.is_ok(),
        "expected flash_attn_dev to accept q_offset + t_q == t_kv, got: {res_ok:?}"
    );

    eprintln!(
        "CUDA flash_attn_dev guards: d_head != 64 rejected, causal q_offset overflow rejected, boundary q_offset + t_q == t_kv accepted"
    );
}

// ---- M2-03 follow-up: CudaDecodeSession reuse via `reset()` -----------------

/// Deterministic decoder-weight fixture used by
/// [`session_reuse_bit_identical`]. Owns every f32 slice a
/// [`DecoderLayerView`] can borrow (all seeded off a single `u64`), plus the
/// token embedding / tied logits head and the final LayerNorm — so a caller
/// can build a [`DecoderLayerView`] slice by borrowing this struct without any
/// per-decode allocation drifting the seeds.
struct DecoderFixture {
    d: usize,
    n_head: usize,
    ff: usize,
    n_ctx: usize,
    n_text_ctx: usize,
    n_vocab: usize,
    /// Per-layer, per-tensor buffers. Order matches [`DecoderLayerView`]
    /// fields; the biases mirror the Whisper convention (present for q / v /
    /// out; absent for k) so this fixture stresses both `Some`/`None` bias
    /// paths inside `CudaDecodeSession::step`.
    layers: Vec<DecoderLayerFixture>,
    token_emb: Vec<f32>,
    ln_post_gamma: Vec<f32>,
    ln_post_beta: Vec<f32>,
    /// Two consecutive decode steps' `[t, d]` embeddings (t = 1 each — a
    /// steady-state single-token step, the tightest bit-identical contract
    /// since it exercises the `start` offset write on the resident KV cache).
    step0_embedded: Vec<f32>,
    step1_embedded: Vec<f32>,
}

struct DecoderLayerFixture {
    self_ln_gamma: Vec<f32>,
    self_ln_beta: Vec<f32>,
    self_q_w: Vec<f32>,
    self_q_bias: Vec<f32>,
    self_k_w: Vec<f32>,
    self_v_w: Vec<f32>,
    self_v_bias: Vec<f32>,
    self_out_w: Vec<f32>,
    self_out_bias: Vec<f32>,
    cross_ln_gamma: Vec<f32>,
    cross_ln_beta: Vec<f32>,
    cross_q_w: Vec<f32>,
    cross_q_bias: Vec<f32>,
    cross_out_w: Vec<f32>,
    cross_out_bias: Vec<f32>,
    cross_k: Vec<f32>,
    cross_v: Vec<f32>,
    mlp_ln_gamma: Vec<f32>,
    mlp_ln_beta: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_bias: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_bias: Vec<f32>,
}

impl DecoderFixture {
    /// Builds the shared fixture (dims / weights / two step embeddings) off a
    /// single seed so both paths in the reuse test read identical bytes.
    fn build() -> Self {
        // Tiny but non-degenerate: 2 layers, d=8, n_head=2 (hd=4), ff=16.
        // n_text_ctx=8 lets the two steady-state steps write rows 0 then 1 of
        // the resident self-KV; n_ctx=4 keeps the cross-KV footprint small.
        let d = 8usize;
        let n_head = 2usize;
        let ff = 16usize;
        let n_ctx = 4usize;
        let n_text_ctx = 8usize;
        let n_vocab = 12usize;
        let n_layers = 2usize;
        let seed_base = 0xC0DE_5E55_u64;

        let mut layers = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            let s = seed_base ^ ((li as u64) << 32);
            layers.push(DecoderLayerFixture {
                self_ln_gamma: rand_vec(s ^ 0x01, d),
                self_ln_beta: rand_vec(s ^ 0x02, d),
                self_q_w: rand_vec(s ^ 0x03, d * d),
                self_q_bias: rand_vec(s ^ 0x04, d),
                self_k_w: rand_vec(s ^ 0x05, d * d),
                self_v_w: rand_vec(s ^ 0x06, d * d),
                self_v_bias: rand_vec(s ^ 0x07, d),
                self_out_w: rand_vec(s ^ 0x08, d * d),
                self_out_bias: rand_vec(s ^ 0x09, d),
                cross_ln_gamma: rand_vec(s ^ 0x0A, d),
                cross_ln_beta: rand_vec(s ^ 0x0B, d),
                cross_q_w: rand_vec(s ^ 0x0C, d * d),
                cross_q_bias: rand_vec(s ^ 0x0D, d),
                cross_out_w: rand_vec(s ^ 0x0E, d * d),
                cross_out_bias: rand_vec(s ^ 0x0F, d),
                cross_k: rand_vec(s ^ 0x10, n_ctx * d),
                cross_v: rand_vec(s ^ 0x11, n_ctx * d),
                mlp_ln_gamma: rand_vec(s ^ 0x12, d),
                mlp_ln_beta: rand_vec(s ^ 0x13, d),
                fc1_w: rand_vec(s ^ 0x14, d * ff),
                fc1_bias: rand_vec(s ^ 0x15, ff),
                fc2_w: rand_vec(s ^ 0x16, ff * d),
                fc2_bias: rand_vec(s ^ 0x17, d),
            });
        }
        Self {
            d,
            n_head,
            ff,
            n_ctx,
            n_text_ctx,
            n_vocab,
            layers,
            token_emb: rand_vec(seed_base ^ 0xA000, n_vocab * d),
            ln_post_gamma: rand_vec(seed_base ^ 0xA001, d),
            ln_post_beta: rand_vec(seed_base ^ 0xA002, d),
            // Two `[t=1, d]` embeddings — same "audio segment" replayed twice.
            step0_embedded: rand_vec(seed_base ^ 0xB000, d),
            step1_embedded: rand_vec(seed_base ^ 0xB001, d),
        }
    }

    /// Borrows the fixture as a `Vec<DecoderLayerView<'_>>` suitable for
    /// [`CudaDecodeSession::new`]. Bias slots follow Whisper's convention:
    /// q / v / out biases present, k absent.
    fn views(&self) -> Vec<DecoderLayerView<'_>> {
        self.layers
            .iter()
            .map(|l| DecoderLayerView {
                self_ln_gamma: &l.self_ln_gamma,
                self_ln_beta: &l.self_ln_beta,
                self_q_w: &l.self_q_w,
                self_q_bias: Some(&l.self_q_bias),
                self_k_w: &l.self_k_w,
                self_k_bias: None,
                self_v_w: &l.self_v_w,
                self_v_bias: Some(&l.self_v_bias),
                self_out_w: &l.self_out_w,
                self_out_bias: Some(&l.self_out_bias),
                cross_ln_gamma: &l.cross_ln_gamma,
                cross_ln_beta: &l.cross_ln_beta,
                cross_q_w: &l.cross_q_w,
                cross_q_bias: Some(&l.cross_q_bias),
                cross_out_w: &l.cross_out_w,
                cross_out_bias: Some(&l.cross_out_bias),
                cross_k: &l.cross_k,
                cross_v: &l.cross_v,
                mlp_ln_gamma: &l.mlp_ln_gamma,
                mlp_ln_beta: &l.mlp_ln_beta,
                fc1_w: &l.fc1_w,
                fc1_bias: Some(&l.fc1_bias),
                fc2_w: &l.fc2_w,
                fc2_bias: Some(&l.fc2_bias),
            })
            .collect()
    }

    /// Constructs a fresh [`CudaDecodeSession`] against this fixture.
    fn new_session(&self) -> vokra_core::Result<CudaDecodeSession> {
        let eps = 1e-5f32;
        let max_t_q = 1usize; // steady-state single-token step, matches step0/step1
        CudaDecodeSession::new(
            self.d,
            self.n_head,
            self.ff,
            self.n_text_ctx,
            self.n_vocab,
            self.n_ctx,
            max_t_q,
            eps,
            &self.views(),
            &self.token_emb,
            &self.ln_post_gamma,
            &self.ln_post_beta,
        )
    }
}

/// **M2-03 follow-up** (change c06). The reuse-via-`reset()` seam the pooled
/// `CudaDecodeSessionPool` (D5) will hang off must be **bit-identical** to
/// freshly built sessions: replaying the SAME two `[t=1, d]` decode steps
/// (the "same GGUF + same audio segment" contract in the plan, expressed here
/// with the same deterministic synthetic fixture the rest of this file uses —
/// no GGUF asset is loaded, so the GGUF-gated skip is de facto satisfied by
/// the fixture path) on
///
/// - **(a)** a brand-new session per pair of steps (build → step(0) → step(1) → drop, twice), vs
/// - **(b)** ONE session that decodes steps (0, 1), then `reset()`, then decodes steps (0, 1) again,
///
/// must yield **`atol == 0.0`** on both `all_logits()` snapshots (both the
/// first and second replay). This is the parity contract the follow-up's D5
/// / R4 depend on: `reset()` fully clears the `pos` and `last_t` clocks, and
/// the self-KV rows the second replay overwrites at row 0 do not leak state
/// from the first replay's row 0 into any attention read (the causal
/// contract only touches `k/v[..pos + i]` after `reset()` snaps `pos = 0`).
///
/// Device-gated via probe (`ctx_or_skip!` matches every other test in this
/// file): runs for real on the vast.ai RTX 4090 (M2-03-T25); skips silently
/// on this CUDA-less Mac. The parent `--features cuda` reaches
/// `vokra-backend-cuda` transitively through `vokra-models`; running
/// `-p vokra-backend-cuda` directly compiles the crate unconditionally (this
/// crate defines no `cuda` feature of its own — see `Cargo.toml`).
#[test]
fn session_reuse_bit_identical() {
    // Probe the device up front so a CUDA-less host skips cleanly (matches
    // the ctx_or_skip! pattern the rest of this file uses; `CudaDecodeSession`
    // has no probe of its own — it just tries to create a context and fails).
    let _probe = ctx_or_skip!("session reuse");
    drop(_probe);

    let fix = DecoderFixture::build();

    // ---- Path (a): freshly built session per replay. ------------------------
    // Two independent sessions, each doing step(0) at pos 0 then step(1) at
    // pos 1 — the reference the reset() path must reproduce byte-for-byte.
    let a_run = |label: &str| -> (Vec<f32>, Vec<f32>) {
        let mut s = fix
            .new_session()
            .unwrap_or_else(|e| panic!("(a) {label}: build session: {e}"));
        assert_eq!(s.positions(), 0, "(a) {label}: fresh pos == 0");
        assert!(
            s.last_logits().is_empty(),
            "(a) {label}: fresh last_logits empty"
        );

        s.step(&fix.step0_embedded, 1, 0)
            .unwrap_or_else(|e| panic!("(a) {label}: step 0: {e}"));
        let logits0 = s.all_logits().to_vec();
        assert_eq!(s.positions(), 1, "(a) {label}: pos advances to 1");

        s.step(&fix.step1_embedded, 1, 1)
            .unwrap_or_else(|e| panic!("(a) {label}: step 1: {e}"));
        let logits1 = s.all_logits().to_vec();
        assert_eq!(s.positions(), 2, "(a) {label}: pos advances to 2");

        (logits0, logits1)
    };
    let (a0_run1, a1_run1) = a_run("run 1");
    let (a0_run2, a1_run2) = a_run("run 2");
    // Sanity: two freshly built sessions on the same deterministic fixture
    // are themselves bit-identical (this is the internal oracle contract —
    // if this fails, the fixture is not deterministic and the whole parity
    // is meaningless).
    assert_eq!(
        max_abs_diff(&a0_run1, &a0_run2),
        0.0,
        "(a) run 1 vs run 2 step 0 logits must be bit-identical",
    );
    assert_eq!(
        max_abs_diff(&a1_run1, &a1_run2),
        0.0,
        "(a) run 1 vs run 2 step 1 logits must be bit-identical",
    );

    // ---- Path (b): one session, reset() between the two replays. ------------
    let mut s = fix.new_session().expect("(b) build session");

    // First replay.
    s.step(&fix.step0_embedded, 1, 0).expect("(b) run 1 step 0");
    let b0_run1 = s.all_logits().to_vec();
    s.step(&fix.step1_embedded, 1, 1).expect("(b) run 1 step 1");
    let b1_run1 = s.all_logits().to_vec();
    assert_eq!(s.positions(), 2, "(b) pos == 2 after run 1");

    // The reset() contract: pos snaps to 0 and last_t clears (so
    // last_logits() / all_logits() view an empty prefix until the next step
    // writes it — same post-reset semantics the CPU decoder has).
    s.reset();
    assert_eq!(s.positions(), 0, "(b) reset() clears pos to 0");
    assert!(
        s.last_logits().is_empty(),
        "(b) reset() clears last_t (last_logits empty)"
    );
    assert!(
        s.all_logits().is_empty(),
        "(b) reset() clears last_t (all_logits empty)"
    );

    // Second replay through the SAME session — the same two embeddings at
    // the same positions the first replay used.
    s.step(&fix.step0_embedded, 1, 0).expect("(b) run 2 step 0");
    let b0_run2 = s.all_logits().to_vec();
    s.step(&fix.step1_embedded, 1, 1).expect("(b) run 2 step 1");
    let b1_run2 = s.all_logits().to_vec();
    assert_eq!(s.positions(), 2, "(b) pos == 2 after run 2");

    // ---- The parity contract ------------------------------------------------
    // Every replay's every step must be bit-identical between (a) and (b):
    // reset() must not leak ANY residual state (the pos clock, the KV rows
    // it will re-overwrite, the tied-head logits scratch, or the resident
    // `h` residual stream) into the second replay.
    let d_a_first_replay_step0 = max_abs_diff(&a0_run1, &b0_run1);
    let d_a_first_replay_step1 = max_abs_diff(&a1_run1, &b1_run1);
    let d_a_second_replay_step0 = max_abs_diff(&a0_run2, &b0_run2);
    let d_a_second_replay_step1 = max_abs_diff(&a1_run2, &b1_run2);
    let d_b_reset_step0 = max_abs_diff(&b0_run1, &b0_run2);
    let d_b_reset_step1 = max_abs_diff(&b1_run1, &b1_run2);

    assert_eq!(
        d_a_first_replay_step0, 0.0,
        "(a run 1) vs (b run 1) step 0 must be bit-identical (Δ={d_a_first_replay_step0:.3e})",
    );
    assert_eq!(
        d_a_first_replay_step1, 0.0,
        "(a run 1) vs (b run 1) step 1 must be bit-identical (Δ={d_a_first_replay_step1:.3e})",
    );
    assert_eq!(
        d_a_second_replay_step0, 0.0,
        "(a run 2) vs (b run 2 after reset) step 0 must be bit-identical (Δ={d_a_second_replay_step0:.3e})",
    );
    assert_eq!(
        d_a_second_replay_step1, 0.0,
        "(a run 2) vs (b run 2 after reset) step 1 must be bit-identical (Δ={d_a_second_replay_step1:.3e})",
    );
    // And the reset() path's two replays through the same session are equal.
    assert_eq!(
        d_b_reset_step0, 0.0,
        "(b) reset() replay must reproduce first-replay step 0 bit-identically (Δ={d_b_reset_step0:.3e})",
    );
    assert_eq!(
        d_b_reset_step1, 0.0,
        "(b) reset() replay must reproduce first-replay step 1 bit-identically (Δ={d_b_reset_step1:.3e})",
    );

    eprintln!(
        "CUDA session reset() reuse: (a) fresh-per-run vs (b) reset()-between-runs are bit-identical over 2 steps × 2 replays (max|Δ| = 0.0)"
    );
}

// ---- M3-01-T11 / T19: long-form decoder session reuse ----------------------

/// M3-01-T11 (long-form decoder session reuse) + T19 (session_reuse_bit_identical
/// extension) — exercise the KV cache reset contract over the fixture's full
/// `n_text_ctx` window. The M2-03 test above proves 2 steps × 2 replays; this
/// test pushes to `n_text_ctx` steps (max the fixture allows without changing
/// the resident KV cache shape), catching a leak that only shows up after
/// several rows of the resident self-KV have been written and then re-cleared
/// by `reset()`.
///
/// Positioning vs the M3-01 ticket: the ticket text asks for 60 s / 120 s
/// long-form audio (~200 / ~400 steps at 25 tok/s). The fixture's
/// `n_text_ctx = 8` is deliberately small (M2-03 land — keeps the parity
/// contract tight and CI runs fast). Extending it to 8 fills the whole
/// resident KV cache without shape churn; **the 60 / 120 s numbers in the
/// ticket require the vast.ai / self-hosted runner test using a real
/// whisper-large-v3 GGUF**, which is out of the CI scope. This test is the
/// structural sibling that lives on the vast.ai path via the same
/// `ctx_or_skip!` gate.
#[test]
fn session_reuse_bit_identical_long_form() {
    let _probe = ctx_or_skip!("session reuse long-form");
    drop(_probe);

    let fix = DecoderFixture::build();
    // Fill the whole n_text_ctx window (8 steps). Each step uses one of the
    // two step embeddings the fixture publishes, alternating deterministically
    // so the KV cache holds a mix — the same content on both paths so any
    // divergence is a reset() leak, not an input asymmetry.
    let n_steps = fix.n_text_ctx;
    let step_seq: Vec<&[f32]> = (0..n_steps)
        .map(|i| {
            if i % 2 == 0 {
                fix.step0_embedded.as_slice()
            } else {
                fix.step1_embedded.as_slice()
            }
        })
        .collect();

    // (a) Fresh sessions per replay — build → run all n_steps → collect
    // last-step logits — twice (should be internally bit-identical, the
    // fixture is deterministic).
    let run_a = || -> Vec<f32> {
        let mut s = fix.new_session().expect("(a) build session");
        for (pos, emb) in step_seq.iter().enumerate() {
            s.step(emb, 1, pos)
                .unwrap_or_else(|e| panic!("(a) step pos {pos}: {e}"));
        }
        assert_eq!(s.positions(), n_steps, "(a) pos == n_steps");
        s.all_logits().to_vec()
    };
    let a_run1 = run_a();
    let a_run2 = run_a();
    assert_eq!(
        max_abs_diff(&a_run1, &a_run2),
        0.0,
        "(a) two independent long-form runs must be bit-identical (fixture is deterministic)"
    );

    // (b) One session, reset() between the two replays.
    let mut s = fix.new_session().expect("(b) build session");
    for (pos, emb) in step_seq.iter().enumerate() {
        s.step(emb, 1, pos)
            .unwrap_or_else(|e| panic!("(b) run 1 step pos {pos}: {e}"));
    }
    let b_run1 = s.all_logits().to_vec();
    assert_eq!(s.positions(), n_steps, "(b) pos == n_steps after run 1");

    s.reset();
    assert_eq!(s.positions(), 0, "(b) reset() clears pos to 0");
    assert!(s.last_logits().is_empty(), "(b) reset() clears last_t");

    for (pos, emb) in step_seq.iter().enumerate() {
        s.step(emb, 1, pos)
            .unwrap_or_else(|e| panic!("(b) run 2 step pos {pos}: {e}"));
    }
    let b_run2 = s.all_logits().to_vec();
    assert_eq!(s.positions(), n_steps, "(b) pos == n_steps after run 2");

    // Long-form contract: reset() must fully clear the resident self-KV
    // (n_text_ctx rows worth) + the pos clock + last-token logits scratch.
    // Any residual state would surface as a non-zero Δ on the second replay.
    let d_a_vs_b_run1 = max_abs_diff(&a_run1, &b_run1);
    let d_a_vs_b_run2 = max_abs_diff(&a_run2, &b_run2);
    let d_b_reset = max_abs_diff(&b_run1, &b_run2);
    assert_eq!(
        d_a_vs_b_run1, 0.0,
        "(a run 1) vs (b run 1) long-form must be bit-identical (Δ={d_a_vs_b_run1:.3e})"
    );
    assert_eq!(
        d_a_vs_b_run2, 0.0,
        "(a run 2) vs (b run 2 after reset) long-form must be bit-identical (Δ={d_a_vs_b_run2:.3e})"
    );
    assert_eq!(
        d_b_reset, 0.0,
        "(b) reset() long-form replay must reproduce first replay bit-identically (Δ={d_b_reset:.3e})"
    );

    eprintln!(
        "CUDA long-form session reuse: {n_steps} steps × 2 replays are bit-identical (fresh-per-run vs reset()-between-runs, max|Δ| = 0.0). Fixture: n_text_ctx = {n_steps}."
    );
}

// =====================================================================
// M3-04 fused KV-cache dequant + GEMV parity
// =====================================================================
//
// The three CUDA kernels (`vokra_dequant_gemv_q{4,5,8}_0_f32`) are the GPU
// implementation of the [`vokra_core::KvQuantDequantGemvOps`] seam. Their CPU
// differential oracle is
// [`vokra_core::kv_quant::dequant_gemm::dequant_gemv_scalar`] — a scalar
// row-major `y = A · x` GEMV with per-block dequant in the reduction. Because
// both paths consume the identical on-wire byte layout produced by
// [`vokra_core::kv_quant::dequant_gemm::pack_matrix_to_bytes`], the parity is
// exact up to FP32 GEMV rounding; NFR-QL-01 pins the ceiling to `atol = 0.01`
// but for the small shapes we exercise here the observed drift is well under
// `atol = 1e-4`.

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
fn dequant_gemv_cuda_matches_cpu_q8_0() {
    let ctx = ctx_or_skip!("dequant_gemv Q8_0");
    dequant_gemv_parity(&ctx, vokra_core::KvQuant::Q8_0);
}

#[test]
fn dequant_gemv_cuda_matches_cpu_q5_0() {
    let ctx = ctx_or_skip!("dequant_gemv Q5_0");
    dequant_gemv_parity(&ctx, vokra_core::KvQuant::Q5_0);
}

#[test]
fn dequant_gemv_cuda_matches_cpu_q4_0() {
    let ctx = ctx_or_skip!("dequant_gemv Q4_0");
    dequant_gemv_parity(&ctx, vokra_core::KvQuant::Q4_0);
}

fn dequant_gemv_parity(ctx: &CudaContext, mode: vokra_core::KvQuant) {
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
            .expect("cuda dequant_gemv_f32 (direct)");
        let gpu_trait = {
            use vokra_core::KvQuantDequantGemvOps;
            ctx.fused_dequant_gemv(mode, &bytes, n_rows, n_bpr, &x)
                .expect("cuda fused_dequant_gemv (trait)")
        };
        assert_eq!(
            gpu_direct, gpu_trait,
            "trait and direct entry points must be identical for {mode:?}"
        );

        let d = max_abs_diff(&gpu_direct, &cpu_y);
        eprintln!("dequant_gemv {mode:?} n_rows={n_rows:<4} n_bpr={n_bpr:<3} max|Δ|={d:.3e}");
        // FP32 GEMV rounding ceiling on this shape; loose vs the tight
        // ATOL=0.01 for the raw kernels above, but the CPU oracle is bit-for-bit
        // the same scalar reduction so any drift is only FP32 non-associativity.
        assert!(
            d <= 1e-4,
            "{mode:?} n_rows={n_rows} n_bpr={n_bpr}: {d} > 1e-4 (fused kernel drift too large)"
        );
        worst = worst.max(d);
    }
    eprintln!("CUDA fused dequant_gemv {mode:?} vs CPU: global max|Δ| = {worst:.3e} (bound 1e-4)");
}

#[test]
fn dequant_gemv_cuda_rejects_fp32_mode() {
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
fn dequant_gemv_cuda_rejects_shape_mismatch() {
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
