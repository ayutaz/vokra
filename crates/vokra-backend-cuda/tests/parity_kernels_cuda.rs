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
