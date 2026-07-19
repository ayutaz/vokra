//! M4-13-T04〜T08 — native GLSL-mirror parity, HOST-UNCONDITIONAL.
//!
//! The device-gated suite (`tests/parity_vulkan.rs`, T12) is the *real* GPU
//! truth, but it SKIPS on the Apple-Silicon authoring host (no Vulkan loader,
//! and the glslc `.spv` blobs are an owner task, M4-13-T16). So a `.comp`
//! transcription bug — the exact failure the M3-02 skeleton shipped, a **tanh**
//! GELU while *claiming* CPU parity, and the subgroup-only reductions that are
//! wrong when the workgroup spans several subgroups — is invisible on this
//! machine until an owner runs lavapipe.
//!
//! This file closes that gap. For every `kernels/glsl/*.comp` there is a
//! pure-Rust **mirror** that transcribes the committed shader arithmetic
//! operation-for-operation, diffed against the `vokra-backend-cpu` kernel of
//! identical shape/semantics within the NFR-QL-01 FP32 gate (`atol = 0.01`).
//! These tests run on **every** host with **no** `vulkan` feature and **no**
//! device — that is the whole point: they catch an arithmetic drift the
//! device-gated suite cannot see here.
//!
//! # Faithful mirrors of the barrier-based reductions (M4-13-T04〜T06)
//!
//! `gemv` / `softmax` / `softmax_causal` / `layer_norm` were rewritten off the
//! subgroup skeleton to **barrier-based shared-memory tree reductions**
//! (`WG_SIZE = 32` lanes, `scratch[lid] += scratch[lid + s]` for `s =
//! WG_SIZE/2 … 1`). The mirrors reproduce that tree order faithfully, so the
//! parity vs the CPU backend's *sequential* reduction holds **within atol, not
//! bit-for-bit** — exactly as each `.comp` header documents.
//!
//! # Drift guard
//!
//! A mirror is only trustworthy while it matches the bytes on disk, so each
//! test also asserts the committed `.comp` (loaded via `include_str!`) still
//! contains the **needle** substrings its mirror was transcribed from. Every
//! needle was `grep -F`-confirmed present in the fc8feec source before being
//! asserted here; a silent `.comp` edit that drifts from the mirror trips the
//! needle assertion. (The subgroup-skeleton names — `subgroupAdd`,
//! `subgroupMax`, `subgroupElect`, `tanh` — appear in the *explanatory header
//! comments* that document the rewrite, so their **absence** is NOT a usable
//! guard; the positive tree-reduction / erf needles are. The one exception is
//! GELU's `0.044715`: that tanh-approximation magic number is genuinely absent,
//! so its continued absence is a meaningful anti-regression check.)
//!
//! # Kernels with no dedicated CPU oracle
//!
//! `transpose` / `gather` / `softmax_causal` have no matching
//! `vokra-backend-cpu::kernels` function (see that crate's `kernels/mod.rs`:
//! transpose / embedding-lookup are "deliberately not SIMD kernels here", and
//! there is no `softmax_causal_f32`). Those use a hand fixture plus
//! self-consistency invariants; `softmax_causal` additionally cross-checks the
//! documented `exp(-inf) = 0` host-mask equivalence against the real
//! `softmax_f32` primitive.

use vokra_backend_cpu::kernels as cpu;

/// FP32 parity gate (NFR-QL-01) — the same ceiling as the device-gated suite.
const ATOL: f32 = 0.01;

/// Workgroup width of the reduction shaders: `const uint WG_SIZE = 32u;` in
/// `gemv` / `softmax` / `softmax_causal` / `layer_norm` (pinned by each test's
/// drift guard). The mirrors iterate exactly this many lanes so the tree
/// reduction order matches the shader.
const WG_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// Shared fixtures + helpers (mirrors the parity_vulkan.rs generators so both
// suites feed the kernels the same well-conditioned distributions).
// ---------------------------------------------------------------------------

/// Deterministic SplitMix64-derived f32s in `[-1, 1)` (same generator as
/// `tests/parity_vulkan.rs`).
fn splitmix_f32s(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            ((z >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
        })
        .collect()
}

/// Xavier-ish scaling so deep reductions stay well-conditioned (keeps the
/// tree-vs-sequential FP32 gap far inside `atol`).
fn splitmix_weights(seed: u64, len: usize, fan_in: usize) -> Vec<f32> {
    let scale = 1.0 / (fan_in as f32).sqrt();
    splitmix_f32s(seed, len).iter().map(|v| v * scale).collect()
}

/// Assert `got` and `want` agree within `ATOL`, logging the max abs delta
/// (never a fabricated pass — a real divergence fails loudly).
fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut max_abs = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        assert!(
            d <= ATOL,
            "{what}: diverged at {i}: got {g}, want {w} (|Δ| = {d} > atol {ATOL})"
        );
        max_abs = max_abs.max(d);
    }
    eprintln!("{what}: max |Δ| = {max_abs:.3e} (atol {ATOL})");
}

/// Drift guard: every needle must be present verbatim in the committed `.comp`
/// source. Each was `grep -F`-confirmed present in fc8feec before being
/// asserted; if a `.comp` edit drops it, the mirror has silently drifted and
/// this fails.
fn assert_needles(kernel: &str, src: &str, needles: &[&str]) {
    for n in needles {
        assert!(
            src.contains(n),
            "{kernel}.comp drift: needle absent -> {n:?}\n     the pure-Rust mirror in this \
             file was transcribed from that substring; update the mirror AND this needle \
             together (M4-13 GLSL-mirror parity)."
        );
    }
}

// ===========================================================================
// gelu.comp — exact (erf-based) GELU (M4-13-T06). erf coefficients transcribed
// verbatim (A&S 7.1.26); identical to vokra-backend-cpu's ERF_P / ERF_A1..A5.
// ===========================================================================

const SRC_GELU: &str = include_str!("../kernels/glsl/gelu.comp");

// Spelled identically to vokra-backend-cpu's ERF_P / ERF_A1..A5
// (kernels/scalar.rs), which carry the same allow: the canonical A&S constants
// are kept verbatim (auditable) and the excess digits round to the same f32
// harmlessly.
#[allow(clippy::excessive_precision)]
const ERF_P: f32 = 0.327_591_1;
#[allow(clippy::excessive_precision)]
const ERF_A1: f32 = 0.254_829_592;
#[allow(clippy::excessive_precision)]
const ERF_A2: f32 = -0.284_496_736;
#[allow(clippy::excessive_precision)]
const ERF_A3: f32 = 1.421_413_741;
#[allow(clippy::excessive_precision)]
const ERF_A4: f32 = -1.453_152_027;
#[allow(clippy::excessive_precision)]
const ERF_A5: f32 = 1.061_405_429;
// Verbatim from `const float FRAC_1_SQRT_2 = 0.70710678;` in gelu.comp. clippy
// would rewrite it to `std::f32::consts::FRAC_1_SQRT_2` (a value differing by
// <1 ULP) or trim a digit — but a mirror reproduces the shader's own literal,
// and the drift needle pins this exact spelling.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
const FRAC_1_SQRT_2_GLSL: f32 = 0.70710678;

/// Mirror of the shader's `erf_approx`.
fn mirror_erf_approx(x: f32) -> f32 {
    let sign_x = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + ERF_P * ax);
    let poly = ((((ERF_A5 * t + ERF_A4) * t + ERF_A3) * t + ERF_A2) * t + ERF_A1) * t;
    sign_x * (1.0 - poly * (-ax * ax).exp())
}

/// Mirror of `out_buf[i] = 0.5 * x * (1.0 + erf_approx(x * FRAC_1_SQRT_2));`.
fn mirror_gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + mirror_erf_approx(x * FRAC_1_SQRT_2_GLSL))
}

#[test]
fn gelu_mirror_matches_cpu_and_pins_erf_form() {
    assert_needles(
        "gelu",
        SRC_GELU,
        &[
            "erf_approx(x * FRAC_1_SQRT_2)",
            "0.5 * x * (1.0 + erf_approx(x * FRAC_1_SQRT_2));",
            "const float ERF_P  = 0.3275911;",
            "const float FRAC_1_SQRT_2 = 0.70710678;",
            "poly = ((((ERF_A5 * t + ERF_A4) * t + ERF_A3) * t + ERF_A2) * t + ERF_A1) * t;",
        ],
    );
    // Anti-regression: the tanh-approx GELU (the M3-02 skeleton bug) is defined
    // by the magic number 0.044715. Its absence proves the committed shader
    // stayed on the erf form. (`tanh` itself only appears in the header comment
    // that narrates the fix, so it is not a usable negative needle — 0.044715
    // is: grep -F confirmed it absent in fc8feec.)
    assert!(
        !SRC_GELU.contains("0.044715"),
        "gelu.comp regressed to the tanh approximation (0.044715 reappeared)"
    );

    // ±4 stretches the erf() spread; the exact/erf CPU kernel is the oracle.
    let x: Vec<f32> = splitmix_f32s(150, 4096).iter().map(|v| v * 4.0).collect();
    let got: Vec<f32> = x.iter().map(|&v| mirror_gelu(v)).collect();
    let mut want = vec![0.0f32; x.len()];
    cpu::gelu_f32(&x, &mut want).expect("cpu gelu");
    assert_close(&got, &want, "gelu mirror vs cpu (erf form)");
}

// ===========================================================================
// gemv.comp — y[i] = Σ_j A[i,j]·x[j] (+ b[i]) via a barrier-based tree
// reduction (M4-13-T04). One workgroup per row; WG_SIZE lanes stride the sum.
// ===========================================================================

const SRC_GEMV: &str = include_str!("../kernels/glsl/gemv.comp");

/// Mirror of gemv.comp: per row, WG_SIZE lanes each sum a strided slice into
/// `scratch`, then a `s = WG_SIZE/2 … 1` tree reduction, then optional bias.
fn mirror_gemv(m: usize, n: usize, a: &[f32], x: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let mut y = vec![0.0f32; m];
    for (i, yi) in y.iter_mut().enumerate() {
        let mut scratch = [0.0f32; WG_SIZE];
        for (lid, slot) in scratch.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            let mut j = lid;
            while j < n {
                acc += a[i * n + j] * x[j];
                j += WG_SIZE;
            }
            *slot = acc;
        }
        let mut s = WG_SIZE / 2;
        while s > 0 {
            for lid in 0..s {
                scratch[lid] += scratch[lid + s];
            }
            s >>= 1;
        }
        let mut r = scratch[0];
        if let Some(b) = bias {
            r += b[i];
        }
        *yi = r;
    }
    y
}

#[test]
fn gemv_mirror_matches_cpu_and_pins_tree_reduction() {
    assert_needles(
        "gemv",
        SRC_GEMV,
        &[
            "const uint WG_SIZE = 32u;",
            "acc += A[i * pc.n + j] * x[j];",
            "for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {",
            "scratch[lid] += scratch[lid + s];",
            "barrier();",
            "r += b[i];",
        ],
    );

    // Whisper-ish decoder-step shape: m rows, k=n=512 inner (the reduction
    // depth that drives the FP32 budget). Both bias and no-bias legs.
    let (m, n) = (384usize, 512usize);
    let a = splitmix_weights(200, m * n, n);
    let x = splitmix_f32s(201, n);
    let bias = splitmix_f32s(202, m);

    for (label, b) in [("no-bias", None), ("bias", Some(&bias[..]))] {
        let got = mirror_gemv(m, n, &a, &x, b);
        let mut want = vec![0.0f32; m];
        cpu::gemv_f32(m, n, &a, &x, b, &mut want).expect("cpu gemv");
        assert_close(
            &got,
            &want,
            &format!("gemv mirror vs cpu [{label}] {m}x{n}"),
        );
    }
}

// ===========================================================================
// softmax.comp — numerically-stable row softmax via two tree reductions
// (M4-13-T05). One workgroup per row; -FLT_MAX sentinel = -3.402823e38.
// ===========================================================================

const SRC_SOFTMAX: &str = include_str!("../kernels/glsl/softmax.comp");

/// Mirror of softmax.comp: pass 1 tree-max, pass 2 tree-sum of `exp(x - max)`,
/// pass 3 normalise. `-3.402823e38` is the shader's -FLT_MAX lane sentinel.
fn mirror_softmax(rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let base = row * cols;
        // Pass 1: row max (tree).
        let mut scratch = [0.0f32; WG_SIZE];
        for (lid, slot) in scratch.iter_mut().enumerate() {
            let mut lane_max = -3.402823e38f32;
            let mut j = lid;
            while j < cols {
                lane_max = lane_max.max(x[base + j]);
                j += WG_SIZE;
            }
            *slot = lane_max;
        }
        let mut s = WG_SIZE / 2;
        while s > 0 {
            for lid in 0..s {
                scratch[lid] = scratch[lid].max(scratch[lid + s]);
            }
            s >>= 1;
        }
        let row_max = scratch[0];
        // Pass 2: Σ exp(x - max) (tree).
        for (lid, slot) in scratch.iter_mut().enumerate() {
            let mut lane_sum = 0.0f32;
            let mut j = lid;
            while j < cols {
                lane_sum += (x[base + j] - row_max).exp();
                j += WG_SIZE;
            }
            *slot = lane_sum;
        }
        let mut s = WG_SIZE / 2;
        while s > 0 {
            for lid in 0..s {
                scratch[lid] += scratch[lid + s];
            }
            s >>= 1;
        }
        let inv_sum = 1.0 / scratch[0];
        // Pass 3: write y = exp(x - max) / sum.
        for j in 0..cols {
            out[base + j] = (x[base + j] - row_max).exp() * inv_sum;
        }
    }
    out
}

#[test]
fn softmax_mirror_matches_cpu_and_pins_tree_reduction() {
    assert_needles(
        "softmax",
        SRC_SOFTMAX,
        &[
            "float lane_max = -3.402823e38; // -FLT_MAX",
            "lane_sum += exp(in_buf[row * pc.cols + j] - row_max);",
            "scratch[lid] = max(scratch[lid], scratch[lid + s]);",
            "float inv_sum = 1.0 / scratch[0];",
            "barrier();",
        ],
    );

    let (rows, cols) = (48usize, 512usize);
    // ±4 stretches the exp() spread (Whisper scores are Q·Kᵀ/√64-scaled).
    let scores: Vec<f32> = splitmix_f32s(210, rows * cols)
        .iter()
        .map(|v| v * 4.0)
        .collect();
    let got = mirror_softmax(rows, cols, &scores);
    let mut want = vec![0.0f32; rows * cols];
    cpu::softmax_f32(&scores, &mut want, rows, cols).expect("cpu softmax");
    assert_close(&got, &want, &format!("softmax mirror vs cpu {rows}x{cols}"));
}

// ===========================================================================
// layer_norm.comp — (x - mean)/sqrt(var + eps)·gamma + beta, biased variance,
// two tree reductions (M4-13-T06). `* inv_cols`, never `/ cols`.
// ===========================================================================

const SRC_LAYER_NORM: &str = include_str!("../kernels/glsl/layer_norm.comp");

/// Mirror of layer_norm.comp: tree-sum mean, tree-sum biased variance, then
/// normalise + affine. Uses `* inv_cols` op-for-op like the shader / CPU.
fn mirror_layer_norm(
    rows: usize,
    cols: usize,
    eps: f32,
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let base = row * cols;
        let inv_cols = 1.0 / cols as f32;
        // Pass 1: mean = (Σ x) * inv_cols (tree).
        let mut scratch = [0.0f32; WG_SIZE];
        for (lid, slot) in scratch.iter_mut().enumerate() {
            let mut lane_sum = 0.0f32;
            let mut j = lid;
            while j < cols {
                lane_sum += x[base + j];
                j += WG_SIZE;
            }
            *slot = lane_sum;
        }
        let mut s = WG_SIZE / 2;
        while s > 0 {
            for lid in 0..s {
                scratch[lid] += scratch[lid + s];
            }
            s >>= 1;
        }
        let mean = scratch[0] * inv_cols;
        // Pass 2: biased variance = (Σ (x - mean)^2) * inv_cols (tree).
        for (lid, slot) in scratch.iter_mut().enumerate() {
            let mut lane_var = 0.0f32;
            let mut j = lid;
            while j < cols {
                let d = x[base + j] - mean;
                lane_var += d * d;
                j += WG_SIZE;
            }
            *slot = lane_var;
        }
        let mut s = WG_SIZE / 2;
        while s > 0 {
            for lid in 0..s {
                scratch[lid] += scratch[lid + s];
            }
            s >>= 1;
        }
        let var = scratch[0] * inv_cols;
        let inv_std = 1.0 / (var + eps).sqrt();
        // Pass 3: normalise + scale + shift.
        for j in 0..cols {
            let v = (x[base + j] - mean) * inv_std;
            out[base + j] = v * gamma[j] + beta[j];
        }
    }
    out
}

#[test]
fn layer_norm_mirror_matches_cpu_and_pins_biased_variance() {
    assert_needles(
        "layer_norm",
        SRC_LAYER_NORM,
        &[
            "float inv_cols = 1.0 / float(pc.cols);",
            "float mean = scratch[0] * inv_cols;",
            "lane_var += d * d;",
            "float inv_std = 1.0 / sqrt(var + pc.eps);",
            "out_buf[row * pc.cols + j] = v * gamma[j] + beta[j];",
            "barrier();",
        ],
    );

    let (rows, cols) = (48usize, 512usize);
    let eps = 1e-5f32; // Whisper's PyTorch nn.LayerNorm default, passed verbatim.
    let x = splitmix_f32s(220, rows * cols);
    let gamma = splitmix_f32s(221, cols);
    let beta = splitmix_f32s(222, cols);
    let got = mirror_layer_norm(rows, cols, eps, &x, &gamma, &beta);
    let mut want = vec![0.0f32; rows * cols];
    cpu::layer_norm_f32(&x, &mut want, rows, cols, &gamma, &beta, eps).expect("cpu layer_norm");
    assert_close(
        &got,
        &want,
        &format!("layer_norm mirror vs cpu {rows}x{cols}"),
    );
}

// ===========================================================================
// activation.comp — relu / sigmoid / tanh selected by the KIND spec constant
// (M4-13-T07).
// ===========================================================================

const SRC_ACTIVATION: &str = include_str!("../kernels/glsl/activation.comp");

/// Activation KIND, mirroring the spec-constant values in activation.comp.
#[derive(Clone, Copy)]
enum ActKind {
    Relu,
    Sigmoid,
    Tanh,
}

/// Mirror of activation.comp's per-element body for each KIND.
fn mirror_activation(kind: ActKind, x: f32) -> f32 {
    match kind {
        ActKind::Relu => 0.0f32.max(x),               // KIND 0: max(0.0, x)
        ActKind::Sigmoid => 1.0 / (1.0 + (-x).exp()), // KIND 1
        ActKind::Tanh => x.tanh(),                    // KIND 2
    }
}

#[test]
fn activation_mirror_matches_cpu_for_each_kind() {
    assert_needles(
        "activation",
        SRC_ACTIVATION,
        &[
            "y = max(0.0, x);",
            "y = 1.0 / (1.0 + exp(-x));",
            "y = tanh(x);",
            "layout(constant_id = 0) const uint KIND = 0u;",
        ],
    );

    // ±4 covers the saturating tails of sigmoid / tanh and the relu hinge.
    let x: Vec<f32> = splitmix_f32s(230, 4096).iter().map(|v| v * 4.0).collect();

    let relu: Vec<f32> = x
        .iter()
        .map(|&v| mirror_activation(ActKind::Relu, v))
        .collect();
    let mut want = vec![0.0f32; x.len()];
    cpu::relu_f32(&x, &mut want).expect("cpu relu");
    assert_close(&relu, &want, "activation[relu] mirror vs cpu");

    let sig: Vec<f32> = x
        .iter()
        .map(|&v| mirror_activation(ActKind::Sigmoid, v))
        .collect();
    let mut want = vec![0.0f32; x.len()];
    cpu::sigmoid_f32(&x, &mut want).expect("cpu sigmoid");
    assert_close(&sig, &want, "activation[sigmoid] mirror vs cpu");

    let tanh: Vec<f32> = x
        .iter()
        .map(|&v| mirror_activation(ActKind::Tanh, v))
        .collect();
    let mut want = vec![0.0f32; x.len()];
    cpu::tanh_f32(&x, &mut want).expect("cpu tanh");
    assert_close(&tanh, &want, "activation[tanh] mirror vs cpu");
}

// ===========================================================================
// conv1d.comp — direct (non-im2col) batched 1-D convolution (M4-13-T07). The
// CPU oracle is im2col + GEMM but accumulates in the same (ic, kk) order.
// ===========================================================================

const SRC_CONV1D: &str = include_str!("../kernels/glsl/conv1d.comp");

/// Mirror of conv1d.comp: one output element `[b, oc, t]` per invocation,
/// summing over (ic, kk) with `int` window indexing and padding skip.
#[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
fn mirror_conv1d(
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    out_len: usize,
    kernel_len: usize,
    stride: usize,
    padding: usize,
    in_buf: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_ch * out_len];
    for b in 0..batch {
        for oc in 0..out_ch {
            for t in 0..out_len {
                let mut acc = 0.0f32;
                let t_in_base = (t * stride) as i32 - padding as i32;
                for ic in 0..in_ch {
                    for kk in 0..kernel_len {
                        let t_in = t_in_base + kk as i32;
                        if t_in < 0 || t_in >= in_len as i32 {
                            continue;
                        }
                        let in_idx = ((b * in_ch) + ic) * in_len + t_in as usize;
                        let w_idx = ((oc * in_ch) + ic) * kernel_len + kk;
                        acc += in_buf[in_idx] * weight[w_idx];
                    }
                }
                if let Some(bs) = bias {
                    acc += bs[oc];
                }
                let out_idx = ((b * out_ch) + oc) * out_len + t;
                out[out_idx] = acc;
            }
        }
    }
    out
}

#[test]
fn conv1d_mirror_matches_cpu_over_stride_envelope() {
    assert_needles(
        "conv1d",
        SRC_CONV1D,
        &[
            "int t_in_base = int(t * pc.stride) - int(pc.padding);",
            "acc += in_buf[in_idx] * weight[w_idx];",
            "uint w_idx = ((oc * pc.in_ch) + ic) * pc.kernel_len + kk;",
            "uint out_idx = ((b * pc.out_ch) + oc) * pc.out_len + t;",
        ],
    );

    // Whisper conv-stem envelope, batch = 1: conv1 (k3 s1 p1) then conv2
    // (k3 s2 p1). Channel counts reduced for speed; the (ic, kk) reduction
    // depth is what drives the FP32 budget.
    let (in_ch, out_ch, in_len, k) = (80usize, 64usize, 64usize, 3usize);
    let mel = splitmix_f32s(240, in_ch * in_len);
    let w1 = splitmix_weights(241, out_ch * in_ch * k, in_ch * k);
    let b1 = splitmix_f32s(242, out_ch);

    for (stride, padding, label) in [(1usize, 1usize, "s1 p1"), (2usize, 1usize, "s2 p1")] {
        let out_len = (in_len + 2 * padding - k) / stride + 1;
        let got = mirror_conv1d(
            1,
            in_ch,
            out_ch,
            in_len,
            out_len,
            k,
            stride,
            padding,
            &mel,
            &w1,
            Some(&b1),
        );
        let mut want = vec![0.0f32; out_ch * out_len];
        cpu::conv1d_f32(
            &mel,
            in_ch,
            in_len,
            &w1,
            out_ch,
            k,
            Some(&b1),
            stride,
            padding,
            &mut want,
        )
        .expect("cpu conv1d");
        assert_close(&got, &want, &format!("conv1d mirror vs cpu [{label}]"));
    }
}

// ===========================================================================
// elementwise.comp — add / mul selected by the OP spec constant (M4-13-T07).
// ===========================================================================

const SRC_ELEMENTWISE: &str = include_str!("../kernels/glsl/elementwise.comp");

/// Elementwise OP, mirroring the spec-constant values in elementwise.comp.
#[derive(Clone, Copy)]
enum EwOp {
    Add,
    Mul,
}

/// Mirror of elementwise.comp's per-element body for each OP.
fn mirror_elementwise(op: EwOp, a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(&av, &bv)| match op {
            EwOp::Add => av + bv,
            EwOp::Mul => av * bv,
        })
        .collect()
}

#[test]
fn elementwise_mirror_matches_cpu_for_add_and_mul() {
    assert_needles(
        "elementwise",
        SRC_ELEMENTWISE,
        &[
            "out_buf[i] = a[i] + b[i];",
            "out_buf[i] = a[i] * b[i];",
            "layout(constant_id = 0) const uint OP = 0u; // 0 = add, 1 = mul",
        ],
    );

    let a = splitmix_f32s(250, 4096);
    let b = splitmix_f32s(251, 4096);

    let add = mirror_elementwise(EwOp::Add, &a, &b);
    let mut want = vec![0.0f32; a.len()];
    cpu::add_f32(&a, &b, &mut want).expect("cpu add");
    assert_close(&add, &want, "elementwise[add] mirror vs cpu");

    let mul = mirror_elementwise(EwOp::Mul, &a, &b);
    let mut want = vec![0.0f32; a.len()];
    cpu::mul_f32(&a, &b, &mut want).expect("cpu mul");
    assert_close(&mul, &want, "elementwise[mul] mirror vs cpu");
}

// ===========================================================================
// gemm_subgroup.comp / gemm_coopmat.comp — out[m,n] = Σ_k a[m,k]·b[k,n], row
// major (M4-13-T03). Both share the same accumulation body (the coop-matrix
// path is the identical-to-subgroup skeleton until the extension lands), so
// both mirror `mirror_gemm` and each pins its own header needles.
// ===========================================================================

const SRC_GEMM_SUBGROUP: &str = include_str!("../kernels/glsl/gemm_subgroup.comp");
const SRC_GEMM_COOPMAT: &str = include_str!("../kernels/glsl/gemm_coopmat.comp");

/// Mirror of the shared GEMM body: sequential k-loop inner product, row-major.
fn mirror_gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for gm in 0..m {
        for gn in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[gm * k + kk] * b[kk * n + gn];
            }
            out[gm * n + gn] = acc;
        }
    }
    out
}

#[test]
fn gemm_subgroup_mirror_matches_cpu_and_pins_indexing() {
    assert_needles(
        "gemm_subgroup",
        SRC_GEMM_SUBGROUP,
        &[
            "acc += lhs[idx_lhs(gm, kk)] * rhs[idx_rhs(kk, gn)];",
            "uint idx_lhs(uint mm, uint kk) { return mm * pc.k + kk; }",
            "uint idx_rhs(uint kk, uint nn) { return kk * pc.n + nn; }",
            "#extension GL_KHR_shader_subgroup_arithmetic : enable",
        ],
    );

    let (m, n, k) = (48usize, 64usize, 512usize); // k = deepest reduction.
    let a = splitmix_weights(260, m * k, k);
    let b = splitmix_weights(261, k * n, k);
    let got = mirror_gemm(m, n, k, &a, &b);
    let mut want = vec![0.0f32; m * n];
    cpu::gemm_f32(m, n, k, &a, &b, None, &mut want).expect("cpu gemm");
    assert_close(
        &got,
        &want,
        &format!("gemm_subgroup mirror vs cpu {m}x{n}x{k}"),
    );
}

#[test]
fn gemm_coopmat_mirror_matches_cpu_and_pins_indexing() {
    assert_needles(
        "gemm_coopmat",
        SRC_GEMM_COOPMAT,
        &[
            "acc += lhs[gm * pc.k + kk] * rhs[kk * pc.n + gn];",
            "#extension GL_KHR_cooperative_matrix : enable",
            "out_buf[gm * pc.n + gn] = acc;",
        ],
    );

    let (m, n, k) = (48usize, 64usize, 512usize);
    let a = splitmix_weights(262, m * k, k);
    let b = splitmix_weights(263, k * n, k);
    let got = mirror_gemm(m, n, k, &a, &b);
    let mut want = vec![0.0f32; m * n];
    cpu::gemm_f32(m, n, k, &a, &b, None, &mut want).expect("cpu gemm");
    assert_close(
        &got,
        &want,
        &format!("gemm_coopmat mirror vs cpu {m}x{n}x{k}"),
    );
}

// ===========================================================================
// transpose.comp — 2-D transpose [m,n] → [n,m] (M4-13-T08). No CPU-kernel
// oracle (vokra-backend-cpu leaves transpose to the model layer), so a hand
// fixture + a double-transpose-is-identity self-consistency check.
// ===========================================================================

const SRC_TRANSPOSE: &str = include_str!("../kernels/glsl/transpose.comp");

/// Mirror of `out_buf[j * pc.m + i] = in_buf[i * pc.n + j];`.
fn mirror_transpose(m: usize, n: usize, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = x[i * n + j];
        }
    }
    out
}

#[test]
fn transpose_mirror_hand_fixture_and_self_consistency() {
    assert_needles(
        "transpose",
        SRC_TRANSPOSE,
        &["out_buf[j * pc.m + i] = in_buf[i * pc.n + j];"],
    );

    // Hand fixture: [[1,2,3],[4,5,6]] (2x3) → [[1,4],[2,5],[3,6]] (3x2).
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let got = mirror_transpose(2, 3, &x);
    assert_eq!(got, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    // Self-consistency: transpose∘transpose = identity, bit-verbatim.
    let (m, n) = (7usize, 11usize);
    let src = splitmix_f32s(270, m * n);
    let round_trip = mirror_transpose(n, m, &mirror_transpose(m, n, &src));
    for (a, b) in round_trip.iter().zip(&src) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "double transpose must be identity"
        );
    }
}

// ===========================================================================
// gather.comp — embedding lookup out[i,:] = in[indices[i],:] with an OOB →
// zero-row guard (M4-13-T08). No CPU-kernel oracle; hand fixture (incl. an OOB
// index) + an identity-gather self-consistency check.
// ===========================================================================

const SRC_GATHER: &str = include_str!("../kernels/glsl/gather.comp");

/// Mirror of gather.comp: `idx < vocab` copies the row, else a zero row.
fn mirror_gather(vocab: usize, dim: usize, table: &[f32], indices: &[u32]) -> Vec<f32> {
    let mut out = vec![0.0f32; indices.len() * dim];
    for (i, &raw) in indices.iter().enumerate() {
        let idx = raw as usize;
        for d in 0..dim {
            let v = if idx < vocab {
                table[idx * dim + d]
            } else {
                0.0
            };
            out[i * dim + d] = v;
        }
    }
    out
}

#[test]
fn gather_mirror_hand_fixture_and_self_consistency() {
    assert_needles(
        "gather",
        SRC_GATHER,
        &[
            "v = in_buf[idx * pc.dim + d];",
            "out_buf[i * pc.dim + d] = v;",
            "if (idx < pc.vocab) {",
        ],
    );

    // Hand fixture: vocab 3, dim 2, one OOB index (5 ≥ vocab) → zero row.
    let table = [10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
    let indices = [2u32, 0, 5];
    let got = mirror_gather(3, 2, &table, &indices);
    assert_eq!(got, [30.0, 31.0, 10.0, 11.0, 0.0, 0.0]);

    // Self-consistency: identity indices reproduce the table bit-verbatim.
    let (vocab, dim) = (16usize, 32usize);
    let big = splitmix_f32s(280, vocab * dim);
    let ids: Vec<u32> = (0..vocab as u32).collect();
    let round_trip = mirror_gather(vocab, dim, &big, &ids);
    for (a, b) in round_trip.iter().zip(&big) {
        assert_eq!(a.to_bits(), b.to_bits(), "identity gather must be verbatim");
    }
}

// ===========================================================================
// softmax_causal.comp — decoder self-attention softmax over cols 0..=row;
// masked cols get exactly 0 (M4-13-T05). No dedicated CPU kernel, so: a hand
// fixture + self-consistency (valid entries sum to 1, masked entries are
// bit-zero) + the documented `exp(-inf) = 0` host-mask equivalence against the
// real softmax_f32 primitive.
// ===========================================================================

const SRC_SOFTMAX_CAUSAL: &str = include_str!("../kernels/glsl/softmax_causal.comp");

/// Mirror of softmax_causal.comp: tree reductions over `valid_cols = row + 1`
/// (clamped to cols); masked cols written as exactly `0.0`.
fn mirror_softmax_causal(rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let base = row * cols;
        let valid_cols = (row + 1).min(cols);
        // Pass 1: max over valid cols (tree).
        let mut scratch = [0.0f32; WG_SIZE];
        for (lid, slot) in scratch.iter_mut().enumerate() {
            let mut lane_max = -3.402823e38f32;
            let mut j = lid;
            while j < valid_cols {
                lane_max = lane_max.max(x[base + j]);
                j += WG_SIZE;
            }
            *slot = lane_max;
        }
        let mut s = WG_SIZE / 2;
        while s > 0 {
            for lid in 0..s {
                scratch[lid] = scratch[lid].max(scratch[lid + s]);
            }
            s >>= 1;
        }
        let row_max = scratch[0];
        // Pass 2: Σ exp(x - max) over valid cols (tree).
        for (lid, slot) in scratch.iter_mut().enumerate() {
            let mut lane_sum = 0.0f32;
            let mut j = lid;
            while j < valid_cols {
                lane_sum += (x[base + j] - row_max).exp();
                j += WG_SIZE;
            }
            *slot = lane_sum;
        }
        let mut s = WG_SIZE / 2;
        while s > 0 {
            for lid in 0..s {
                scratch[lid] += scratch[lid + s];
            }
            s >>= 1;
        }
        let inv_sum = 1.0 / scratch[0];
        // Pass 3: valid cols get the softmax value, masked cols exactly 0.
        for j in 0..cols {
            out[base + j] = if j < valid_cols {
                (x[base + j] - row_max).exp() * inv_sum
            } else {
                0.0
            };
        }
    }
    out
}

#[test]
fn softmax_causal_mirror_hand_fixture_self_consistency_and_mask_equivalence() {
    assert_needles(
        "softmax_causal",
        SRC_SOFTMAX_CAUSAL,
        &[
            "uint valid_cols = row + 1u;",
            "exp(in_buf[row * pc.cols + j] - row_max) * inv_sum;",
            "out_buf[row * pc.cols + j] = 0.0;",
            "float lane_max = -3.402823e38; // -FLT_MAX",
            "barrier();",
        ],
    );

    // Hand fixture: 3x3. Row 0 sees only col 0 → [1,0,0]. Rows 1/2 softmax
    // over their causal prefix; masked cols are exactly 0.
    let x = [
        5.0, -9.0, 3.0, // row 0: only col 0 valid
        1.0, 1.0, 7.0, // row 1: cols 0,1 valid (equal → 0.5,0.5), col 2 masked
        0.0, 0.0, 0.0, // row 2: all valid, equal → 1/3 each
    ];
    let got = mirror_softmax_causal(3, 3, &x);
    // Row 0.
    assert_eq!(got[0], 1.0);
    assert_eq!(got[1].to_bits(), 0.0f32.to_bits());
    assert_eq!(got[2].to_bits(), 0.0f32.to_bits());
    // Row 1: equal logits → 0.5 / 0.5, col 2 masked to exact 0.
    assert!((got[3] - 0.5).abs() < 1e-6 && (got[4] - 0.5).abs() < 1e-6);
    assert_eq!(got[5].to_bits(), 0.0f32.to_bits());
    // Row 2: 1/3 each.
    for &g in &got[6..9] {
        assert!((g - 1.0 / 3.0).abs() < 1e-6);
    }

    // Self-consistency + host-mask equivalence at Whisper-ish decoder shape.
    let (rows, cols) = (48usize, 48usize);
    let scores: Vec<f32> = splitmix_f32s(290, rows * cols)
        .iter()
        .map(|v| v * 4.0)
        .collect();
    let got = mirror_softmax_causal(rows, cols, &scores);

    // Self-consistency: each row's valid prefix sums to 1; masked cols bit-0.
    for row in 0..rows {
        let base = row * cols;
        let mut sum = 0.0f32;
        for (j, &g) in got[base..base + cols].iter().enumerate() {
            if j <= row {
                sum += g;
            } else {
                assert_eq!(g.to_bits(), 0.0f32.to_bits(), "masked col must be exact 0");
            }
        }
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "row {row} valid prefix must sum to 1 (got {sum})"
        );
    }

    // Documented `exp(-inf) = 0` equivalence: masking cols j > i to -inf and
    // running the real softmax_f32 must match the causal mirror within atol.
    let mut masked = scores.clone();
    for row in 0..rows {
        for j in (row + 1)..cols {
            masked[row * cols + j] = f32::NEG_INFINITY;
        }
    }
    let mut want = vec![0.0f32; rows * cols];
    cpu::softmax_f32(&masked, &mut want, rows, cols).expect("cpu softmax (host-masked)");
    assert_close(
        &got,
        &want,
        &format!("softmax_causal mirror vs host-masked cpu {rows}x{cols}"),
    );
}
