//! Differential harness: scalar oracle vs the host SIMD path (M0-08-T09).
//!
//! Every dispatch-table kernel is run twice on identical, fixed-seed random
//! inputs — once forced onto [`IsaPath::Scalar`] (the oracle) and once onto
//! the host's SIMD path ([`CpuFeatures::best_isa`], i.e. `Avx2` on an AVX2
//! x86-64 runner, `Neon` on AArch64) — and the outputs are compared. On a
//! host without a SIMD path the SIMD side resolves to `Scalar` and the check
//! degenerates to a trivial identity comparison, so the harness is green
//! everywhere (M0-08-T09) while genuinely exercising AVX2 / NEON where they
//! exist.
//!
//! # Tolerance policy (M0-08-T09)
//!
//! FMA reorders rounding, so bitwise equality is not required. The ceiling is
//! the FP32 parity bound NFR-QL-01 `atol = 0.01`; each kernel uses a tighter
//! value recorded in [`GEMM_ATOL`] / [`ELTWISE_ATOL`] / [`REDUCTION_ATOL`]
//! below (all well under the ceiling). The authoritative quality gate remains
//! M0-06's PyTorch-reference parity (NFR-QL-01); this harness only guards
//! scalar⇔SIMD self-consistency.
//!
//! Sizes deliberately include SIMD-lane multiples **and** ragged tails
//! (AVX2 = 8 lanes, NEON = 4 lanes) so the scalar-tail code paths are covered.
//!
//! New SIMD kernels register here by adding one case to the relevant test
//! (M0-08-T10..T15).

use vokra_backend_cpu::kernels;
use vokra_backend_cpu::{CpuFeatures, IsaPath, active_isa};

/// FP32 parity ceiling (NFR-QL-01); per-kernel tolerances stay under it.
const ATOL_CEILING: f32 = 0.01;
/// GEMM tolerance — larger because error grows with the K-reduction length.
const GEMM_ATOL: f32 = 1e-3;
const GEMM_RTOL: f32 = 1e-4;
/// GEMV tolerance: a per-row `k`-length dot product reordered from the scalar
/// oracle's sequential sum into four lane-partial FMA accumulators. The
/// absolute term is tight (1e-4, the design bound); [`GEMM_RTOL`] absorbs the
/// magnitude scaling on large-`k` rows, and all of it stays far under the
/// NFR-QL-01 ceiling.
const GEMV_ATOL: f32 = 1e-4;
/// Element-wise / activation tolerance.
const ELTWISE_ATOL: f32 = 1e-6;
/// Softmax / layer-norm tolerance (reductions + a division / rsqrt; the
/// larger of the two is layer-norm, whose `1/sqrt(var + eps)` can amplify
/// reduction-order differences on low-variance rows). Also covers softmax's
/// pass-2 `exp` once it is vectorized under `simd-transcendental` (the ULP
/// delta mostly cancels after row normalization).
const REDUCTION_ATOL: f32 = 1e-4;

/// Activation tolerance under `simd-transcendental`: the native vectorized
/// `exp` (M1-05-EXP) drifts from `std::f32::exp` by a few ULP, far under the
/// NFR-QL-01 ceiling. Unused (bit-identical) when the feature is off.
#[cfg(feature = "simd-transcendental")]
const ACTIVATION_ATOL: f32 = 1e-4;

/// The host's SIMD path (equals `Scalar` if the host has none).
fn simd_isa() -> IsaPath {
    CpuFeatures::detect().best_isa()
}

/// Minimal reproducible PRNG (xorshift64*), avoiding an external `rand`
/// dependency (the workspace keeps zero external deps).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // xorshift needs a non-zero state.
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform f32 in `[-1, 1)`.
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 bits
        let unit = bits as f32 / (1u32 << 24) as f32; // [0, 1)
        unit * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

#[track_caller]
fn assert_close(a: &[f32], b: &[f32], atol: f32, rtol: f32, ctx: &str) {
    assert!(
        atol <= ATOL_CEILING,
        "{ctx}: atol {atol} exceeds NFR-QL-01 ceiling"
    );
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let tol = atol + rtol * y.abs();
        assert!(
            (x - y).abs() <= tol,
            "{ctx}: index {i}: scalar={y}, simd={x}, |diff|={} > tol {tol}",
            (x - y).abs()
        );
    }
}

#[test]
fn gemm_scalar_matches_simd() {
    let mut rng = Rng::new(0x1234_5678);
    // (m, n, k) — n deliberately spans lane multiples and tails.
    let shapes = [
        (1, 1, 1),
        (2, 3, 4),
        (3, 8, 5),
        (4, 9, 7), // n = 9 → AVX2 tail
        (5, 4, 6), // n = 4 → NEON exact, AVX2 tail
        (8, 16, 32),
        (2, 17, 3), // n = 17 → both tails
        (7, 5, 11),
    ];
    for (m, n, k) in shapes {
        let a = rng.vec(m * k);
        let b = rng.vec(k * n);
        let bias = rng.vec(n);
        for use_bias in [false, true] {
            let bias_opt = use_bias.then_some(bias.as_slice());
            let mut out_ref = vec![0.0; m * n];
            let mut out_simd = vec![0.0; m * n];
            kernels::gemm_f32_on(IsaPath::Scalar, m, n, k, &a, &b, bias_opt, &mut out_ref).unwrap();
            kernels::gemm_f32_on(simd_isa(), m, n, k, &a, &b, bias_opt, &mut out_simd).unwrap();
            assert_close(
                &out_simd,
                &out_ref,
                GEMM_ATOL,
                GEMM_RTOL,
                &format!("gemm {m}x{n}x{k} bias={use_bias}"),
            );
        }
    }
}

/// Register-blocked GEMM microkernel (M1-08) vs the scalar oracle over shapes
/// that deliberately straddle the tile geometry. The NEON tile is `MR = 8` ×
/// `NR = 8` and the AVX2 tile is `MR = 6` × `NR = 16`, so these sizes span:
/// exact multiples of both `MR`s (pure main-path tiles), `m` values that are a
/// multiple of neither `MR` (row tail on both ISAs), and `n` values that force
/// the `NR` → 1-vector → scalar column-remainder cascade. Multi-tile sizes
/// exercise the `i += MR` / `j += NR` advance more than once, and the large-`k`
/// cases stress the accumulation the blocking reorders.
#[test]
fn blocked_gemm_matches_scalar_incl_ragged_tails() {
    let mut rng = Rng::new(0x0BB1_0CE5);
    // (m, n, k):
    let shapes = [
        // Multi-tile, fully ragged on both ISAs (row + all column remainders).
        (13, 11, 9),
        (25, 27, 17),
        (41, 43, 33),
        (50, 37, 24),
        // m a multiple of neither 8 nor 6; n spans NR + 1-vector + scalar tail.
        (17, 23, 16),
        (100, 19, 40),
        // Exact tile multiples (pure main path, no remainder) on each ISA.
        (16, 16, 32), // NEON: 2×2 tiles exact
        (48, 48, 10), // 48 = 6*8 = 8*6: exact rows/cols for both ISAs
        (24, 32, 8),  // AVX2: 4 row-tiles × 2 col-tiles exact
        // Wide-k accumulation with small tiles.
        (12, 12, 129),
        (7, 9, 200),
        // Single-row and single-column degenerate shapes (all tail, no tile).
        (1, 40, 17),
        (40, 1, 17),
        (5, 3, 6),
    ];
    for (m, n, k) in shapes {
        let a = rng.vec(m * k);
        let b = rng.vec(k * n);
        let bias = rng.vec(n);
        for use_bias in [false, true] {
            let bias_opt = use_bias.then_some(bias.as_slice());
            let mut out_ref = vec![0.0; m * n];
            let mut out_simd = vec![0.0; m * n];
            kernels::gemm_f32_on(IsaPath::Scalar, m, n, k, &a, &b, bias_opt, &mut out_ref).unwrap();
            kernels::gemm_f32_on(simd_isa(), m, n, k, &a, &b, bias_opt, &mut out_simd).unwrap();
            assert_close(
                &out_simd,
                &out_ref,
                GEMM_ATOL,
                GEMM_RTOL,
                &format!("blocked gemm {m}x{n}x{k} bias={use_bias}"),
            );
        }
    }
}

/// GEMV (matrix-vector, the tied-logits-head fast path) vs the scalar oracle.
/// `k` deliberately spans the kernel's reduction geometry: the 16-lane (NEON)
/// / 32-lane (AVX2) 4-accumulator inner loop, the 4-/8-wide single-accumulator
/// remainder, and the scalar `k % 4` / `k % 8` tail — plus the `k = 512` shape
/// of the real Whisper logits head (`token_emb[v, d] @ h[d]`, `d = 512`). `m`
/// covers the single-row degenerate case and larger row counts.
#[test]
fn gemv_scalar_matches_simd() {
    let mut rng = Rng::new(0x9E11_0AC7);
    // (m, k):
    let shapes = [
        (1, 1),     // degenerate: pure scalar tail
        (3, 4),     // NEON exact 4-wide, AVX2 scalar tail
        (5, 7),     // NEON 4 + 3-tail; AVX2 all-scalar tail
        (2, 8),     // AVX2 exact 8-wide; NEON 2×4
        (4, 15),    // both ISAs ragged (8+7 / 4·3+3)
        (8, 16),    // NEON exact 16 (one 4-acc iter); AVX2 8+8
        (2, 20),    // NEON 16 + 4-remainder (no scalar tail); AVX2 8+8+4
        (7, 31),    // AVX2 8·3 + 7-tail; NEON 4·7 + 3-tail
        (6, 32),    // AVX2 exact 32 (one 4-acc iter); NEON 8×4
        (3, 33),    // one full inner iter + 1 scalar (both ISAs)
        (2, 129),   // 32·4 + 1: multi inner iters + scalar tail
        (100, 200), // larger m; k = 32·6 + 8 (AVX2 8-remainder)
        (51865, 1), // v×1 extreme: every element is the scalar tail
        (64, 512),  // the real logits-head reduction width (d = 512)
    ];
    for (m, k) in shapes {
        let a = rng.vec(m * k);
        let x = rng.vec(k);
        let bias = rng.vec(m);
        for use_bias in [false, true] {
            let bias_opt = use_bias.then_some(bias.as_slice());
            let mut out_ref = vec![0.0; m];
            let mut out_simd = vec![0.0; m];
            kernels::gemv_f32_on(IsaPath::Scalar, m, k, &a, &x, bias_opt, &mut out_ref).unwrap();
            kernels::gemv_f32_on(simd_isa(), m, k, &a, &x, bias_opt, &mut out_simd).unwrap();
            assert_close(
                &out_simd,
                &out_ref,
                GEMV_ATOL,
                GEMM_RTOL,
                &format!("gemv {m}x{k} bias={use_bias}"),
            );
        }
    }
}

/// GEMV must equal the general GEMM's `n = 1` column on identical data — the
/// numeric equivalence the decoder's tied-logits-head routing relies on (the
/// scalar oracle is bit-identical; the SIMD paths agree within [`GEMV_ATOL`]).
#[test]
fn gemv_matches_gemm_n1_column() {
    let mut rng = Rng::new(0x10A1_7ED5);
    for (m, k) in [(1usize, 512usize), (17, 63), (2, 8), (129, 40)] {
        let a = rng.vec(m * k);
        let x = rng.vec(k); // the single b-column of an [k, 1] gemm b
        let mut gemm_out = vec![0.0; m];
        let mut gemv_out = vec![0.0; m];
        // gemm(m, n=1, k): b is [k, 1] = x, out is [m, 1].
        kernels::gemm_f32_on(simd_isa(), m, 1, k, &a, &x, None, &mut gemm_out).unwrap();
        kernels::gemv_f32_on(simd_isa(), m, k, &a, &x, None, &mut gemv_out).unwrap();
        assert_close(
            &gemv_out,
            &gemm_out,
            GEMV_ATOL,
            GEMM_RTOL,
            &format!("gemv-vs-gemm-n1 {m}x{k}"),
        );
    }
}

/// The production `gemm_f32` (row-parallel over the `pool` when `feature =
/// parallel` and the host is multi-core) must be **bit-for-bit identical** to
/// the single-thread `gemm_f32_on(active_isa)` — splitting the output rows only
/// reorders *which thread* runs a row, never the per-element FMA chain. Shapes
/// straddle the pool's size threshold (`m*n*k ≥ GEMM_MIN_MACS`) and the row
/// chunking (multi-task + ragged final chunk); the sub-threshold shape runs
/// inline and is trivially identical. On a single-core host the pool is absent,
/// so this degenerates to the same inline call on both sides (still exact).
#[test]
fn parallel_gemm_bit_identical_to_single_thread() {
    let mut rng = Rng::new(0x9A2D_11FE);
    let isa = active_isa();
    // (m, n, k): large enough to trigger the pool, plus one tiny inline case.
    let shapes = [
        (1500, 64, 80), // ~7.7M MACs, encoder-height rows → many tasks
        (300, 128, 64), // ~2.4M, ragged row chunks
        (1001, 33, 41), // ~1.35M, ragged rows AND columns (kernel tails)
        (512, 96, 64),  // ~3.1M
        (3, 4, 5),      // sub-threshold: inline, trivially identical
    ];
    for (m, n, k) in shapes {
        let a = rng.vec(m * k);
        let b = rng.vec(k * n);
        let bias = rng.vec(n);
        for use_bias in [false, true] {
            let bias_opt = use_bias.then_some(bias.as_slice());
            let mut prod = vec![0.0; m * n];
            let mut single = vec![0.0; m * n];
            // Production path (pool-routed when parallel + multi-core).
            kernels::gemm_f32(m, n, k, &a, &b, bias_opt, &mut prod).unwrap();
            // Single-thread reference on the same ISA (never the pool).
            kernels::gemm_f32_on(isa, m, n, k, &a, &b, bias_opt, &mut single).unwrap();
            assert_eq!(
                prod, single,
                "gemm {m}x{n}x{k} bias={use_bias}: threaded output must be bit-identical to single-thread"
            );
        }
    }
}

/// Same bit-identical guarantee for the tied-logits-head `gemv_f32` fast path
/// (per-row bias is sliced to each task's rows). `m` spans the pool threshold
/// (`m*k ≥ GEMV_MIN_MACS`) and multiple ragged row chunks.
#[test]
fn parallel_gemv_bit_identical_to_single_thread() {
    let mut rng = Rng::new(0x9A2D_22FE);
    let isa = active_isa();
    // (m, k): logits-head-shaped row counts + a sub-threshold inline case.
    let shapes = [
        (60000, 32), // ~1.9M, many tasks
        (4096, 300), // ~1.2M
        (2000, 512), // ~1M, real logits reduction width (d = 512)
        (1001, 257), // ragged rows + kernel k-tail
        (3, 7),      // sub-threshold: inline, trivially identical
    ];
    for (m, k) in shapes {
        let a = rng.vec(m * k);
        let x = rng.vec(k);
        let bias = rng.vec(m);
        for use_bias in [false, true] {
            let bias_opt = use_bias.then_some(bias.as_slice());
            let mut prod = vec![0.0; m];
            let mut single = vec![0.0; m];
            kernels::gemv_f32(m, k, &a, &x, bias_opt, &mut prod).unwrap();
            kernels::gemv_f32_on(isa, m, k, &a, &x, bias_opt, &mut single).unwrap();
            assert_eq!(
                prod, single,
                "gemv {m}x{k} bias={use_bias}: threaded output must be bit-identical to single-thread"
            );
        }
    }
}

#[test]
fn elementwise_scalar_matches_simd() {
    let mut rng = Rng::new(0xC0FF_EE00);
    let lens = [1usize, 3, 4, 7, 8, 9, 15, 16, 17, 31, 64, 100];
    for len in lens {
        let a = rng.vec(len);
        let b = rng.vec(len);
        let mut r_add = vec![0.0; len];
        let mut s_add = vec![0.0; len];
        kernels::add_f32_on(IsaPath::Scalar, &a, &b, &mut r_add).unwrap();
        kernels::add_f32_on(simd_isa(), &a, &b, &mut s_add).unwrap();
        assert_close(&s_add, &r_add, ELTWISE_ATOL, 0.0, &format!("add len={len}"));

        let mut r_mul = vec![0.0; len];
        let mut s_mul = vec![0.0; len];
        kernels::mul_f32_on(IsaPath::Scalar, &a, &b, &mut r_mul).unwrap();
        kernels::mul_f32_on(simd_isa(), &a, &b, &mut s_mul).unwrap();
        assert_close(&s_mul, &r_mul, ELTWISE_ATOL, 0.0, &format!("mul len={len}"));

        let mut r_relu = vec![0.0; len];
        let mut s_relu = vec![0.0; len];
        kernels::relu_f32_on(IsaPath::Scalar, &a, &mut r_relu).unwrap();
        kernels::relu_f32_on(simd_isa(), &a, &mut s_relu).unwrap();
        assert_close(
            &s_relu,
            &r_relu,
            ELTWISE_ATOL,
            0.0,
            &format!("relu len={len}"),
        );
    }
}

/// Activation SIMD-vs-scalar comparison. Without `simd-transcendental` the
/// SIMD paths are scalar-backed, so they must be **bit-for-bit** identical;
/// with the feature they use the native vectorized `exp` (M1-05-EXP), so they
/// match within the bounded [`ACTIVATION_ATOL`] ULP delta.
#[track_caller]
fn assert_activation(simd: &[f32], scalar: &[f32], ctx: &str) {
    #[cfg(not(feature = "simd-transcendental"))]
    assert_eq!(simd, scalar, "{ctx} must be bit-identical (scalar-backed)");
    #[cfg(feature = "simd-transcendental")]
    assert_close(simd, scalar, ACTIVATION_ATOL, 0.0, ctx);
}

#[test]
fn activations_scalar_matches_simd() {
    let mut rng = Rng::new(0xABCD_1234);
    let lens = [1usize, 7, 8, 9, 16, 33];
    for len in lens {
        let x = rng.vec(len);
        let mut r = vec![0.0; len];
        let mut s = vec![0.0; len];
        kernels::sigmoid_f32_on(IsaPath::Scalar, &x, &mut r).unwrap();
        kernels::sigmoid_f32_on(simd_isa(), &x, &mut s).unwrap();
        assert_activation(&s, &r, &format!("sigmoid len={len}"));

        kernels::tanh_f32_on(IsaPath::Scalar, &x, &mut r).unwrap();
        kernels::tanh_f32_on(simd_isa(), &x, &mut s).unwrap();
        assert_activation(&s, &r, &format!("tanh len={len}"));

        kernels::gelu_f32_on(IsaPath::Scalar, &x, &mut r).unwrap();
        kernels::gelu_f32_on(simd_isa(), &x, &mut s).unwrap();
        assert_activation(&s, &r, &format!("gelu len={len}"));
    }
}

/// With the vectorized `exp`, the activations must still track the scalar
/// oracle through **saturation** (well outside the `[-1, 1)` fuzzed range):
/// `sigmoid → {0, 1}`, `tanh → {-1, 1}`, `gelu` grows ~linearly. This is where
/// the `exp` domain clamp in `kernels::vexp` matters.
#[cfg(feature = "simd-transcendental")]
#[test]
fn vectorized_activations_saturate_like_scalar() {
    let xs: Vec<f32> = vec![
        -40.0, -20.0, -8.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 8.0, 20.0, 40.0,
    ];
    let n = xs.len();
    let mut r = vec![0.0; n];
    let mut s = vec![0.0; n];

    kernels::sigmoid_f32_on(IsaPath::Scalar, &xs, &mut r).unwrap();
    kernels::sigmoid_f32_on(simd_isa(), &xs, &mut s).unwrap();
    assert_close(&s, &r, ACTIVATION_ATOL, 0.0, "sigmoid wide-range");

    kernels::tanh_f32_on(IsaPath::Scalar, &xs, &mut r).unwrap();
    kernels::tanh_f32_on(simd_isa(), &xs, &mut s).unwrap();
    assert_close(&s, &r, ACTIVATION_ATOL, 0.0, "tanh wide-range");

    kernels::gelu_f32_on(IsaPath::Scalar, &xs, &mut r).unwrap();
    kernels::gelu_f32_on(simd_isa(), &xs, &mut s).unwrap();
    // gelu(x) ~ x for large x, so allow a relative term on the linear tail.
    assert_close(&s, &r, ACTIVATION_ATOL, 1e-4, "gelu wide-range");
}

#[test]
fn softmax_scalar_matches_simd() {
    let mut rng = Rng::new(0x5EED_0009);
    let cases = [
        (1usize, 1usize),
        (2, 4),
        (3, 7),
        (2, 8),
        (4, 9),
        (1, 16),
        (3, 33),
    ];
    for (rows, cols) in cases {
        let input = rng.vec(rows * cols);
        let mut r = vec![0.0; rows * cols];
        let mut s = vec![0.0; rows * cols];
        kernels::softmax_f32_on(IsaPath::Scalar, &input, &mut r, rows, cols).unwrap();
        kernels::softmax_f32_on(simd_isa(), &input, &mut s, rows, cols).unwrap();
        assert_close(
            &s,
            &r,
            REDUCTION_ATOL,
            0.0,
            &format!("softmax {rows}x{cols}"),
        );
    }
}

#[test]
fn layer_norm_scalar_matches_simd() {
    let mut rng = Rng::new(0x1A1B_1C1D);
    let cases = [(1usize, 4usize), (2, 7), (3, 8), (2, 9), (4, 16), (1, 33)];
    for (rows, cols) in cases {
        let input = rng.vec(rows * cols);
        let gamma = rng.vec(cols);
        let beta = rng.vec(cols);
        let mut r = vec![0.0; rows * cols];
        let mut s = vec![0.0; rows * cols];
        let eps = kernels::LAYER_NORM_DEFAULT_EPS;
        kernels::layer_norm_f32_on(
            IsaPath::Scalar,
            &input,
            &mut r,
            rows,
            cols,
            &gamma,
            &beta,
            eps,
        )
        .unwrap();
        kernels::layer_norm_f32_on(simd_isa(), &input, &mut s, rows, cols, &gamma, &beta, eps)
            .unwrap();
        assert_close(
            &s,
            &r,
            REDUCTION_ATOL,
            0.0,
            &format!("layer_norm {rows}x{cols}"),
        );
    }
}

#[test]
fn conv1d_rides_simd_gemm() {
    // conv1d has no dedicated SIMD kernel; it must match between scalar-GEMM
    // and SIMD-GEMM lowering (M0-08-T12/T15).
    let mut rng = Rng::new(0xC01D_1D1D);
    // (in_ch, in_len, out_ch, kernel, stride, padding)
    let cases = [
        (1, 16, 1, 3, 1, 0),
        (2, 20, 3, 5, 2, 2),
        (3, 9, 4, 3, 1, 1),
        (1, 50, 8, 7, 3, 3), // out_ch = 8 → AVX2 exact row, NEON tail
    ];
    for (in_ch, in_len, out_ch, kernel, stride, padding) in cases {
        let input = rng.vec(in_ch * in_len);
        let weight = rng.vec(out_ch * in_ch * kernel);
        let bias = rng.vec(out_ch);
        let padded = in_len + 2 * padding;
        let out_len = (padded - kernel) / stride + 1;
        let mut r = vec![0.0; out_ch * out_len];
        let mut s = vec![0.0; out_ch * out_len];
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
            padding,
            &mut r,
        )
        .unwrap();
        kernels::conv1d_f32_on(
            simd_isa(),
            &input,
            in_ch,
            in_len,
            &weight,
            out_ch,
            kernel,
            Some(&bias),
            stride,
            padding,
            &mut s,
        )
        .unwrap();
        assert_close(
            &s,
            &r,
            GEMM_ATOL,
            GEMM_RTOL,
            &format!("conv1d ic{in_ch} il{in_len} oc{out_ch} k{kernel} s{stride} p{padding}"),
        );
    }
}

#[test]
fn forcing_unavailable_path_is_explicit_error() {
    // Whichever SIMD path the host lacks must be an explicit error when
    // forced, never a silent switch (FR-EX-08 principle).
    let feats = CpuFeatures::detect();
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [1.0f32, 1.0, 1.0, 1.0];
    let mut out = [0.0f32; 4];
    for isa in [IsaPath::Avx2, IsaPath::Neon] {
        if !feats.supports(isa) {
            assert!(kernels::add_f32_on(isa, &a, &b, &mut out).is_err());
        }
    }
}
