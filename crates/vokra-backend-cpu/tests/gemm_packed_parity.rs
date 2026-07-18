//! M5-14 Wave-1 packed-GEMM parity (T05/T08/T09/T10).
//!
//! The production `gemm_f32` now routes through the packed / cache-blocked
//! driver (`kernels::gemm_driver`): B panels are packed into contiguous
//! `NR`-column strips, A tiles into `MR`-row strips, the k-loop is split into
//! `KC` blocks accumulated **in ascending k order with a single accumulator
//! per output element**, and `m == 1` calls take a dedicated register-blocked
//! row path. All of that is a pure data-layout / scheduling change: every
//! output element is still the same bias-seeded chain over `l = 0..k` as the
//! pre-existing register-blocked kernel — fused FMA in the vector column
//! region, plain mul+add in the scalar column tail — so the packed path must
//! be **BIT-IDENTICAL** to the legacy kernel (reached via `gemm_f32_on`,
//! which pins the forced-ISA single-thread path and never the driver).
//!
//! That bit-identity is the parity red-line for M5-14 Wave 1: committed
//! whisper / kokoro / piper parity fixtures cannot shift if `gemm_f32`'s
//! bits do not shift. The tests here therefore assert exact `to_bits`
//! equality (not a tolerance) between the driver and the legacy kernel, over
//! a shape grid that crosses every blocking boundary (MC / KC / NC, MR row
//! strips, NR column strips, the 4-wide NEON strip, scalar column tails) and
//! the routing thresholds, plus a scalar-oracle tolerance check mirroring
//! `tests/differential.rs`.

use vokra_backend_cpu::gemm_test_probe as probe;
use vokra_backend_cpu::kernels;
use vokra_backend_cpu::{CpuFeatures, active_isa};

/// Minimal xorshift64* PRNG (no `rand`; NFR-DS-02), mirroring
/// `tests/differential.rs`.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

#[track_caller]
fn assert_bits_eq(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{ctx}: bit mismatch at flat index {i}: got {g} want {w}"
        );
    }
}

/// The adversarial shape grid. Covers:
/// - m == 1 (the axpy fast path) with vector-region + scalar-tail columns;
/// - m below / at / above the MR row-strip height (padded row strips);
/// - n crossing the NR strip width, the NEON 4-wide strip window
///   (`n % 8 >= 4`), scalar column tails (`n % 4 != 0`), and the NC slab
///   boundary;
/// - k == 0 (bias passthrough), tiny k, and k crossing the KC block boundary
///   (multi-block accumulation through the C buffer);
/// - shapes straddling the routing thresholds so both the packed and the
///   legacy route are exercised and compared against the same oracle.
fn shape_grid() -> Vec<(usize, usize, usize)> {
    let mc = probe::MC;
    let kc = probe::KC;
    let nc = probe::NC;
    let mr = probe::PACK_MR;
    vec![
        // -- degenerate / tiny (legacy or m1 route) --
        (1, 1, 1),
        (1, 1, 0),
        (2, 3, 0),
        (3, 5, 4),
        (7, 16, 32),
        (8, 8, 8),
        (1, 7, 5),
        (1, 8, 8),
        (1, 17, 9),
        (1, 260, 96),
        (1, 771, 129), // vector region + 3-col scalar tail, k odd
        // -- m == 1, large enough to cross the m1 parallel threshold --
        (1, 2048, 1024),
        // -- packed route: row strips exact / padded --
        (mr, 64, 64),
        (mr + 1, 64, 64),
        (2 * mr + 5, 64, 64),
        // -- column tails: 8-strip + 4-strip + scalar (NEON), scalar (AVX2),
        //    masked (AVX-512) --
        (17, 23, 40),
        (37, 177, 96), // n % 8 == 1: 1 scalar tail col (the piper-decoder shape family)
        (33, 44, 128), // n % 8 == 4: exercises the 4-wide strip
        (65, 132, 200), // n % 8 == 4 again, k > 128
        (16, 19, 64),  // n % 16 == 3: AVX-512 masked tail region
        // -- KC boundary: multi-block accumulation through C --
        (mr + 3, 64, kc + 9),
        (mc + 3, 40, 2 * kc + 1),
        // -- NC boundary: multi-slab packing --
        (mr + 1, nc + 12, 33),
        // -- MC boundary: multiple parallel tasks --
        (mc + mr + 3, 132, kc + 9),
        (2 * mc + 7, 264, 96),
        // -- thin-m over the pool gate: the column-strip parallel mode
        //    (m <= MC, m*n*k >= 1M), incl. an NC-crossing n with scalar tail --
        (32, 2048, 96),
        (mc - mr - 3, nc + 19, 128),
        // -- attention-shaped (n or k small, the whisper per-head family) --
        (150, 64, 150),
        (150, 150, 64),
    ]
}

/// Production `gemm_f32` (packed driver, pool-eligible) must be bit-identical
/// to the forced-ISA legacy kernel (`gemm_f32_on(active_isa)`, single-thread,
/// never routed through the driver) — the packed path is a pure reordering of
/// data movement, never of per-element arithmetic.
#[test]
fn packed_driver_bit_identical_to_legacy_kernel() {
    let isa = active_isa();
    let mut rng = Rng::new(0x5A14_0001);
    for (m, n, k) in shape_grid() {
        let a = rng.vec(m * k);
        let b = rng.vec(k * n);
        let bias = rng.vec(n);
        for bias_opt in [None, Some(bias.as_slice())] {
            let mut legacy = vec![f32::NAN; m * n];
            kernels::gemm_f32_on(isa, m, n, k, &a, &b, bias_opt, &mut legacy).unwrap();
            let mut driver = vec![f32::NAN; m * n];
            kernels::gemm_f32(m, n, k, &a, &b, bias_opt, &mut driver).unwrap();
            assert_bits_eq(
                &driver,
                &legacy,
                &format!("gemm {m}x{n}x{k} bias={}", bias_opt.is_some()),
            );
        }
    }
}

/// Scalar-oracle differential for the same grid (mirrors
/// `tests/differential.rs` tolerances): the driver must stay within the
/// existing FMA-reordering bound of the scalar reference, independent of the
/// bit-identity pin above.
#[test]
fn packed_driver_matches_scalar_oracle() {
    const GEMM_ATOL: f32 = 1e-3;
    const GEMM_RTOL: f32 = 1e-4;
    let mut rng = Rng::new(0x5A14_0002);
    for (m, n, k) in shape_grid() {
        let a = rng.vec(m * k);
        let b = rng.vec(k * n);
        let bias = rng.vec(n);
        for bias_opt in [None, Some(bias.as_slice())] {
            let mut oracle = vec![f32::NAN; m * n];
            kernels::gemm_f32_on(
                vokra_backend_cpu::IsaPath::Scalar,
                m,
                n,
                k,
                &a,
                &b,
                bias_opt,
                &mut oracle,
            )
            .unwrap();
            let mut driver = vec![f32::NAN; m * n];
            kernels::gemm_f32(m, n, k, &a, &b, bias_opt, &mut driver).unwrap();
            for (i, (&d, &o)) in driver.iter().zip(&oracle).enumerate() {
                let tol = GEMM_ATOL + GEMM_RTOL * o.abs();
                assert!(
                    (d - o).abs() <= tol,
                    "gemm {m}x{n}x{k} bias={} flat {i}: driver {d} vs scalar {o}",
                    bias_opt.is_some()
                );
            }
        }
    }
}

/// k == 0 writes exactly the bias (or zeros) through every route — the packed
/// driver must not leave the output unwritten when the k-block loop is empty.
#[test]
fn k_zero_writes_bias_or_zeros() {
    let mut rng = Rng::new(0x5A14_0003);
    for (m, n) in [(1usize, 9usize), (5, 8), (16, 132)] {
        let bias = rng.vec(n);
        let mut out = vec![f32::NAN; m * n];
        kernels::gemm_f32(m, n, 0, &[], &[], Some(&bias), &mut out).unwrap();
        for r in 0..m {
            assert_bits_eq(
                &out[r * n..(r + 1) * n],
                &bias,
                &format!("k=0 bias row {r}"),
            );
        }
        let mut out = vec![f32::NAN; m * n];
        kernels::gemm_f32(m, n, 0, &[], &[], None, &mut out).unwrap();
        assert!(out.iter().all(|v| *v == 0.0), "k=0 no-bias must zero-fill");
    }
}

/// Run-to-run determinism under the thread pool: the same pooled call twice
/// must produce identical bits (disjoint output tiles + fixed per-task
/// accumulation order make the result independent of thread scheduling).
#[test]
fn packed_driver_is_run_to_run_deterministic() {
    let (m, n, k) = (2 * probe::MC + 7, 264, 96); // above the pool MACs gate
    let mut rng = Rng::new(0x5A14_0004);
    let a = rng.vec(m * k);
    let b = rng.vec(k * n);
    let bias = rng.vec(n);
    let mut first = vec![0.0f32; m * n];
    kernels::gemm_f32(m, n, k, &a, &b, Some(&bias), &mut first).unwrap();
    for round in 0..3 {
        let mut again = vec![f32::NAN; m * n];
        kernels::gemm_f32(m, n, k, &a, &b, Some(&bias), &mut again).unwrap();
        assert_bits_eq(&again, &first, &format!("determinism round {round}"));
    }
}

/// Routing sanity (anti-fake-green): the Wave-0 hot shapes must actually take
/// the packed path, the decoder-step shape must take the m1 path, and tiny
/// shapes must stay on the legacy kernel. Guards against a silent "driver
/// never fires" regression that would leave every test above vacuously green
/// on the legacy path.
#[test]
fn routing_thresholds_route_the_hot_shapes() {
    // Wave-0 encoder shapes (whisper small fc1 / medium fc1 pathological).
    assert!(probe::would_use_packed(1500, 3072, 768));
    assert!(probe::would_use_packed(1500, 4096, 1024));
    // Whisper per-head attention shapes.
    assert!(probe::would_use_packed(1500, 64, 1500));
    assert!(probe::would_use_packed(1500, 1500, 64));
    // CAM++ thin-m / huge-n conv-as-GEMM shape.
    assert!(probe::would_use_packed(32, 43920, 288));
    // Tiny shapes stay on the legacy kernel.
    assert!(!probe::would_use_packed(2, 4, 4));
    assert!(!probe::would_use_packed(4, 8, 8));
    // m == 1 routes to the axpy row path wherever the ISA provides it, and
    // never to the packed path.
    assert!(!probe::would_use_packed(1, 768, 768));
    // On hosts whose active ISA ships the packed micro-kernels (NEON / AVX2 /
    // AVX-512), the m1 + packed routes must actually be live.
    let has_simd_gemm = !matches!(
        CpuFeatures::detect().best_isa(),
        vokra_backend_cpu::IsaPath::Scalar
    );
    if has_simd_gemm && cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
        assert!(
            probe::active_gemm_has_packed(),
            "active ISA should provide packed micro-kernels"
        );
        assert!(
            probe::active_gemm_has_m1(),
            "active ISA should provide the m1 row kernel"
        );
    }
}

/// The m == 1 fast path must be bit-identical to the legacy kernel's m == 1
/// row (same fma chain in the vector region, same mul+add scalar tail),
/// including under the column-split parallel path for large n*k.
#[test]
fn m1_row_path_bit_identical_to_legacy_kernel() {
    let isa = active_isa();
    let mut rng = Rng::new(0x5A14_0005);
    for (n, k) in [
        (1usize, 1usize),
        (3, 7),
        (4, 16),
        (12, 33),
        (768, 768),
        (771, 129),
        (3072, 768),
        (2048, 1024), // crosses the m1 parallel threshold
    ] {
        let a = rng.vec(k);
        let b = rng.vec(k * n);
        let bias = rng.vec(n);
        for bias_opt in [None, Some(bias.as_slice())] {
            let mut legacy = vec![f32::NAN; n];
            kernels::gemm_f32_on(isa, 1, n, k, &a, &b, bias_opt, &mut legacy).unwrap();
            let mut driver = vec![f32::NAN; n];
            kernels::gemm_f32(1, n, k, &a, &b, bias_opt, &mut driver).unwrap();
            assert_bits_eq(
                &driver,
                &legacy,
                &format!("m1 gemm 1x{n}x{k} bias={}", bias_opt.is_some()),
            );
        }
    }
}
