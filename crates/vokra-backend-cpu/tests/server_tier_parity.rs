//! M4-17-T19: scalar-oracle parity for the server-tier kernels, forced per
//! ISA path (the `rvv_dispatch_parity.rs` pattern).
//!
//! Every check is **runtime-detect gated**: a path this host cannot run is
//! skipped (with an eprintln so CI logs show the skip), never fabricated.
//! On this Apple M1 dev machine the NEON fp16 / dotprod legs and the NEON
//! K-quants dequant run against real silicon; the AVX-512 family legs run
//! on AVX-512-capable x86-64 runners (Ice-Lake/Zen4-class GitHub hosts) or
//! on the owner's cloud VM (M4-17-T23); i8mm / bf16 run on owner silicon
//! (M4-17-T24).
//!
//! Tolerances are per-kernel and transcribed from the tickets (M4-17-T07..
//! T17; ADR M4-17 kernel table): f32 GEMM 1e-3+rtol / GEMV 1e-4 /
//! elementwise bit-exact / softmax+layer_norm 1e-4 / INT8 bit-identical to
//! the scalar-int8 reference + input-derived quantization bound vs f32 /
//! fp16 ±2 ulp vs the structural emulation / bf16 architectural band.

use vokra_backend_cpu::kernels::{
    self, KQuantDtype, add_f32_on, gemm_f32_on, gemv_f32_on, layer_norm_f32_on, mul_f32_on,
    relu_f32_on, softmax_f32_on,
};
use vokra_backend_cpu::{CpuFeatures, IsaPath};

/// FP32 parity ceiling (NFR-QL-01); per-kernel tolerances stay under it.
const GEMM_ATOL: f32 = 1e-3;
const GEMM_RTOL: f32 = 1e-4;
const GEMV_ATOL: f32 = 1e-4;
const REDUCTION_ATOL: f32 = 1e-4;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
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
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as u32) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next_u64() >> 32) as u8).collect()
    }
}

#[track_caller]
fn assert_close(got: &[f32], want: &[f32], atol: f32, rtol: f32, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = atol + rtol * w.abs();
        assert!(
            (g - w).abs() <= tol,
            "{ctx}: index {i}: scalar={w}, simd={g}, |diff|={} > tol {tol}",
            (g - w).abs()
        );
    }
}

/// K-quant payload with pinned-small f16 scales (finite magnitudes; every
/// other byte pattern is a valid payload for all three formats).
fn random_blocks(rng: &mut Rng, dtype: KQuantDtype, nb: usize) -> Vec<u8> {
    let mut bytes = rng.bytes(nb * dtype.block_bytes());
    for b in 0..nb {
        let base = b * dtype.block_bytes();
        match dtype {
            KQuantDtype::Q4K | KQuantDtype::Q5K => {
                bytes[base + 1] = 0x2C;
                bytes[base + 3] = 0x24;
            }
            KQuantDtype::Q6K => {
                bytes[base + 209] = 0x2C;
            }
        }
    }
    bytes
}

// ---------------------------------------------------------------------
// AVX-512 f32 tier (T07-T09): forced-path vs scalar, runtime-detect gated.
// ---------------------------------------------------------------------

#[test]
fn avx512_f32_kernels_match_scalar_oracle() {
    let feats = CpuFeatures::detect();
    if !feats.supports(IsaPath::Avx512) {
        eprintln!("skip: AVX-512 f32 bundle not available on this host (owner cloud VM leg, T23)");
        return;
    }
    let isa = IsaPath::Avx512;
    let mut rng = Rng::new(0xA512_A512);

    // GEMM shapes spanning the 32/16-wide paths and the masked (<16) tail,
    // plus the MR row tail.
    for &(m, n, k) in &[
        (1usize, 1usize, 1usize),
        (6, 32, 9),   // full MR block, two full 16-lane vectors
        (7, 33, 5),   // row tail + masked col tail (rem = 1)
        (13, 47, 21), // row tail + 16-wide + masked (rem = 15)
    ] {
        let a = rng.vec(m * k);
        let b = rng.vec(k * n);
        let bias = rng.vec(n);
        let mut want = vec![0.0f32; m * n];
        gemm_f32_on(IsaPath::Scalar, m, n, k, &a, &b, Some(&bias), &mut want).unwrap();
        let mut got = vec![0.0f32; m * n];
        gemm_f32_on(isa, m, n, k, &a, &b, Some(&bias), &mut got).unwrap();
        assert_close(
            &got,
            &want,
            GEMM_ATOL,
            GEMM_RTOL,
            &format!("gemm {m}x{n}x{k}"),
        );
    }

    // GEMV spanning the 64-wide unroll, 16-wide remainder and scalar tail.
    for &(m, k) in &[(3usize, 16usize), (5, 67), (2, 130)] {
        let a = rng.vec(m * k);
        let x = rng.vec(k);
        let bias = rng.vec(m);
        let mut want = vec![0.0f32; m];
        gemv_f32_on(IsaPath::Scalar, m, k, &a, &x, Some(&bias), &mut want).unwrap();
        let mut got = vec![0.0f32; m];
        gemv_f32_on(isa, m, k, &a, &x, Some(&bias), &mut got).unwrap();
        assert_close(&got, &want, GEMV_ATOL, GEMM_RTOL, &format!("gemv {m}x{k}"));
    }

    // Elementwise: bit-exact (per-lane ops, masked tail included).
    let a = rng.vec(41);
    let b = rng.vec(41);
    let mut want = vec![0.0f32; 41];
    let mut got = vec![0.0f32; 41];
    add_f32_on(IsaPath::Scalar, &a, &b, &mut want).unwrap();
    add_f32_on(isa, &a, &b, &mut got).unwrap();
    assert_eq!(got, want, "avx512 add must be bit-exact");
    mul_f32_on(IsaPath::Scalar, &a, &b, &mut want).unwrap();
    mul_f32_on(isa, &a, &b, &mut got).unwrap();
    assert_eq!(got, want, "avx512 mul must be bit-exact");
    relu_f32_on(IsaPath::Scalar, &a, &mut want).unwrap();
    relu_f32_on(isa, &a, &mut got).unwrap();
    assert_eq!(got, want, "avx512 relu must be bit-exact");

    // Reductions: softmax 3x17 / layer_norm 2x29 (tails on both).
    let sm = rng.vec(3 * 17);
    let mut want = vec![0.0f32; sm.len()];
    let mut got = vec![0.0f32; sm.len()];
    softmax_f32_on(IsaPath::Scalar, &sm, &mut want, 3, 17).unwrap();
    softmax_f32_on(isa, &sm, &mut got, 3, 17).unwrap();
    assert_close(&got, &want, REDUCTION_ATOL, 0.0, "avx512 softmax");

    let ln = rng.vec(2 * 29);
    let gamma = rng.vec(29);
    let beta = rng.vec(29);
    let mut want = vec![0.0f32; ln.len()];
    let mut got = vec![0.0f32; ln.len()];
    layer_norm_f32_on(IsaPath::Scalar, &ln, &mut want, 2, 29, &gamma, &beta, 1e-5).unwrap();
    layer_norm_f32_on(isa, &ln, &mut got, 2, 29, &gamma, &beta, 1e-5).unwrap();
    assert_close(&got, &want, REDUCTION_ATOL, 0.0, "avx512 layer_norm");
}

// ---------------------------------------------------------------------
// K-quants dequant fusion bit-identity (T12), per host-supported family.
// ---------------------------------------------------------------------

#[test]
fn kquant_dequant_bit_identical_on_every_supported_path() {
    let feats = CpuFeatures::detect();
    let mut rng = Rng::new(0x0DE0_0DE0);
    let mut checked = 0;
    for isa in IsaPath::ALL_SIMD {
        if !feats.supports(isa) || matches!(isa, IsaPath::Rvv | IsaPath::WasmSimd128) {
            continue;
        }
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q5K, KQuantDtype::Q6K] {
            let bytes = random_blocks(&mut rng, dtype, 4);
            let want = kernels::kquant_dequant_on(IsaPath::Scalar, dtype, &bytes, 1024).unwrap();
            let got = kernels::kquant_dequant_on(isa, dtype, &bytes, 1024).unwrap();
            assert_eq!(
                got, want,
                "{dtype:?} dequant on {isa} must be bit-identical to the core reference (atol = 0.0)"
            );
        }
        checked += 1;
    }
    if checked == 0 {
        eprintln!("skip: no SIMD dequant family on this host (scalar-only)");
    }
}

// ---------------------------------------------------------------------
// INT8 tiers (T10/T13/T15): bit-identity across paths + honest quant band.
// ---------------------------------------------------------------------

#[test]
fn int8_gemv_paths_are_bit_identical_and_track_f32_within_derived_bound() {
    let feats = CpuFeatures::detect();
    let mut rng = Rng::new(0x1478_1478);
    for dtype in [KQuantDtype::Q4K, KQuantDtype::Q5K, KQuantDtype::Q6K] {
        let (m, k) = (4usize, 768usize);
        let nb = k / 256;
        let w: Vec<u8> = (0..m)
            .flat_map(|_| random_blocks(&mut rng, dtype, nb))
            .collect();
        let x = rng.vec(k);

        let mut reference = vec![0.0f32; m];
        kernels::kquant_gemv_i8_on(IsaPath::Scalar, dtype, m, k, &w, &x, &mut reference).unwrap();

        // (a) Every host-supported SIMD INT8 path is bit-identical to the
        // scalar-int8 reference (exact integer sums + shared combine).
        let mut simd_checked = 0;
        for isa in [
            IsaPath::Avx512Vnni,
            IsaPath::AvxVnni256,
            IsaPath::NeonDotprod,
        ] {
            if !feats.supports(isa) {
                continue;
            }
            let mut got = vec![0.0f32; m];
            kernels::kquant_gemv_i8_on(isa, dtype, m, k, &w, &x, &mut got).unwrap();
            assert_eq!(
                got, reference,
                "{dtype:?} INT8 GEMV on {isa} vs scalar-int8"
            );
            simd_checked += 1;
        }
        if simd_checked == 0 {
            eprintln!("skip: no SIMD INT8 tier on this host for {dtype:?} (scalar-int8 leg only)");
        }

        // (b) The scalar-int8 reference tracks the f32 dequant GEMV within
        // 2x the input-derived quantization bound (honest atol — the bound
        // is computed from the actual weights/activations, not invented).
        let row_bytes = nb * dtype.block_bytes();
        for i in 0..m {
            let row = &w[i * row_bytes..(i + 1) * row_bytes];
            let wf = kernels::kquant_dequant_on(IsaPath::Scalar, dtype, row, k).unwrap();
            let want: f32 = wf.iter().zip(&x).map(|(&a, &b)| a * b).sum();
            let bound = kernels::int8_error_bound(dtype, row, &x).max(1e-6);
            assert!(
                (reference[i] - want).abs() <= 2.0 * bound,
                "{dtype:?} row {i}: int8 {} vs f32 {} exceeds 2x derived bound {bound}",
                reference[i],
                want
            );
        }
    }
}

#[test]
fn int8_gemv2_tile_path_matches_column_path() {
    let feats = CpuFeatures::detect();
    let mut rng = Rng::new(0x2222_8888);
    let dtype = KQuantDtype::Q4K;
    let (m, k) = (5usize, 512usize); // odd m exercises the tile row tail
    let nb = k / 256;
    let w: Vec<u8> = (0..m)
        .flat_map(|_| random_blocks(&mut rng, dtype, nb))
        .collect();
    let x2 = rng.vec(2 * k);

    let mut want = vec![0.0f32; 2 * m];
    kernels::kquant_gemv2_i8_on(IsaPath::Scalar, dtype, m, k, &w, &x2, &mut want).unwrap();

    // The dotprod column path (runs on this M1) and the SMMLA tile path
    // (owner M2+ silicon) must both be bit-identical to the scalar columns.
    for isa in [IsaPath::NeonDotprod, IsaPath::NeonI8mm, IsaPath::Avx512Vnni] {
        if !feats.supports(isa) {
            eprintln!("skip: {isa} GEMV2 leg not available on this host");
            continue;
        }
        let mut got = vec![0.0f32; 2 * m];
        kernels::kquant_gemv2_i8_on(isa, dtype, m, k, &w, &x2, &mut got).unwrap();
        assert_eq!(got, want, "INT8 GEMV2 on {isa} vs scalar-int8 columns");
    }
}

#[test]
fn int8_auto_selector_matches_forced_best_path() {
    let feats = CpuFeatures::detect();
    let mut rng = Rng::new(0x3113_3113);
    let dtype = KQuantDtype::Q4K;
    let (m, k) = (3usize, 256usize);
    let w: Vec<u8> = (0..m)
        .flat_map(|_| random_blocks(&mut rng, dtype, 1))
        .collect();
    let x = rng.vec(k);
    let mut auto = vec![0.0f32; m];
    kernels::kquant_gemv_i8(dtype, m, k, &w, &x, &mut auto).unwrap();
    let isa = feats.best_int8_isa().unwrap_or(IsaPath::Scalar);
    let mut forced = vec![0.0f32; m];
    kernels::kquant_gemv_i8_on(isa, dtype, m, k, &w, &x, &mut forced).unwrap();
    assert_eq!(auto, forced, "auto INT8 selection must equal best_int8_isa");
}

// ---------------------------------------------------------------------
// fp16 (T14) / bf16 (T11/T17) tiers vs their oracles + f32 bands.
// ---------------------------------------------------------------------

#[test]
fn fp16_gemm_matches_structural_emulation_and_f32_band() {
    let feats = CpuFeatures::detect();
    let mut rng = Rng::new(0xF16A_F16A);
    let (m, n, k) = (4usize, 19usize, 40usize); // n=19: two strips + tail
    let a = rng.vec(m * k);
    let b = rng.vec(k * n);

    let mut emu = vec![0.0f32; m * n];
    kernels::gemm_fp16_on(IsaPath::Scalar, m, n, k, &a, &b, &mut emu).unwrap();

    // Architectural fp16 band vs the f32 GEMM (always runnable).
    let mut f32_ref = vec![0.0f32; m * n];
    gemm_f32_on(IsaPath::Scalar, m, n, k, &a, &b, None, &mut f32_ref).unwrap();
    for i in 0..m {
        for j in 0..n {
            let a_row: Vec<f32> = (0..k).map(|l| a[i * k + l]).collect();
            let b_col: Vec<f32> = (0..k).map(|l| b[l * n + j]).collect();
            let scale: f32 = a_row.iter().zip(&b_col).map(|(&x, &y)| (x * y).abs()).sum();
            let bound = kernels::dot_precision_bound(&a_row, &b_col, kernels::FP16_REL)
                + scale * kernels::FP16_REL * 2.0 * k as f32 / 8.0;
            assert!(
                (emu[i * n + j] - f32_ref[i * n + j]).abs() <= 2.0 * bound,
                "fp16 emulation vs f32 at ({i},{j}) exceeds 2x architectural bound"
            );
        }
    }

    if !feats.supports(IsaPath::NeonFp16) {
        eprintln!("skip: NeonFp16 not available on this host (fp16 kernel leg)");
        return;
    }
    // Real fp16 FMLA silicon leg (this Apple M1 executes it): ±2 fp16 ulp
    // of the structurally identical emulation.
    let mut got = vec![0.0f32; m * n];
    kernels::gemm_fp16_on(IsaPath::NeonFp16, m, n, k, &a, &b, &mut got).unwrap();
    for (i, (&g, &e)) in got.iter().zip(&emu).enumerate() {
        let band = e.abs() * 4.0 * kernels::FP16_REL + 2.0 * 2f32.powi(-24);
        assert!(
            (g - e).abs() <= band && !g.is_nan(),
            "fp16 FMLA vs emulation at {i}: got {g}, emu {e}, band {band}"
        );
    }
}

#[test]
fn bf16_matmul_tracks_f32_within_architectural_band() {
    let feats = CpuFeatures::detect();
    let mut rng = Rng::new(0xBF16_BF16);
    let (m, n, k) = (3usize, 7usize, 48usize);
    let a = rng.vec(m * k);
    let b = rng.vec(k * n);

    let mut f32_ref = vec![0.0f32; m * n];
    gemm_f32_on(IsaPath::Scalar, m, n, k, &a, &b, None, &mut f32_ref).unwrap();

    let band_at = |i: usize, j: usize| {
        let a_row: Vec<f32> = (0..k).map(|l| a[i * k + l]).collect();
        let b_col: Vec<f32> = (0..k).map(|l| b[l * n + j]).collect();
        2.0 * kernels::dot_precision_bound(&a_row, &b_col, kernels::BF16_REL)
    };

    // Emulation vs f32 (always runnable — pins the band derivation).
    let mut emu = vec![0.0f32; m * n];
    kernels::gemm_bf16_on(IsaPath::Scalar, m, n, k, &a, &b, &mut emu).unwrap();
    for i in 0..m {
        for j in 0..n {
            assert!(
                (emu[i * n + j] - f32_ref[i * n + j]).abs() <= band_at(i, j),
                "bf16 emulation vs f32 at ({i},{j}) exceeds the architectural band"
            );
        }
    }

    // Hardware legs (owner silicon / AVX-512-BF16 runners): both vs f32 and
    // vs the emulation within the same architectural band (the exact
    // internal rounding of vdpbf16ps / BFMMLA is NOT asserted — ADR M4-17
    // §(f), tightened after the owner run).
    let mut hw_checked = 0;
    for isa in [IsaPath::Avx512Bf16, IsaPath::NeonBf16] {
        if !feats.supports(isa) {
            continue;
        }
        let mut got = vec![0.0f32; m * n];
        kernels::gemm_bf16_on(isa, m, n, k, &a, &b, &mut got).unwrap();
        for i in 0..m {
            for j in 0..n {
                let band = band_at(i, j);
                let g = got[i * n + j];
                assert!(
                    (g - f32_ref[i * n + j]).abs() <= band && !g.is_nan(),
                    "{isa} bf16 vs f32 at ({i},{j}) exceeds the architectural band {band}"
                );
                assert!(
                    (g - emu[i * n + j]).abs() <= band,
                    "{isa} bf16 vs emulation at ({i},{j}) exceeds the architectural band {band}"
                );
            }
        }
        hw_checked += 1;
    }
    if hw_checked == 0 {
        eprintln!(
            "skip: no bf16 silicon on this host (Apple M1 lacks bf16; owner M2+/Zen4 leg, T23/T24)"
        );
    }
}

// ---------------------------------------------------------------------
// T21: forced-path negatives — SIGILL-free explicit errors.
// ---------------------------------------------------------------------

#[test]
fn forcing_unsupported_server_tier_is_explicit_error_not_sigill() {
    let feats = CpuFeatures::detect();
    let mut rng = Rng::new(0x5161_1234);
    let x = rng.vec(256);
    let w = random_blocks(&mut rng, KQuantDtype::Q4K, 1);
    let mut out = vec![0.0f32; 1];
    for isa in IsaPath::ALL_SIMD {
        if feats.supports(isa) {
            continue;
        }
        // The mere act of *requesting* an unsupported tier must return an
        // explicit BackendUnavailable — reaching this line at all proves no
        // SIGILL was executed (milestones.md M4-17 completion criterion).
        let err = kernels::kquant_gemv_i8_on(isa, KQuantDtype::Q4K, 1, 256, &w, &x, &mut out)
            .unwrap_err();
        assert!(
            matches!(err, vokra_core::VokraError::BackendUnavailable(_)),
            "forcing {isa} on an unsupporting host must be BackendUnavailable, got {err:?}"
        );
        let err = kernels::kquant_dequant_on(isa, KQuantDtype::Q4K, &w, 256).unwrap_err();
        assert!(matches!(err, vokra_core::VokraError::BackendUnavailable(_)));
    }
    // And the ladder never lands on an unsupported tier by itself.
    assert!(feats.supports(feats.best_isa()));
    if let Some(isa) = feats.best_int8_isa() {
        assert!(feats.supports(isa));
    }
    if let Some(isa) = feats.best_bf16_isa() {
        assert!(feats.supports(isa));
    }
    if let Some(isa) = feats.best_fp16_isa() {
        assert!(feats.supports(isa));
    }
}
