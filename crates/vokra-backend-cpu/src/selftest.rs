//! Single-binary runtime self-consistency proof (M1-05).
//!
//! [`selftest`] is a small, allocation-light routine that a *shipped* Vokra
//! binary can call to prove — on the actual host CPU, at run time — that the
//! ISA path it selected ([`crate::active_isa`]) computes the same results as
//! the portable scalar oracle. It underpins the "single binary runs on
//! x86-64 **and** ARM64 via runtime dispatch" completion claim (FR-BE-01,
//! FR-EX-06): the one artifact detects its host features, dispatches to the
//! matching SIMD kernels, and here re-checks that dispatch against the scalar
//! reference before doing real work.
//!
//! It is deliberately an **internal oracle** (differential-vs-scalar), never a
//! fabricated reference number: every host-supported SIMD path is run against
//! the scalar path on the same fixed, seeded inputs and the maximum absolute
//! deviation is asserted under [`SELFTEST_ATOL`] (mirroring the ceiling used
//! by `tests/differential.rs`, itself well under the FP32 parity bound
//! NFR-QL-01 `atol = 0.01`). A deviation above the tolerance means a SIMD
//! kernel is miscompiled or the CPU is misdetected on this host, so the
//! backend is not trustworthy here — reported as
//! [`VokraError::BackendUnavailable`].
//!
//! Intended callers: the `vokra-cli` `doctor` / `bench` surface (M1-10) and
//! the ASR demo's one-line ISA log (M0-06-T26) — both can print
//! [`SelftestReport`] to show which path is live and that it is self-consistent.

use vokra_core::{Result, VokraError};

use crate::dispatch::{self, active_isa};
use crate::features::{CpuFeatures, IsaPath};
use crate::kernels;

/// Absolute-tolerance ceiling for the self-consistency check.
///
/// Matches the loosest per-kernel tolerance in `tests/differential.rs`
/// (`GEMM_ATOL`, whose error grows with the K-reduction length) and stays far
/// under the FP32 parity bound NFR-QL-01 `atol = 0.01`. A companion relative
/// term ([`SELFTEST_RTOL`]) absorbs GEMM's magnitude-dependent rounding.
pub const SELFTEST_ATOL: f32 = 1e-3;

/// Relative-tolerance companion to [`SELFTEST_ATOL`] (GEMM reduction rounding).
pub const SELFTEST_RTOL: f32 = 1e-4;

/// Outcome of [`selftest`]: which path is live and how far the host SIMD
/// kernels drifted from the scalar oracle.
#[derive(Debug, Clone)]
pub struct SelftestReport {
    /// The ISA path this process selected (host default, or the
    /// `VOKRA_CPU_ISA` override) — see [`crate::active_isa`].
    pub active_isa: IsaPath,
    /// The CPU features detected on this host.
    pub features: CpuFeatures,
    /// The SIMD paths that were cross-checked against the scalar oracle.
    ///
    /// Empty on a host with no SIMD path (only scalar is available), in which
    /// case the check is a trivial identity and always passes.
    pub checked_paths: Vec<IsaPath>,
    /// The largest absolute deviation observed between any host SIMD kernel
    /// and the scalar oracle (0.0 when no SIMD path was checked).
    pub max_abs_diff: f32,
    /// The absolute tolerance applied ([`SELFTEST_ATOL`]).
    pub tolerance: f32,
}

impl core::fmt::Display for SelftestReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cpu selftest OK: active={} features(avx2={} fma={} avx512={}/{}/{}/{} vnni={} bf16={} avxvnni={} neon={} fp16={} dotprod={} i8mm={} nbf16={} rvv_v={} zvfh={} rvv071={}{}) checked={:?} max_abs_diff={:.3e} (atol {:.0e})",
            self.active_isa,
            self.features.avx2,
            self.features.fma,
            self.features.avx512f,
            self.features.avx512dq,
            self.features.avx512bw,
            self.features.avx512vl,
            self.features.avx512vnni,
            self.features.avx512bf16,
            self.features.avxvnni256,
            self.features.neon,
            self.features.neon_fp16,
            self.features.neon_dotprod,
            self.features.neon_i8mm,
            self.features.neon_bf16,
            self.features.rvv_v,
            self.features.rvv_zvfh,
            self.features.rvv_071,
            if self.features.rvv_071 && !self.features.rvv_071_auto {
                " (override-only)"
            } else {
                ""
            },
            self.checked_paths,
            self.max_abs_diff,
            self.tolerance,
        )
    }
}

/// Minimal reproducible PRNG (xorshift64*), so the selftest needs no external
/// `rand` dependency (NFR-DS-02 zero-dependency invariant) and produces the
/// same inputs on every host.
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
        let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32; // 24 bits
        bits as f32 / (1u32 << 24) as f32 * 2.0 - 1.0 // [-1, 1)
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Compares `simd` against the scalar `oracle`, updating `max_abs_diff` and
/// returning an explicit error the moment any element exceeds the
/// atol+rtol band — a genuine self-consistency failure (never a silent pass).
fn compare(
    kernel: &str,
    isa: IsaPath,
    oracle: &[f32],
    simd: &[f32],
    max_abs_diff: &mut f32,
) -> Result<()> {
    for (i, (&o, &s)) in oracle.iter().zip(simd).enumerate() {
        let diff = (o - s).abs();
        if diff > *max_abs_diff {
            *max_abs_diff = diff;
        }
        let tol = SELFTEST_ATOL + SELFTEST_RTOL * o.abs();
        // `diff > tol` alone would let a NaN (from a corrupt SIMD result)
        // slip through, since every NaN comparison is false; fail on it too.
        if diff > tol || diff.is_nan() {
            return Err(VokraError::BackendUnavailable(format!(
                "cpu selftest failed: {isa} kernel `{kernel}` deviates from the scalar oracle at \
                 index {i} (scalar={o}, {isa}={s}, |diff|={diff} > tol {tol}); the SIMD path is \
                 miscompiled or the host CPU is misdetected"
            )));
        }
    }
    Ok(())
}

/// Runs every host-supported kernel path against the scalar oracle on fixed
/// seeded inputs and returns a [`SelftestReport`], or
/// [`VokraError::BackendUnavailable`] if any path disagrees.
///
/// This is the runtime companion to the compile-time `tests/differential.rs`
/// harness: the same differential-vs-scalar idea, but callable from the
/// shipped binary so a deployed host can self-verify its dispatch
/// (FR-BE-01, FR-EX-06). It is cheap (small fixed shapes) and deterministic.
///
/// # Errors
/// Returns [`VokraError::BackendUnavailable`] if a host SIMD kernel's output
/// drifts from the scalar oracle beyond [`SELFTEST_ATOL`] (a miscompiled SIMD
/// path or a misdetected CPU), and propagates any
/// [`VokraError::InvalidArgument`] from the kernels (not expected here, since
/// all shapes are internally consistent).
pub fn selftest() -> Result<SelftestReport> {
    let features = CpuFeatures::detect();
    let active_isa = active_isa();

    // The SIMD paths this host can actually run (Scalar is the oracle, not a
    // "checked" path). Arch families are exclusive, but within a family a
    // host may support several tiers (this Apple M1 dev machine reports
    // Neon + NeonFp16 + NeonDotprod; a Zen4 server reports Avx2 + the four
    // x86 server tiers) — every supported tier's f32 table is cross-checked
    // below, which is the M4-17 "runtime detect で非対応 CPU は SIGILL せず"
    // completion surface: a misdetected path fails here with an explicit
    // BackendUnavailable before any real work runs. On riscv64, Rvv / Rvv071
    // (M3-13 / M4-08) are generation-exclusive — the M4-08 probe never
    // reports both (the RVV 1.0 misdetection guard) — so the `supports`
    // filter guarantees the selftest never touches 1.0 encodings on a 0.7.1
    // hart (and vice versa): the on-device safety net behind the M4-08
    // completion condition ("LicheePi 4A で動作確認"); any oracle deviation
    // surfaces as an explicit `BackendUnavailable` from `compare`, never a
    // silent pass.
    let checked_paths: Vec<IsaPath> = IsaPath::ALL_SIMD
        .into_iter()
        .filter(|&isa| features.supports(isa))
        .collect();

    let mut max_abs_diff = 0.0f32;

    // Fixed seeds per kernel keep the inputs identical across hosts and runs.
    let mut rng = Rng::new(0x5E1F_7E57);

    // GEMM 6x13x9 with bias — n = 13 exercises both the AVX2 (8) and NEON (4)
    // lane tails; the K = 9 reduction is where FP32 error is largest.
    let (m, n, k) = (6usize, 13usize, 9usize);
    let ga = rng.vec(m * k);
    let gb = rng.vec(k * n);
    let gbias = rng.vec(n);
    // Element-wise / activation inputs — len 41 = 5*8+1 = 10*4+1 (ragged tail
    // on both ISAs).
    let ex = rng.vec(41);
    let ey = rng.vec(41);
    // Softmax 3x17 and layer-norm 2x13 (reduction kernels with tails).
    let sm = rng.vec(3 * 17);
    let ln = rng.vec(2 * 13);
    let gamma = rng.vec(13);
    let beta = rng.vec(13);
    // GEMV 7x19 with per-row bias — k = 19 spans the 16-lane (NEON) /
    // single-8-wide + scalar (AVX2) reduction remainders of the logits-head
    // fast path. (Generated last so the other kernels' seeded inputs are
    // unchanged.)
    let (gvm, gvk) = (7usize, 19usize);
    let gva = rng.vec(gvm * gvk);
    let gvx = rng.vec(gvk);
    let gvbias = rng.vec(gvm);

    // Scalar oracles (computed once).
    let mut o_gemm = vec![0.0; m * n];
    kernels::gemm_f32_on(
        IsaPath::Scalar,
        m,
        n,
        k,
        &ga,
        &gb,
        Some(&gbias),
        &mut o_gemm,
    )?;
    let mut o_gemv = vec![0.0; gvm];
    kernels::gemv_f32_on(
        IsaPath::Scalar,
        gvm,
        gvk,
        &gva,
        &gvx,
        Some(&gvbias),
        &mut o_gemv,
    )?;
    let mut o_add = vec![0.0; ex.len()];
    kernels::add_f32_on(IsaPath::Scalar, &ex, &ey, &mut o_add)?;
    let mut o_mul = vec![0.0; ex.len()];
    kernels::mul_f32_on(IsaPath::Scalar, &ex, &ey, &mut o_mul)?;
    let mut o_relu = vec![0.0; ex.len()];
    kernels::relu_f32_on(IsaPath::Scalar, &ex, &mut o_relu)?;
    let mut o_sig = vec![0.0; ex.len()];
    kernels::sigmoid_f32_on(IsaPath::Scalar, &ex, &mut o_sig)?;
    let mut o_tanh = vec![0.0; ex.len()];
    kernels::tanh_f32_on(IsaPath::Scalar, &ex, &mut o_tanh)?;
    let mut o_gelu = vec![0.0; ex.len()];
    kernels::gelu_f32_on(IsaPath::Scalar, &ex, &mut o_gelu)?;
    let mut o_sm = vec![0.0; sm.len()];
    kernels::softmax_f32_on(IsaPath::Scalar, &sm, &mut o_sm, 3, 17)?;
    let mut o_ln = vec![0.0; ln.len()];
    let eps = kernels::LAYER_NORM_DEFAULT_EPS;
    kernels::layer_norm_f32_on(IsaPath::Scalar, &ln, &mut o_ln, 2, 13, &gamma, &beta, eps)?;

    for &isa in &checked_paths {
        let mut buf = vec![0.0; m * n];
        kernels::gemm_f32_on(isa, m, n, k, &ga, &gb, Some(&gbias), &mut buf)?;
        compare("gemm", isa, &o_gemm, &buf, &mut max_abs_diff)?;

        let mut buf = vec![0.0; gvm];
        kernels::gemv_f32_on(isa, gvm, gvk, &gva, &gvx, Some(&gvbias), &mut buf)?;
        compare("gemv", isa, &o_gemv, &buf, &mut max_abs_diff)?;

        let mut buf = vec![0.0; ex.len()];
        kernels::add_f32_on(isa, &ex, &ey, &mut buf)?;
        compare("add", isa, &o_add, &buf, &mut max_abs_diff)?;
        kernels::mul_f32_on(isa, &ex, &ey, &mut buf)?;
        compare("mul", isa, &o_mul, &buf, &mut max_abs_diff)?;
        kernels::relu_f32_on(isa, &ex, &mut buf)?;
        compare("relu", isa, &o_relu, &buf, &mut max_abs_diff)?;
        kernels::sigmoid_f32_on(isa, &ex, &mut buf)?;
        compare("sigmoid", isa, &o_sig, &buf, &mut max_abs_diff)?;
        kernels::tanh_f32_on(isa, &ex, &mut buf)?;
        compare("tanh", isa, &o_tanh, &buf, &mut max_abs_diff)?;
        kernels::gelu_f32_on(isa, &ex, &mut buf)?;
        compare("gelu", isa, &o_gelu, &buf, &mut max_abs_diff)?;

        let mut buf = vec![0.0; sm.len()];
        kernels::softmax_f32_on(isa, &sm, &mut buf, 3, 17)?;
        compare("softmax", isa, &o_sm, &buf, &mut max_abs_diff)?;

        let mut buf = vec![0.0; ln.len()];
        kernels::layer_norm_f32_on(isa, &ln, &mut buf, 2, 13, &gamma, &beta, eps)?;
        compare("layer_norm", isa, &o_ln, &buf, &mut max_abs_diff)?;
    }

    // M2-04-T06: cross-check the fused log-mel dispatch path against the
    // scalar oracle on fixed seeded inputs. Runs the same differential
    // pattern as the other kernels — the failure surface is
    // `VokraError::BackendUnavailable` with a diagnostic pointing at the
    // fused_logmel kernel, so a shipped binary can report SIMD miscompile /
    // CPU misdetection in the log-mel front-end at run time.
    fused_logmel_selftest(&features, &checked_paths, &mut max_abs_diff)?;

    // M4-17-T18: cross-check the specialized server-tier kernels (K-quants
    // dequant fusion, INT8 dot, fp16/bf16 matmul) that live outside the f32
    // KernelTable. Their bounds are their own (bit-identity for dequant /
    // INT8, architectural mantissa bands for fp16/bf16), so they do not feed
    // `max_abs_diff` (which is the f32-table SELFTEST_ATOL budget) — but any
    // violation is the same explicit BackendUnavailable failure surface.
    server_tier_selftest(&features)?;

    Ok(SelftestReport {
        active_isa,
        features,
        checked_paths,
        max_abs_diff,
        tolerance: SELFTEST_ATOL,
    })
}

/// M4-17-T18: differential self-check of the specialized server-tier
/// kernels on fixed seeded inputs. Every host-supported specialized path is
/// compared against its in-crate oracle:
///
/// - K-quants dequant fusion: **bit-identical** to the `vokra-core` scalar
///   reference (T12 contract);
/// - INT8 GEMV (VNNI-512 / VNNI-256 / dotprod) and the i8mm GEMV2:
///   **bit-identical** to the scalar-int8 reference (exact integer sums +
///   shared combine, ADR M4-17 §(e));
/// - fp16 GEMM (NeonFp16): within ±2 fp16 ulp of the structurally identical
///   scalar emulation;
/// - bf16 matmul (Avx512Bf16 / NeonBf16): within the architectural bf16
///   band of the scalar bf16 emulation (ADR M4-17 §(f)).
///
/// Paths the host does not support are skipped (runtime-detect skip — on
/// this Apple M1 dev machine that is i8mm / bf16 / all AVX-512 tiers; owner
/// silicon covers them, M4-17-T23/T24).
fn server_tier_selftest(features: &CpuFeatures) -> Result<()> {
    use crate::kernels::KQuantDtype;

    let mut rng = Rng::new(0x5EA7_1E55);
    // One Q4_K row of two super-blocks; random bytes are valid payloads, but
    // pin the f16 scales to small positive normals (finite magnitudes).
    let (m, k) = (2usize, 512usize);
    let dtype = KQuantDtype::Q4K;
    let nb = k / 256;
    let mut w = vec![0u8; m * nb * dtype.block_bytes()];
    for byte in w.iter_mut() {
        *byte = (rng.next_f32().to_bits() >> 9) as u8;
    }
    for blk in 0..m * nb {
        let base = blk * dtype.block_bytes();
        w[base + 1] = 0x2C; // d exponent bits: small positive normal f16
        w[base + 3] = 0x24; // dmin
    }
    let x = rng.vec(k);

    // Scalar-int8 reference (always runnable).
    let mut int8_ref = vec![0.0f32; m];
    kernels::kquant_gemv_i8_on(IsaPath::Scalar, dtype, m, k, &w, &x, &mut int8_ref)?;

    for isa in [
        IsaPath::Avx512Vnni,
        IsaPath::AvxVnni256,
        IsaPath::NeonDotprod,
    ] {
        if !features.supports(isa) {
            continue;
        }
        let mut got = vec![0.0f32; m];
        kernels::kquant_gemv_i8_on(isa, dtype, m, k, &w, &x, &mut got)?;
        if got != int8_ref {
            return Err(VokraError::BackendUnavailable(format!(
                "cpu selftest failed: {isa} K-quant INT8 GEMV deviates from the scalar-int8 \
                 reference (exact integer sums must be bit-identical); the SIMD path is \
                 miscompiled or the host CPU is misdetected"
            )));
        }
    }
    if features.supports(IsaPath::NeonI8mm) {
        let x2: Vec<f32> = x.iter().chain(x.iter().rev()).copied().collect();
        let mut ref2 = vec![0.0f32; 2 * m];
        kernels::kquant_gemv2_i8_on(IsaPath::Scalar, dtype, m, k, &w, &x2, &mut ref2)?;
        let mut got2 = vec![0.0f32; 2 * m];
        kernels::kquant_gemv2_i8_on(IsaPath::NeonI8mm, dtype, m, k, &w, &x2, &mut got2)?;
        if got2 != ref2 {
            return Err(VokraError::BackendUnavailable(
                "cpu selftest failed: neon-i8mm K-quant INT8 GEMV2 deviates from the \
                 scalar-int8 reference"
                    .into(),
            ));
        }
    }

    // Dequant fusion bit-identity on the host's SIMD family (T12). The
    // RISC-V (Rvv / Rvv071) and WASM tiers have no SIMD dequant kernel in
    // this WP (kquant_dequant_on would return an explicit UnsupportedOp),
    // so only the x86/ARM families are cross-checked here.
    let host_best = features.best_isa();
    if !matches!(
        host_best,
        IsaPath::Scalar | IsaPath::Rvv | IsaPath::Rvv071 | IsaPath::WasmSimd128
    ) {
        let simd =
            kernels::kquant_dequant_on(host_best, dtype, &w[..nb * dtype.block_bytes()], 512)?;
        let oracle = kernels::kquant_dequant_on(
            IsaPath::Scalar,
            dtype,
            &w[..nb * dtype.block_bytes()],
            512,
        )?;
        if simd != oracle {
            return Err(VokraError::BackendUnavailable(format!(
                "cpu selftest failed: {host_best} K-quants dequant fusion is not bit-identical \
                 to the vokra-core scalar reference (T12 contract)"
            )));
        }
    }

    // fp16 GEMM vs its structurally identical emulation (±2 fp16 ulp band).
    if features.supports(IsaPath::NeonFp16) {
        let (gm, gn, gk) = (3usize, 9usize, 24usize);
        let a = rng.vec(gm * gk);
        let b = rng.vec(gk * gn);
        let mut emu = vec![0.0f32; gm * gn];
        kernels::gemm_fp16_on(IsaPath::Scalar, gm, gn, gk, &a, &b, &mut emu)?;
        let mut got = vec![0.0f32; gm * gn];
        kernels::gemm_fp16_on(IsaPath::NeonFp16, gm, gn, gk, &a, &b, &mut got)?;
        for (i, (&g, &e)) in got.iter().zip(&emu).enumerate() {
            let band = e.abs() * 2.0 * 2.0 * kernels::FP16_REL + 2.0 * 2f32.powi(-24);
            if (g - e).abs() > band || g.is_nan() {
                return Err(VokraError::BackendUnavailable(format!(
                    "cpu selftest failed: neon-fp16 GEMM deviates from the fp16 emulation \
                     oracle at index {i} (emu={e}, fp16={g}, band={band})"
                )));
            }
        }
    }

    // bf16 matmul vs the scalar bf16 emulation (architectural band; runs on
    // Zen4/Cooper-Lake-class x86 or M2+/Graviton3-class ARM hosts only).
    for isa in [IsaPath::Avx512Bf16, IsaPath::NeonBf16] {
        if !features.supports(isa) {
            continue;
        }
        let (gm, gn, gk) = (3usize, 5usize, 32usize);
        let a = rng.vec(gm * gk);
        let b = rng.vec(gk * gn);
        let mut emu = vec![0.0f32; gm * gn];
        kernels::gemm_bf16_on(IsaPath::Scalar, gm, gn, gk, &a, &b, &mut emu)?;
        let mut got = vec![0.0f32; gm * gn];
        kernels::gemm_bf16_on(isa, gm, gn, gk, &a, &b, &mut got)?;
        for i in 0..gm {
            for j in 0..gn {
                let a_row: Vec<f32> = (0..gk).map(|l| a[i * gk + l]).collect();
                let b_col: Vec<f32> = (0..gk).map(|l| b[l * gn + j]).collect();
                let band = 2.0 * kernels::dot_precision_bound(&a_row, &b_col, kernels::BF16_REL);
                let (g, e) = (got[i * gn + j], emu[i * gn + j]);
                if (g - e).abs() > band || g.is_nan() {
                    return Err(VokraError::BackendUnavailable(format!(
                        "cpu selftest failed: {isa} bf16 matmul deviates from the bf16 \
                         emulation oracle at ({i},{j}) (emu={e}, bf16={g}, band={band})"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// M2-04-T06: cross-check the fused log-mel per-frame kernel across every
/// host-supported ISA path against the scalar oracle. Called from
/// [`selftest`] at feature-detect init and updates `max_abs_diff` with the
/// worst deviation observed. On disagreement, returns
/// [`VokraError::BackendUnavailable`] naming the fused_logmel kernel.
///
/// # Tolerance
///
/// The fused kernel's SIMD paths use polynomial `log10` approximations with
/// worst-case per-element error ≪ 1e-6, and the surrounding mel-band
/// accumulation is FMA-associated (bit-close to the scalar reduction). Uses
/// the [`SELFTEST_ATOL`] + [`SELFTEST_RTOL`] band, matching the other
/// kernels — still an order of magnitude inside the FP32 parity ceiling
/// NFR-QL-01 `atol = 0.01`.
fn fused_logmel_selftest(
    features: &CpuFeatures,
    checked_paths: &[IsaPath],
    max_abs_diff: &mut f32,
) -> Result<()> {
    // A Whisper-shaped tile so both the AVX2 eight-lane FMA tail
    // (`n_bins % 8 = 1`) and the NEON four-lane FMA tail (`n_bins % 4 = 1`)
    // exercise the ragged-tail scalar cleanup that must match the scalar
    // reference bit-close.
    let (n_mels, n_bins) = (16usize, 41usize);
    let mut rng = Rng::new(0x10E1_5EED);
    let mut weights = Vec::with_capacity(n_mels * n_bins);
    for _ in 0..n_mels * n_bins {
        // Filterbank weights are non-negative — draw in [0, 1] to match the
        // real Mel filter magnitudes; using signed inputs here would let
        // opposite-sign cancellations dominate the FP32 error budget.
        weights.push((rng.next_f32() + 1.0) * 0.5);
    }
    let mut power = Vec::with_capacity(n_bins);
    for _ in 0..n_bins {
        // Power spectrum entries are non-negative; magnitude range spans the
        // Whisper log-mel domain (small-signal floor to peak power).
        power.push(((rng.next_f32() + 1.0) * 0.5) * 1e3);
    }

    // Scalar oracle via the safe crate-public dispatch on `Scalar`.
    let mut oracle = vec![0.0f32; n_mels];
    let scalar_table = dispatch::table_for(IsaPath::Scalar)?;
    (scalar_table.fused_logmel)(&weights, &power, n_mels, n_bins, 1e-10, &mut oracle);

    for &isa in checked_paths {
        // `table_for` on an unsupported path is an explicit error (already
        // covered elsewhere); the filter here mirrors the invariant that
        // `checked_paths` only contains ISAs the host reports as supported.
        if !features.supports(isa) {
            continue;
        }
        let table = dispatch::table_for(isa)?;
        let mut buf = vec![0.0f32; n_mels];
        (table.fused_logmel)(&weights, &power, n_mels, n_bins, 1e-10, &mut buf);
        compare("fused_logmel", isa, &oracle, &buf, max_abs_diff)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selftest_passes_on_this_host() {
        let report = selftest().expect("cpu selftest must pass on the CI/dev host");
        // The active path must be one this host actually supports.
        assert!(report.features.supports(report.active_isa));
        // Every checked path is a genuine SIMD path the host supports, and the
        // active default path is included whenever it is not scalar.
        for &isa in &report.checked_paths {
            assert!(report.features.supports(isa));
            assert_ne!(isa, IsaPath::Scalar);
        }
        if report.active_isa != IsaPath::Scalar {
            assert!(report.checked_paths.contains(&report.active_isa));
        }
        // Self-consistency held within tolerance.
        assert!(
            report.max_abs_diff <= SELFTEST_ATOL,
            "max_abs_diff {} exceeded atol {SELFTEST_ATOL}",
            report.max_abs_diff
        );
    }

    #[test]
    fn report_display_is_one_line() {
        let s = selftest().unwrap().to_string();
        assert!(s.starts_with("cpu selftest OK:"));
        assert!(!s.contains('\n'));
    }
}
