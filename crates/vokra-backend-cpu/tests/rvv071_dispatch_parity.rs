//! M4-08-T10 differential parity: scalar ↔ RVV **draft 0.7.1** kernel path.
//!
//! Per `docs/adr/M4-08-rvv-071-fallback.md`, the M4-08 scope is a runtime
//! dispatch scaffold for T-Head C910/C906 harts (LicheePi 4A / Milk-V Duo):
//! `kernels::rvv071::add` emits real 0.7.1 instruction words via `.insn`
//! (LLVM has no xtheadvector assembler support — ADR §T01) and the remaining
//! kernels delegate to `kernels::scalar::*` pending M4+/M5 rewrites. This
//! integration test cross-checks the two paths at the public wrapper level
//! using the `_on(IsaPath::Rvv071, …)` entry points — the M3-13-T11 shape.
//!
//! # Where does this run?
//!
//! - On the **riscv64** cross-build (CI `riscv-cross-build` job) the test
//!   compiles but is not *executed* — upstream qemu has no xtheadvector /
//!   0.7.1 emulation (ADR M4-08 §T01), so execution is only possible on
//!   real T-Head silicon. The owner track (M4-08-T15 LicheePi 4A /
//!   M4-08-T16 Milk-V Duo) exercises it on-device.
//! - On **x86-64 / aarch64** hosts (default `cargo test` matrix)
//!   `IsaPath::Rvv071` is not supported by the probe, so the `_on` entry
//!   points must return an explicit `BackendUnavailable` — asserted below to
//!   prove the FR-EX-08 no-silent-fallback contract on the primary CI
//!   runners (never a fabricated pass: the unavailable branch *asserts* the
//!   explicit error instead of silently skipping).
//!
//! Both branches share the same test names so the CI log is uniform.

use vokra_backend_cpu::{CpuFeatures, IsaPath, kernels};
use vokra_core::VokraError;

const GEMM_M: usize = 6;
const GEMM_N: usize = 13;
const GEMM_K: usize = 9;

fn seeded_vec(seed: u64, n: usize) -> Vec<f32> {
    // Deterministic xorshift64* — same PRNG used by `selftest.rs`, so this
    // test needs no external `rand` crate (NFR-DS-02 zero-dep).
    let mut state: u64 = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let mixed = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let bits = (mixed >> 40) as u32;
            (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Forced-path binary kernel entry signature (`kernels::add_f32_on` shape).
type BinaryOn = fn(IsaPath, &[f32], &[f32], &mut [f32]) -> vokra_core::Result<()>;
/// Forced-path unary kernel entry signature (`kernels::relu_f32_on` shape).
type UnaryOn = fn(IsaPath, &[f32], &mut [f32]) -> vokra_core::Result<()>;

/// Runs `op` on both the scalar oracle and the forced Rvv071 path; on a
/// non-0.7.1 host asserts the explicit `BackendUnavailable` instead.
fn check_binary(name: &str, feats: &CpuFeatures, op: BinaryOn) {
    let a = seeded_vec(0xDEAD_BEEF, 41);
    let b = seeded_vec(0xF00D_BABE, 41);
    let mut out_scalar = vec![0.0f32; 41];
    op(IsaPath::Scalar, &a, &b, &mut out_scalar).unwrap();

    let mut out_071 = vec![0.0f32; 41];
    let result = op(IsaPath::Rvv071, &a, &b, &mut out_071);
    if feats.supports(IsaPath::Rvv071) {
        result.unwrap_or_else(|e| panic!("{name}: Rvv071 must succeed on a 0.7.1 host: {e}"));
        // `add` is a single IEEE-754 op per element (bit-exact even through
        // the real 0.7.1 vfadd.vv); `mul` is the scalar delegate scaffold —
        // bit-exact by construction (ADR M4-08 §d).
        assert_eq!(out_scalar, out_071, "{name}: rvv071 must match scalar");
    } else {
        let err = result.expect_err("Rvv071 on a non-0.7.1 host = explicit error");
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "{name}: expected BackendUnavailable, got {err:?}"
        );
    }
}

/// Same shape for the unary activations.
fn check_unary(name: &str, feats: &CpuFeatures, op: UnaryOn) {
    let x = seeded_vec(0x5E1F_7E57, 41);
    let mut out_scalar = vec![0.0f32; 41];
    op(IsaPath::Scalar, &x, &mut out_scalar).unwrap();

    let mut out_071 = vec![0.0f32; 41];
    let result = op(IsaPath::Rvv071, &x, &mut out_071);
    if feats.supports(IsaPath::Rvv071) {
        result.unwrap_or_else(|e| panic!("{name}: Rvv071 must succeed on a 0.7.1 host: {e}"));
        // Scalar delegate scaffolds: bit-exact (the code path IS scalar).
        assert_eq!(out_scalar, out_071, "{name}: rvv071 must match scalar");
    } else {
        let err = result.expect_err("Rvv071 on a non-0.7.1 host = explicit error");
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "{name}: expected BackendUnavailable, got {err:?}"
        );
    }
}

#[test]
fn rvv071_dispatch_elementwise_and_activations_parity_or_explicit_error() {
    let feats = CpuFeatures::detect();
    if !feats.supports(IsaPath::Rvv071) {
        // Not a fabricated pass: the checks below still run and *assert* the
        // explicit-error contract; this line only documents why no numeric
        // comparison happens on this host (FR-EX-08 diagnostics).
        eprintln!(
            "note: host is not an RVV 0.7.1 hart — asserting explicit \
             BackendUnavailable instead of numeric parity"
        );
    }
    check_binary("add", &feats, kernels::add_f32_on);
    check_binary("mul", &feats, kernels::mul_f32_on);
    check_unary("relu", &feats, kernels::relu_f32_on);
    check_unary("sigmoid", &feats, kernels::sigmoid_f32_on);
    check_unary("tanh", &feats, kernels::tanh_f32_on);
    check_unary("gelu", &feats, kernels::gelu_f32_on);
}

#[test]
fn rvv071_dispatch_gemm_gemv_parity_or_explicit_error() {
    let feats = CpuFeatures::detect();
    let a = seeded_vec(0x5E1F_7E57, GEMM_M * GEMM_K);
    let b = seeded_vec(0xA5A5_A5A5, GEMM_K * GEMM_N);
    let bias = seeded_vec(0x1357_9BDF, GEMM_N);

    let mut out_scalar = vec![0.0f32; GEMM_M * GEMM_N];
    kernels::gemm_f32_on(
        IsaPath::Scalar,
        GEMM_M,
        GEMM_N,
        GEMM_K,
        &a,
        &b,
        Some(&bias),
        &mut out_scalar,
    )
    .expect("scalar path always supported");

    let mut out_071 = vec![0.0f32; GEMM_M * GEMM_N];
    let result_071 = kernels::gemm_f32_on(
        IsaPath::Rvv071,
        GEMM_M,
        GEMM_N,
        GEMM_K,
        &a,
        &b,
        Some(&bias),
        &mut out_071,
    );

    if feats.supports(IsaPath::Rvv071) {
        result_071.expect("Rvv071 dispatch must succeed on a 0.7.1 host");
        // Bit-exact: gemm is the scalar delegate scaffold (ADR M4-08 §d);
        // once an 0.7.1 rewrite lands this loosens to the per-kernel atol.
        assert_eq!(
            out_scalar, out_071,
            "M4-08 scaffold: rvv071 gemm delegate must match scalar exactly"
        );

        let x = seeded_vec(0x0BAD_F00D, GEMM_K);
        let mut ov = vec![0.0f32; GEMM_M];
        let mut rv = vec![0.0f32; GEMM_M];
        kernels::gemv_f32_on(IsaPath::Scalar, GEMM_M, GEMM_K, &a, &x, None, &mut ov).unwrap();
        kernels::gemv_f32_on(IsaPath::Rvv071, GEMM_M, GEMM_K, &a, &x, None, &mut rv).unwrap();
        assert_eq!(ov, rv, "rvv071 gemv delegate must match scalar exactly");
    } else {
        let err = result_071.expect_err(
            "Rvv071 dispatch on a non-0.7.1 host must be an explicit BackendUnavailable error",
        );
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "expected BackendUnavailable, got {err:?}"
        );
    }
}

#[test]
fn rvv071_dispatch_reductions_parity_or_explicit_error() {
    let feats = CpuFeatures::detect();
    let sm = seeded_vec(0xCAFE_D00D, 3 * 17);
    let mut os = vec![0.0f32; sm.len()];
    kernels::softmax_f32_on(IsaPath::Scalar, &sm, &mut os, 3, 17).unwrap();
    let mut rs = vec![0.0f32; sm.len()];
    let softmax_071 = kernels::softmax_f32_on(IsaPath::Rvv071, &sm, &mut rs, 3, 17);

    let ln = seeded_vec(0xB16B_00B5, 2 * 13);
    let gamma = seeded_vec(0x600D_CAFE, 13);
    let beta = seeded_vec(0xFEED_FACE, 13);
    let eps = kernels::LAYER_NORM_DEFAULT_EPS;
    let mut ol = vec![0.0f32; ln.len()];
    kernels::layer_norm_f32_on(IsaPath::Scalar, &ln, &mut ol, 2, 13, &gamma, &beta, eps).unwrap();
    let mut rl = vec![0.0f32; ln.len()];
    let ln_071 =
        kernels::layer_norm_f32_on(IsaPath::Rvv071, &ln, &mut rl, 2, 13, &gamma, &beta, eps);

    if feats.supports(IsaPath::Rvv071) {
        softmax_071.expect("Rvv071 softmax must succeed on a 0.7.1 host");
        ln_071.expect("Rvv071 layer_norm must succeed on a 0.7.1 host");
        assert_eq!(os, rs, "rvv071 softmax delegate must match scalar exactly");
        assert_eq!(
            ol, rl,
            "rvv071 layer_norm delegate must match scalar exactly"
        );
    } else {
        for (name, result) in [("softmax", softmax_071), ("layer_norm", ln_071)] {
            let err = result.expect_err("Rvv071 on a non-0.7.1 host = explicit error");
            assert!(
                matches!(err, VokraError::BackendUnavailable(_)),
                "{name}: expected BackendUnavailable, got {err:?}"
            );
        }
    }
}
