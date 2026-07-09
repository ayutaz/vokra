//! M3-13-T11 differential parity: scalar ↔ RVV kernel path.
//!
//! Per `docs/adr/M3-13-riscv-rvv-1.0.md`, the M3-13 scope is a runtime
//! dispatch scaffold: `kernels::rvv::add` uses actual RVV 1.0 inline asm
//! (`.option arch, +v` — `vsetvli` / `vle32.v` / `vfadd.vv` / `vse32.v`) and
//! the remaining kernels delegate to `kernels::scalar::*` pending an M4+
//! inline-asm rewrite. This integration test cross-checks the two paths at
//! the public wrapper level using the `_on(IsaPath::Rvv, …)` entry points.
//!
//! # Where does this run?
//!
//! - On the **riscv64** cross-build (via CI's `riscv-cross-build` job) the
//!   test compiles but is not *executed* by the CI runner (GitHub-hosted
//!   Ubuntu has no RVV silicon and no qemu-user pre-installed). It provides
//!   the M3-13 scaffold that a real SpacemiT K1 / Banana Pi BPI-F3 owner
//!   run (or an owner-run qemu leg) will exercise.
//! - On **x86-64 / aarch64** hosts (default `cargo test` matrix)
//!   `IsaPath::Rvv` is not `.supported()` by the probe, so the `_on(Rvv, …)`
//!   entry points return an explicit `BackendUnavailable` error — which we
//!   assert here to prove the FR-EX-08 no-silent-fallback contract still
//!   holds on the primary CI runners even though there is no RVV hardware.
//!
//! Both branches share the same test names so the CI log is uniform.

use vokra_backend_cpu::{CpuFeatures, IsaPath, kernels};
use vokra_core::VokraError;

/// GEMM shape tuned to the M0-08 differential harness — small enough that the
/// scalar oracle can be recomputed cheaply, large enough to hit both AVX2
/// (8-lane) and NEON (4-lane) tails on the reference builds.
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

#[test]
fn rvv_dispatch_matches_scalar_when_supported_else_errors_explicitly() {
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

    let mut out_rvv = vec![0.0f32; GEMM_M * GEMM_N];
    let rvv_result = kernels::gemm_f32_on(
        IsaPath::Rvv,
        GEMM_M,
        GEMM_N,
        GEMM_K,
        &a,
        &b,
        Some(&bias),
        &mut out_rvv,
    );

    if feats.supports(IsaPath::Rvv) {
        rvv_result.expect("Rvv dispatch must succeed on an RVV-supporting host");
        // Bit-exact match: gemm is currently the scalar delegate scaffold
        // (M3-13 ADR §"RVV kernel カバレッジ"); once the M4+ inline-asm gemm
        // rewrite lands this assertion loosens to atol=1e-4 per ticket.
        assert_eq!(
            out_scalar, out_rvv,
            "M3-13 scaffold: rvv gemm delegate must match scalar exactly"
        );
    } else {
        // FR-EX-08 principle: forcing an unavailable ISA is an explicit
        // BackendUnavailable, never a silent switch to scalar / avx2 / neon.
        let err = rvv_result.expect_err(
            "Rvv dispatch on a non-RVV host must be an explicit BackendUnavailable error",
        );
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "expected BackendUnavailable, got {err:?}"
        );
    }
}

#[test]
fn rvv_dispatch_elementwise_add_parity_or_explicit_error() {
    // Same shape as the RVV `add` unit test in `kernels::rvv::tests`, but
    // exercised through the public dispatch layer so we cover the whole
    // wrapper → dispatch → kernel path (this is the M3-13-T11 shape).
    let feats = CpuFeatures::detect();
    let a = seeded_vec(0xDEAD_BEEF, 41);
    let b = seeded_vec(0xF00D_BABE, 41);

    let mut out_scalar = vec![0.0f32; 41];
    kernels::add_f32_on(IsaPath::Scalar, &a, &b, &mut out_scalar).unwrap();

    let mut out_rvv = vec![0.0f32; 41];
    let rvv_result = kernels::add_f32_on(IsaPath::Rvv, &a, &b, &mut out_rvv);

    if feats.supports(IsaPath::Rvv) {
        rvv_result.expect("Rvv dispatch must succeed on an RVV-supporting host");
        // Elementwise f32 add: each output is a single IEEE-754 add, so the
        // scalar reduction and the RVV `vfadd.vv` must be bit-exact.
        assert_eq!(
            out_scalar, out_rvv,
            "rvv::add via dispatch must be bit-exact vs scalar::add"
        );
    } else {
        let err = rvv_result.expect_err("Rvv on a non-RVV host = explicit error");
        assert!(matches!(err, VokraError::BackendUnavailable(_)));
    }
}
