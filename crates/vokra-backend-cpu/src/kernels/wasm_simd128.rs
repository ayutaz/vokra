//! WASM SIMD128 f32 kernels (M4-01-T05, first slice).
//!
//! Uses `core::arch::wasm32` intrinsics (std-builtin — no external crate,
//! NFR-DS-02). Compiled only under
//! `cfg(all(target_arch = "wasm32", target_feature = "simd128"))`: WASM has
//! **no runtime CPU feature detection** — SIMD acceptance is decided when the
//! engine validates the module — so this module exists only inside the
//! simd128 artifact of the 2-artifact distribution
//! (`scripts/build-wasm.sh`, ADR M4-01-webgpu-wasm §4).
//!
//! # Determinism / accumulation order (parity discipline, NFR-QL-01)
//!
//! **Relaxed SIMD is NOT adopted** (Safari-partial per the CLAUDE.md
//! quarterly ISA watch; `f32x4.relaxed_madd` is explicitly nondeterministic
//! across engines). Every kernel below uses the deterministic
//! `f32x4_add(acc, f32x4_mul(a, b))` pair — separately-rounded mul then add,
//! exactly the scalar reference's `acc += a * b` chain:
//!
//! - [`gemm`]: vectorizes across output **columns** (j) while keeping the
//!   per-element accumulation over `l = 0..k` bias-seeded and in ascending
//!   `l` order — the identical FP32 operation sequence per output element as
//!   [`scalar::gemm`], so the result is **bit-identical to scalar** (the
//!   Node harness `tools/wasm/run-kernel-parity.mjs` asserts exact equality).
//!   This differs from NEON/AVX2 (which use fused `vfmaq`/`_mm256_fmadd_ps`)
//!   because baseline WASM SIMD128 has no FMA — a happy coincidence that
//!   makes the wasm kernel *stricter* than the native SIMD parity bound.
//! - [`add`] / [`mul`]: pure lane-wise ops — bit-identical to scalar.
//! - [`dot`] / [`gemv`]: mirror the NEON `gemv` idiom (4-lane partial sums,
//!   horizontally reduced after the loop). The association differs from the
//!   scalar left-to-right chain, so the result is NOT bit-identical — the
//!   harness measures the actual delta and asserts the native differential
//!   bounds (`GEMV_ATOL = 1e-4` / `RTOL = 1e-4`,
//!   `crates/vokra-backend-cpu/tests/differential.rs`). **Measured**
//!   (2026-07-15, Node 24.16, m=17 k=129 uniform ±1 inputs): max |Δ| =
//!   2.384e-6 — ~40x inside the bound (honest recorded diff, not a
//!   fabricated exact match).
//!
//! # Unsafe boundary (NFR-RL-07)
//!
//! `v128_load` / `v128_store` are raw-pointer intrinsics, so the inner loops
//! are `unsafe` with `// SAFETY:` comments — in-bounds is guaranteed by the
//! `while … + 4 <= n` guards plus the caller-side length validation done by
//! the public wrappers in [`super`] (same structure as [`super::neon`]).
//! Lane arithmetic (`f32x4_add` etc.) is safe on wasm32 because the feature
//! is compile-time baseline for this artifact.

use core::arch::wasm32::{
    f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, v128, v128_load, v128_store,
};

/// Row-major GEMM with optional per-column bias:
/// `out[i, j] = bias[j] + Σ_l a[i, l] * b[l, j]`.
///
/// Vectorized across `j` (4 columns per `v128`); the `l` accumulation is
/// bias-seeded and ascending, so every output element runs the identical
/// separately-rounded mul+add chain as [`scalar::gemm`] → bit-identical
/// results (see module docs). The column tail (`n % 4`) falls back to the
/// scalar per-element loop with the same ordering.
pub(crate) fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    for i in 0..m {
        let out_row = &mut out[i * n..i * n + n];
        match bias {
            Some(bias) => out_row.copy_from_slice(bias),
            None => out_row.fill(0.0),
        }
        for l in 0..k {
            let a_il = a[i * k + l];
            let av = f32x4_splat(a_il);
            let b_row = &b[l * n..l * n + n];
            let mut j = 0;
            // SAFETY: `j + 4 <= n` keeps every 4-lane load/store inside the
            // length-`n` slices `b_row` and `out_row` (lengths validated by
            // the public wrapper in `super`). The pointers come straight from
            // in-bounds slice indexing; wasm32 has no alignment fault for
            // v128 loads (the engine handles unaligned access).
            unsafe {
                while j + 4 <= n {
                    let bv = v128_load(b_row.as_ptr().add(j) as *const v128);
                    let ov = v128_load(out_row.as_ptr().add(j) as *const v128);
                    v128_store(
                        out_row.as_mut_ptr().add(j) as *mut v128,
                        f32x4_add(ov, f32x4_mul(av, bv)),
                    );
                    j += 4;
                }
            }
            // Scalar column tail — same `+= a*b` chain, same order.
            while j < n {
                out_row[j] += a_il * b_row[j];
                j += 1;
            }
        }
    }
}

/// 4-lane horizontal sum, reduced in fixed lane order `((l0+l1)+l2)+l3` so
/// the result is deterministic across engines.
#[inline]
fn hsum(v: v128) -> f32 {
    ((f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v)) + f32x4_extract_lane::<2>(v))
        + f32x4_extract_lane::<3>(v)
}

/// Dot product with 4-lane partial sums (NEON `gemv` idiom): the association
/// differs from the scalar left-to-right chain, so callers treat the result
/// as tolerance-bounded, not bit-identical (module docs).
#[inline]
pub(crate) fn dot(a: &[f32], x: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), x.len());
    let k = a.len();
    let mut acc = f32x4_splat(0.0);
    let mut l = 0;
    // SAFETY: `l + 4 <= k` keeps every 4-lane load inside the length-`k`
    // slices `a` and `x` (equal lengths asserted above; validated by the
    // public wrappers).
    unsafe {
        while l + 4 <= k {
            let av = v128_load(a.as_ptr().add(l) as *const v128);
            let xv = v128_load(x.as_ptr().add(l) as *const v128);
            acc = f32x4_add(acc, f32x4_mul(av, xv));
            l += 4;
        }
    }
    let mut s = hsum(acc);
    while l < k {
        s += a[l] * x[l];
        l += 1;
    }
    s
}

/// Row-major matrix-vector product with optional per-row bias:
/// `out[i] = bias[i] + Σ_l a[i, l] * x[l]` (the Whisper tied-logits head).
///
/// Per-row [`dot`] with 4-lane partial sums; tolerance-bounded vs scalar
/// (module docs). Bias is added after the reduction, matching the NEON
/// `gemv` ordering.
pub(crate) fn gemv(
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    for i in 0..m {
        let row = &a[i * k..i * k + k];
        let s = dot(row, x);
        out[i] = match bias {
            Some(bias) => bias[i] + s,
            None => s,
        };
    }
}

/// Element-wise `out[i] = a[i] + b[i]` — lane-wise, bit-identical to scalar.
pub(crate) fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = out.len();
    let mut i = 0;
    // SAFETY: `i + 4 <= n` keeps every 4-lane load/store inside the equal
    // length-`n` slices `a` / `b` / `out` (validated by the public wrapper).
    unsafe {
        while i + 4 <= n {
            let av = v128_load(a.as_ptr().add(i) as *const v128);
            let bv = v128_load(b.as_ptr().add(i) as *const v128);
            v128_store(out.as_mut_ptr().add(i) as *mut v128, f32x4_add(av, bv));
            i += 4;
        }
    }
    while i < n {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

/// Element-wise `out[i] = a[i] * b[i]` — lane-wise, bit-identical to scalar.
pub(crate) fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = out.len();
    let mut i = 0;
    // SAFETY: `i + 4 <= n` keeps every 4-lane load/store inside the equal
    // length-`n` slices `a` / `b` / `out` (validated by the public wrapper).
    unsafe {
        while i + 4 <= n {
            let av = v128_load(a.as_ptr().add(i) as *const v128);
            let bv = v128_load(b.as_ptr().add(i) as *const v128);
            v128_store(out.as_mut_ptr().add(i) as *mut v128, f32x4_mul(av, bv));
            i += 4;
        }
    }
    while i < n {
        out[i] = a[i] * b[i];
        i += 1;
    }
}

// The remaining kernels (relu / sigmoid / tanh / gelu / softmax /
// layer_norm / fused_logmel) delegate to the portable scalar reference in
// this first slice — see `crate::dispatch::wasm_simd128_table`, which wires
// `scalar::*` directly (same posture as the M3-13 RVV scaffold). SIMD
// rewrites are a follow-up.
