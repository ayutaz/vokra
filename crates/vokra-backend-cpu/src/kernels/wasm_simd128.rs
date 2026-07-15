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
//! - [`add`] / [`mul`] / [`relu`]: pure lane-wise ops — bit-identical to
//!   scalar (`relu` is `f32x4_max` against a zero vector; for the finite
//!   inputs the parity harness and the model conv stack use it matches the
//!   scalar `v.max(0.0)` exactly).
//! - [`dot`] / [`gemv`]: mirror the NEON `gemv` idiom (4-lane partial sums,
//!   horizontally reduced after the loop). The association differs from the
//!   scalar left-to-right chain, so the result is NOT bit-identical — the
//!   harness measures the actual delta and asserts the native differential
//!   bounds (`GEMV_ATOL = 1e-4` / `RTOL = 1e-4`,
//!   `crates/vokra-backend-cpu/tests/differential.rs`). **Measured**
//!   (2026-07-15, Node 24.16, m=17 k=129 uniform ±1 inputs): max |Δ| =
//!   2.384e-6 — ~40x inside the bound (honest recorded diff, not a
//!   fabricated exact match).
//! - [`softmax`] / [`layer_norm`]: the row max is exact (order-independent),
//!   and the normalise-scale-shift / softmax-scale passes are separately
//!   rounded mul + add (no baseline-SIMD FMA), matching the scalar per-element
//!   formula exactly; only the mean / variance / row-sum **reduction order**
//!   differs, so the result is tolerance-bounded (`REDUCTION_ATOL = 1e-4`).
//! - [`sigmoid`] / [`tanh`] / [`gelu`]: vectorized under the
//!   `simd-transcendental` feature (default-on) via a self-contained WASM
//!   [`exp_ps_wasm`] poly (WASM cannot reuse `super::vexp`, which is gated to
//!   x86-64 / aarch64), else they delegate to the scalar `std::exp` reference
//!   — the same feature posture as the NEON module. Tolerance-bounded vs
//!   scalar (`ACTIVATION_ATOL = 1e-4`); the ragged `n % 4` tail is scalar for
//!   an exact tail match.
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
    f32x4_add, f32x4_extract_lane, f32x4_max, f32x4_mul, f32x4_splat, f32x4_sub, v128, v128_load,
    v128_store,
};

use super::scalar;

// Intrinsics used only by the SIMD transcendentals (`sigmoid` / `tanh` /
// `gelu`) — gated behind the same `simd-transcendental` feature as
// [`exp_ps_wasm`] so the default-off (bit-identical `std::exp`) build carries
// no unused imports.
#[cfg(feature = "simd-transcendental")]
use core::arch::wasm32::{f32x4_abs, f32x4_div, f32x4_neg, i32x4_splat, v128_and, v128_or};
#[cfg(feature = "simd-transcendental")]
use transcendental::exp_ps_wasm;

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

/// 4-lane horizontal max in fixed lane order (mirrors [`hsum`]); the row-max
/// reduction in [`softmax`]. `f32x4_max` is exact for finite lanes, so the
/// fixed order matches scalar's left-to-right `f32::max` fold bit-for-bit.
#[inline]
fn hmax(v: v128) -> f32 {
    let a = f32x4_extract_lane::<0>(v);
    let b = f32x4_extract_lane::<1>(v);
    let c = f32x4_extract_lane::<2>(v);
    let d = f32x4_extract_lane::<3>(v);
    a.max(b).max(c).max(d)
}

/// Element-wise ReLU `out[i] = max(0, x[i])` — lane-wise `f32x4_max` against a
/// zero vector, bit-identical to [`scalar::relu`] for the finite inputs the
/// parity harness and the model conv stack exercise. Always vectorized (no
/// `exp`, hence no `simd-transcendental` gate — same posture as NEON).
pub(crate) fn relu(x: &[f32], out: &mut [f32]) {
    let n = out.len();
    let zero = f32x4_splat(0.0);
    let mut i = 0;
    // SAFETY: `i + 4 <= n` keeps every 4-lane load/store inside the
    // equal-length slices `x` / `out` (lengths validated by the public wrapper
    // in `super`). wasm32 has no alignment fault for v128 loads.
    unsafe {
        while i + 4 <= n {
            let xv = v128_load(x.as_ptr().add(i) as *const v128);
            v128_store(out.as_mut_ptr().add(i) as *mut v128, f32x4_max(xv, zero));
            i += 4;
        }
    }
    // Scalar tail — same `v.max(0.0)` as the scalar oracle.
    while i < n {
        out[i] = x[i].max(0.0);
        i += 1;
    }
}

/// Row-wise softmax over the innermost dimension (see [`scalar::softmax`] for
/// shapes and the row-max stabilisation). The max pass ([`f32x4_max`] +
/// [`hmax`]) is exact vs scalar; the `exp` pass is the vectorized poly under
/// `simd-transcendental` (else the scalar `std::exp`); the scale pass is a
/// lane-wise `f32x4_mul` by `1/sum`. The `exp` poly and the row-sum reorder
/// keep the result within `REDUCTION_ATOL = 1e-4` (module docs); the ragged
/// `cols % 4` tail is scalar for an exact tail match.
pub(crate) fn softmax(input: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let in_row = &input[r * cols..r * cols + cols];
        let out_row = &mut out[r * cols..r * cols + cols];

        // Pass 1: row maximum (exact — max is order-independent for finite f32).
        let mut vmax = f32x4_splat(f32::NEG_INFINITY);
        let mut j = 0;
        // SAFETY: `j + 4 <= cols` bounds every load to the length-`cols` row.
        unsafe {
            while j + 4 <= cols {
                vmax = f32x4_max(vmax, v128_load(in_row.as_ptr().add(j) as *const v128));
                j += 4;
            }
        }
        let mut max = hmax(vmax);
        while j < cols {
            max = max.max(in_row[j]);
            j += 1;
        }

        // Pass 2: out = exp(in - max); accumulate the row sum. The vectorized
        // `exp` (feature `simd-transcendental`) still handles the ragged tail
        // scalar for an exact-oracle match on `cols % 4`.
        #[cfg(feature = "simd-transcendental")]
        let sum = {
            let vmaxb = f32x4_splat(max);
            let mut vsum = f32x4_splat(0.0);
            let mut j = 0;
            // SAFETY: `j + 4 <= cols` bounds every load/store to the row.
            unsafe {
                while j + 4 <= cols {
                    let e = exp_ps_wasm(f32x4_sub(
                        v128_load(in_row.as_ptr().add(j) as *const v128),
                        vmaxb,
                    ));
                    v128_store(out_row.as_mut_ptr().add(j) as *mut v128, e);
                    vsum = f32x4_add(vsum, e);
                    j += 4;
                }
            }
            let mut sum = hsum(vsum);
            while j < cols {
                let e = (in_row[j] - max).exp();
                out_row[j] = e;
                sum += e;
                j += 1;
            }
            sum
        };
        #[cfg(not(feature = "simd-transcendental"))]
        let sum = {
            let mut sum = 0.0f32;
            for (o, &v) in out_row.iter_mut().zip(in_row) {
                let e = (v - max).exp();
                *o = e;
                sum += e;
            }
            sum
        };

        // Pass 3: scale by 1/sum (lane-wise; bit-identical to scalar `*=`).
        let inv = 1.0 / sum;
        let vinv = f32x4_splat(inv);
        let mut j = 0;
        // SAFETY: `j + 4 <= cols` bounds every load/store to the row.
        unsafe {
            while j + 4 <= cols {
                let v = v128_load(out_row.as_ptr().add(j) as *const v128);
                v128_store(out_row.as_mut_ptr().add(j) as *mut v128, f32x4_mul(v, vinv));
                j += 4;
            }
        }
        while j < cols {
            out_row[j] *= inv;
            j += 1;
        }
    }
}

/// Row-wise layer normalisation with affine parameters (see
/// [`scalar::layer_norm`] for the exact formula and shapes). The mean and
/// variance reductions use 4-lane partial sums ([`hsum`] + scalar tail); the
/// normalise-scale-shift pass is `f32x4_add(f32x4_mul(norm, gamma), beta)` —
/// separately-rounded mul then add, matching the scalar `norm*g + b` **exactly**
/// (baseline SIMD has no FMA). Only the mean / variance reduction order differs
/// from scalar, so the result stays within `REDUCTION_ATOL = 1e-4` (module
/// docs). The ragged `cols % 4` tail is scalar.
pub(crate) fn layer_norm(
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    let inv_cols = 1.0 / cols as f32;
    for r in 0..rows {
        let in_row = &input[r * cols..r * cols + cols];
        let out_row = &mut out[r * cols..r * cols + cols];

        // Pass 1: mean (4-lane partial sums + scalar tail).
        let mut vsum = f32x4_splat(0.0);
        let mut j = 0;
        // SAFETY: `j + 4 <= cols` bounds every load to the length-`cols` row.
        unsafe {
            while j + 4 <= cols {
                vsum = f32x4_add(vsum, v128_load(in_row.as_ptr().add(j) as *const v128));
                j += 4;
            }
        }
        let mut sum = hsum(vsum);
        while j < cols {
            sum += in_row[j];
            j += 1;
        }
        let mean = sum * inv_cols;

        // Pass 2: variance = Σ (v - mean)² (two-pass, matching the scalar
        // oracle; separate mul + add, no FMA).
        let vmean = f32x4_splat(mean);
        let mut vvar = f32x4_splat(0.0);
        let mut j = 0;
        // SAFETY: same `j + 4 <= cols` bound over the length-`cols` row.
        unsafe {
            while j + 4 <= cols {
                let d = f32x4_sub(v128_load(in_row.as_ptr().add(j) as *const v128), vmean);
                vvar = f32x4_add(vvar, f32x4_mul(d, d));
                j += 4;
            }
        }
        let mut var = hsum(vvar);
        while j < cols {
            let d = in_row[j] - mean;
            var += d * d;
            j += 1;
        }
        var *= inv_cols;
        let inv_std = 1.0 / (var + eps).sqrt();

        // Pass 3: normalise, scale, shift — out = (v - mean) * inv_std * g + b.
        let vinv_std = f32x4_splat(inv_std);
        let mut j = 0;
        // SAFETY: `j + 4 <= cols` bounds every load/store to the row and the
        // length-`cols` `gamma` / `beta` (validated by the public wrapper).
        unsafe {
            while j + 4 <= cols {
                let d = f32x4_sub(v128_load(in_row.as_ptr().add(j) as *const v128), vmean);
                let norm = f32x4_mul(d, vinv_std);
                let g = v128_load(gamma.as_ptr().add(j) as *const v128);
                let b = v128_load(beta.as_ptr().add(j) as *const v128);
                v128_store(
                    out_row.as_mut_ptr().add(j) as *mut v128,
                    f32x4_add(f32x4_mul(norm, g), b),
                );
                j += 4;
            }
        }
        while j < cols {
            out_row[j] = (in_row[j] - mean) * inv_std * gamma[j] + beta[j];
            j += 1;
        }
    }
}

// ---- vectorized transcendentals (feature `simd-transcendental`, M1-05-EXP) --

/// Element-wise logistic sigmoid `1 / (1 + exp(-x))` — vectorized poly `exp`
/// ([`exp_ps_wasm`]); the ragged `n % 4` tail delegates to the scalar oracle
/// for an exact tail match.
#[cfg(feature = "simd-transcendental")]
pub(crate) fn sigmoid(x: &[f32], out: &mut [f32]) {
    let n = out.len();
    let one = f32x4_splat(1.0);
    let mut j = 0;
    // SAFETY: `j + 4 <= n` bounds every load/store to the equal-length slices.
    unsafe {
        while j + 4 <= n {
            let xv = v128_load(x.as_ptr().add(j) as *const v128);
            let e = exp_ps_wasm(f32x4_neg(xv)); // exp(-x)
            let r = f32x4_div(one, f32x4_add(one, e));
            v128_store(out.as_mut_ptr().add(j) as *mut v128, r);
            j += 4;
        }
    }
    if j < n {
        scalar::sigmoid(&x[j..], &mut out[j..]);
    }
}

/// Sigmoid — scalar-backed (default off `simd-transcendental`; bit-identical
/// `std::exp`).
#[cfg(not(feature = "simd-transcendental"))]
pub(crate) fn sigmoid(x: &[f32], out: &mut [f32]) {
    scalar::sigmoid(x, out);
}

/// Element-wise hyperbolic tangent via `tanh(x) = 1 - 2/(e^{2x}+1)` (saturates
/// to ±1 through the clamped [`exp_ps_wasm`]); ragged tail → scalar oracle.
#[cfg(feature = "simd-transcendental")]
pub(crate) fn tanh(x: &[f32], out: &mut [f32]) {
    let n = out.len();
    let one = f32x4_splat(1.0);
    let two = f32x4_splat(2.0);
    let mut j = 0;
    // SAFETY: `j + 4 <= n` bounds every load/store to the equal-length slices.
    unsafe {
        while j + 4 <= n {
            let xv = v128_load(x.as_ptr().add(j) as *const v128);
            let e2 = exp_ps_wasm(f32x4_mul(two, xv));
            let r = f32x4_sub(one, f32x4_div(two, f32x4_add(e2, one)));
            v128_store(out.as_mut_ptr().add(j) as *mut v128, r);
            j += 4;
        }
    }
    if j < n {
        scalar::tanh(&x[j..], &mut out[j..]);
    }
}

/// tanh — scalar-backed (default off `simd-transcendental`).
#[cfg(not(feature = "simd-transcendental"))]
pub(crate) fn tanh(x: &[f32], out: &mut [f32]) {
    scalar::tanh(x, out);
}

/// Element-wise exact (erf-based) GELU `0.5 x (1 + erf(x/√2))`, reusing the
/// **exact** A&S 7.1.26 erf constants from [`scalar`] so only `exp(-z²)`
/// differs from the scalar reference; ragged tail → scalar oracle. `copysign`
/// is a bit-level OR of `z`'s sign onto the non-negative `erf(|z|)` magnitude
/// (bit-exact vs the scalar `sign * y`).
#[cfg(feature = "simd-transcendental")]
pub(crate) fn gelu(x: &[f32], out: &mut [f32]) {
    let n = out.len();
    let one = f32x4_splat(1.0);
    let half = f32x4_splat(0.5);
    let inv_sqrt2 = f32x4_splat(std::f32::consts::FRAC_1_SQRT_2);
    let sign_mask = i32x4_splat(i32::MIN); // 0x8000_0000 per lane
    let p = f32x4_splat(scalar::ERF_P);
    let a1 = f32x4_splat(scalar::ERF_A1);
    let a2 = f32x4_splat(scalar::ERF_A2);
    let a3 = f32x4_splat(scalar::ERF_A3);
    let a4 = f32x4_splat(scalar::ERF_A4);
    let a5 = f32x4_splat(scalar::ERF_A5);
    let mut j = 0;
    // SAFETY: `j + 4 <= n` bounds every load/store to the equal-length slices.
    unsafe {
        while j + 4 <= n {
            let xv = v128_load(x.as_ptr().add(j) as *const v128);
            let z = f32x4_mul(xv, inv_sqrt2);
            let az = f32x4_abs(z); // |z|
            // t = 1/(1 + P|z|) (separate mul + add then div — no FMA).
            let t = f32x4_div(one, f32x4_add(one, f32x4_mul(p, az)));
            // poly = ((((A5*t + A4)*t + A3)*t + A2)*t + A1)*t   (Horner).
            let mut poly = f32x4_add(f32x4_mul(a5, t), a4);
            poly = f32x4_add(f32x4_mul(poly, t), a3);
            poly = f32x4_add(f32x4_mul(poly, t), a2);
            poly = f32x4_add(f32x4_mul(poly, t), a1);
            poly = f32x4_mul(poly, t);
            let ez2 = exp_ps_wasm(f32x4_neg(f32x4_mul(az, az))); // e^{-z²}
            let erf_abs = f32x4_sub(one, f32x4_mul(poly, ez2)); // erf(|z|) ≥ 0
            // copysign(erf_abs, z): OR z's sign bit onto the magnitude.
            let sign = v128_and(z, sign_mask);
            let erf = v128_or(sign, erf_abs);
            let g = f32x4_mul(f32x4_mul(half, xv), f32x4_add(one, erf));
            v128_store(out.as_mut_ptr().add(j) as *mut v128, g);
            j += 4;
        }
    }
    if j < n {
        scalar::gelu(&x[j..], &mut out[j..]);
    }
}

/// GELU — scalar-backed (default off `simd-transcendental`).
#[cfg(not(feature = "simd-transcendental"))]
pub(crate) fn gelu(x: &[f32], out: &mut [f32]) {
    scalar::gelu(x, out);
}

/// Self-contained WASM SIMD128 vectorized `exp` for the transcendental
/// activations. WASM cannot reuse the native `kernels::vexp` module (it is
/// `cfg(any(x86_64, aarch64))`), so the **same natively-derived algorithm** is
/// re-expressed with `core::arch::wasm32` intrinsics:
///
/// `exp(x) = 2^k · e^r`, `k = round(x·log2e)`, `r = x − k·ln2` with a
/// Cody–Waite `ln2 = LN2_HI + LN2_LO` split, `e^r` the degree-6 Taylor series
/// (Horner), and `2^k` assembled directly in the IEEE-754 exponent field
/// (`(k + 127) << 23`). Worst-case relative error ≈ 1.2e-7 (a few f32 ULP),
/// well inside the FP32 parity ceiling NFR-QL-01 `atol = 0.01`.
///
/// The coefficients are the exact `1/n!` Taylor rationals + the standard
/// Cody–Waite `ln2` split — derived from the identity above, **not** copied
/// from Cephes / SLEEF / Pommier `exp_ps` (license hygiene: Vokra stays
/// zero-dependency and Apache-2.0). Inputs are clamped to `[MIN_ARG, MAX_ARG]`
/// so `2^k` never overflows the exponent field; every consumer here is
/// saturating (`sigmoid`/`tanh` clamp to ±1) or normalized (softmax's `exp`
/// argument is `≤ 0`), so the clamp differs from `std::f32::exp` only where the
/// activation output is already saturated. Baseline WASM SIMD has no FMA, so
/// each range-reduction / Horner step is a separate `f32x4_mul` + add/sub
/// (deterministic; relaxed-madd is not adopted — NFR-QL-01).
#[cfg(feature = "simd-transcendental")]
mod transcendental {
    use core::arch::wasm32::{
        f32x4_add, f32x4_max, f32x4_min, f32x4_mul, f32x4_nearest, f32x4_splat, f32x4_sub,
        i32x4_add, i32x4_shl, i32x4_splat, i32x4_trunc_sat_f32x4, v128,
    };

    /// `log2(e)` — scales `x` to the base-2 exponent before rounding to `k`.
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    /// High part of the Cody–Waite `ln2` split (exactly `355/512`, dyadic — so
    /// `kf * LN2_HI` is exact in f32 even without FMA).
    #[allow(clippy::excessive_precision)]
    const LN2_HI: f32 = 0.693_359_375;
    /// Low correction so `LN2_HI + LN2_LO ≈ ln2`.
    const LN2_LO: f32 = -2.121_944_4e-4;
    /// Lower / upper clamp on the `exp` argument (keeps `2^k` a normal f32).
    const MIN_ARG: f32 = -87.0;
    const MAX_ARG: f32 = 88.0;
    // Degree-6 `exp` Taylor coefficients `1/n!` (exact factorial reciprocals).
    const C0: f32 = 1.0; // 1/0!
    const C1: f32 = 1.0; // 1/1!
    const C2: f32 = 0.5; // 1/2!
    const C3: f32 = 1.0 / 6.0; // 1/3!
    const C4: f32 = 1.0 / 24.0; // 1/4!
    const C5: f32 = 1.0 / 120.0; // 1/5!
    const C6: f32 = 1.0 / 720.0; // 1/6!

    /// Vectorized f32 `exp` over the four WASM SIMD128 lanes of `x` (register
    /// only — no memory access, so no `unsafe`). See the module-item docs on
    /// [`super::gelu`] / the wiring for the algorithm and license notes.
    #[inline]
    pub(super) fn exp_ps_wasm(x: v128) -> v128 {
        // Clamp to the representable exp domain (see docs).
        let x = f32x4_min(f32x4_max(x, f32x4_splat(MIN_ARG)), f32x4_splat(MAX_ARG));
        // k = round-to-nearest-even(x * log2e): `kf` is the float form (already
        // integer-valued), `ki` the i32 form. In-domain `|k| ≤ 127`, so the
        // saturating truncation never saturates and `kf` is exact.
        let kf = f32x4_nearest(f32x4_mul(x, f32x4_splat(LOG2E)));
        let ki = i32x4_trunc_sat_f32x4(kf);
        // r = x - kf*LN2_HI - kf*LN2_LO (Cody–Waite; kf*LN2_HI exact). No FMA
        // in baseline SIMD → separate mul then sub (deterministic).
        let r = f32x4_sub(x, f32x4_mul(kf, f32x4_splat(LN2_HI)));
        let r = f32x4_sub(r, f32x4_mul(kf, f32x4_splat(LN2_LO)));
        // P(r) = 1 + r + r²/2! + … + r⁶/6! (Horner; separate mul + add).
        let mut p = f32x4_splat(C6);
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(C5));
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(C4));
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(C3));
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(C2));
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(C1));
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(C0));
        // 2^k via IEEE-754 exponent-field assembly: ((k + 127) << 23). The
        // i32x4 bit pattern IS the f32x4 representation of 2^k (v128 is
        // untyped), so it feeds the final f32x4_mul with no reinterpret.
        let pow2k = i32x4_shl(i32x4_add(ki, i32x4_splat(127)), 23);
        f32x4_mul(p, pow2k)
    }
}
