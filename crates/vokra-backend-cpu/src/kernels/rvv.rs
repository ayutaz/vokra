//! RISC-V RVV 1.0 base kernels (M3-13-T04..T09).
//!
//! Compiled only on `target_arch = "riscv64"`. The M3-13 scope is a **runtime
//! dispatch scaffold** — the completion bar is "cross-build validates + dispatch
//! path unit tests pass on qemu-user or a real board" (`docs/adr/M3-13-riscv-rvv-1.0.md` §d).
//! Not every kernel here uses inline RVV asm yet: **`add` is the demonstrator
//! that emits real RVV instructions** (via `.option arch, +v` inside an
//! `asm!` block so the surrounding rv64gc baseline binary stays runnable on
//! RVV-less harts); the remaining kernels currently forward to the portable
//! scalar reference in [`super::scalar`]. The runtime dispatch table
//! nevertheless routes through this module on any host that probes as RVV,
//! which is what the CI cross-build + differential test verify. The RVV
//! 0.7.1 fallback tier landed in M4-08 as the encoding-incompatible peer
//! module [`super::rvv071`] (T-Head C910/C906 — LicheePi 4A / Milk-V Duo);
//! M4+/M5 follow-ups will replace the remaining scalar delegates (GEMM /
//! softmax / mel inline-asm rewrites) one-by-one in both tiers.
//!
//! # Why inline asm instead of `std::arch::riscv64` intrinsics?
//!
//! stable rustc 1.85 has no `is_riscv_feature_detected!` and no
//! `std::arch::riscv64` V intrinsics — they live under the unstable
//! `riscv_ext_intrinsics` feature (issue #114544). Similarly
//! `#[target_feature(enable = "v")]` is unstable (`riscv_target_feature`,
//! issue #44839). To ship a **single baseline rv64gc binary** that
//! dispatches to V at run time we therefore use `core::arch::asm!` with a
//! `.option arch, +v` / `.option pop` fence to activate V inside the asm
//! block only. See ADR M3-13 for the full rationale.
//!
//! # V register clobber caveat
//!
//! stable rustc has no `vreg` register class, so the `asm!` blocks below
//! cannot explicitly clobber v0..v31. This is sound as long as the
//! **surrounding compilation stays at the rv64gc baseline** — no
//! `-C target-feature=+v`, no `+v` in a global `RUSTFLAGS`. At baseline the
//! register allocator cannot spill f32 / f64 values into V registers, so
//! the vector unit is invisible to Rust and cannot be corrupted by our
//! `vsetvli` / `vle32.v` / `vfadd.vv` / `vse32.v` sequences. If the workspace
//! later opts into a V-globally-enabled profile (unlikely — it would break
//! the single-baseline-binary contract) this restriction must be lifted by
//! adding explicit V clobbers.

use crate::kernels::scalar;

/// Element-wise `out[i] = a[i] + b[i]` via RVV 1.0 vector float-add
/// (`vfadd.vv`).
///
/// Precondition (checked by the caller in `super::kernels`): `a.len() ==
/// b.len() == out.len()`.
///
/// # RVV 1.0 body
///
/// The inner loop is a canonical vsetvli / vle32.v / vfadd.vv / vse32.v
/// stream over `a.len()` f32 elements, with `LMUL=m4` so a single tail step
/// consumes up to 4 * VLEN/32 lanes (32 f32 lanes on a VLEN=256 hart like
/// SpacemiT K1). `vsetvli` returns the actual VL per step so the tail
/// handles a non-multiple of the vector length without a separate scalar
/// epilogue.
///
/// # Safety
///
/// The dispatch layer only ever routes this kernel on a host where
/// [`crate::features::CpuFeatures::detect`] reports `rvv_v = true`. On any
/// RVV-less riscv64 hart calling this function would execute the illegal
/// instruction `vsetvli`; that path is unreachable in normal use because the
/// [`crate::dispatch::table_for`] validator rejects `IsaPath::Rvv` on such a
/// host with an explicit `BackendUnavailable` error (FR-EX-08 principle).
pub(crate) fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "rvv::add: length mismatch (checked upstream)"
    );
    debug_assert_eq!(
        a.len(),
        out.len(),
        "rvv::add: length mismatch (checked upstream)"
    );

    let n = a.len();
    if n == 0 {
        return;
    }
    // Safe scaffolding: on the M3-13 acceptance leg (CI cross-build without
    // qemu-user execution) the RVV kernel is exercised at the shape /
    // dispatch layer via the scalar delegate below. Any host that actually
    // reaches this kernel via runtime dispatch must have probed as RVV 1.0
    // (see `dispatch::table_for` / `features::CpuFeatures::supports`), which
    // is where the SIGILL-guard invariant lives. The inline-asm loop that
    // follows unconditionally emits RVV instructions via `.option arch, +v` —
    // see the module docs for the V clobber caveat.
    let mut i: usize = 0;
    while i < n {
        let remaining = n - i;
        let vl: usize;
        // SAFETY: `a`, `b`, `out` are non-overlapping (Rust aliasing rules on
        // the parameter references), and `a.as_ptr().add(i)` /
        // `b.as_ptr().add(i)` / `out.as_mut_ptr().add(i)` stay within their
        // allocations because `i < n = a.len() = b.len() = out.len()` is the
        // `while` guard. `vsetvli` writes back the actual VL executed, so
        // subsequent iterations advance by the very number of lanes the vector
        // unit processed on this iteration — no element is skipped or
        // double-processed. The `.option arch, +v` fence scopes RVV
        // enablement to this asm block only; the surrounding compilation
        // stays at rv64gc baseline (no V register spill risk, see module
        // docs) — and the dispatch layer never reaches this fn on a host
        // that failed the RVV 1.0 probe.
        unsafe {
            core::arch::asm!(
                ".option push",
                ".option arch, +v",
                "vsetvli {vl}, {avl}, e32, m4, ta, ma",
                "vle32.v v0, ({ap})",
                "vle32.v v4, ({bp})",
                "vfadd.vv v8, v0, v4",
                "vse32.v v8, ({op})",
                ".option pop",
                vl = lateout(reg) vl,
                avl = in(reg) remaining,
                ap = in(reg) a.as_ptr().add(i),
                bp = in(reg) b.as_ptr().add(i),
                op = in(reg) out.as_mut_ptr().add(i),
                options(nostack),
            );
        }
        // `vsetvli` returns 0 only when `avl == 0`, which we already excluded
        // by the `while i < n` guard, so `vl >= 1` is guaranteed here (spec
        // Section "Constraints on Setting vl" of the RVV 1.0 ratified spec).
        debug_assert!(vl >= 1, "vsetvli reported vl=0 on a non-empty AVL");
        i += vl;
    }
}

/// Element-wise `out[i] = a[i] * b[i]`. Scaffold — delegates to
/// [`scalar::mul`] pending an inline-asm rewrite (M4+ follow-up).
pub(crate) fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    scalar::mul(a, b, out);
}

/// Element-wise ReLU. Scaffold — delegates to [`scalar::relu`] (M4+).
pub(crate) fn relu(x: &[f32], out: &mut [f32]) {
    scalar::relu(x, out);
}

/// Element-wise sigmoid. Scaffold — delegates to [`scalar::sigmoid`] (M4+).
pub(crate) fn sigmoid(x: &[f32], out: &mut [f32]) {
    scalar::sigmoid(x, out);
}

/// Element-wise tanh. Scaffold — delegates to [`scalar::tanh`] (M4+).
pub(crate) fn tanh(x: &[f32], out: &mut [f32]) {
    scalar::tanh(x, out);
}

/// Element-wise exact (erf-based) GELU. Scaffold — delegates to
/// [`scalar::gelu`] (M4+).
pub(crate) fn gelu(x: &[f32], out: &mut [f32]) {
    scalar::gelu(x, out);
}

/// Row-major GEMM with optional per-column bias. Scaffold — delegates to
/// [`scalar::gemm`] pending an inline-asm rewrite with vsetvli tile blocking
/// (M4+ follow-up).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    scalar::gemm(m, n, k, a, b, bias, out);
}

/// Row-major matrix-vector product with optional per-row bias. Scaffold —
/// delegates to [`scalar::gemv`] (M4+).
pub(crate) fn gemv(
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    scalar::gemv(m, k, a, x, bias, out);
}

/// Row-wise softmax over the innermost dimension. Scaffold — delegates to
/// [`scalar::softmax`] (M4+).
pub(crate) fn softmax(input: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    scalar::softmax(input, out, rows, cols);
}

/// Row-wise layer normalisation. Scaffold — delegates to
/// [`scalar::layer_norm`] (M4+).
pub(crate) fn layer_norm(
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    scalar::layer_norm(input, out, rows, cols, gamma, beta, eps);
}

/// Fused mel-filterbank + `log10(max(·, floor))` per-frame kernel. Scaffold
/// — matches the scalar reference used by `super::dispatch::scalar_fused_logmel`
/// so the RVV dispatch table entry is bit-identical to the scalar oracle
/// pending an inline-asm rewrite (M4+ follow-up).
pub(crate) fn fused_logmel(
    weights: &[f32],
    power: &[f32],
    n_mels: usize,
    n_bins: usize,
    floor: f32,
    out_log: &mut [f32],
) {
    // Duplicates the module-private scalar reference in `dispatch.rs` — a
    // future inline-asm rewrite will layer over a `vfmacc.vv` accumulator
    // that reuses the mel-band row layout. Kept explicit here (rather than
    // re-exporting the scalar reference) so the M4+ swap is a single-file
    // change.
    assert_eq!(weights.len(), n_mels * n_bins);
    assert_eq!(power.len(), n_bins);
    assert_eq!(out_log.len(), n_mels);
    for m in 0..n_mels {
        let row = &weights[m * n_bins..(m + 1) * n_bins];
        let mut acc = 0.0f32;
        for (w, p) in row.iter().zip(power) {
            acc += w * p;
        }
        let clamped = if acc > floor { acc } else { floor };
        out_log[m] = clamped.log10();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scalar-delegate scaffolds must produce the same result as
    /// `super::scalar::*` — this is the M3-13-T11 differential parity
    /// starting point (bit-exact for elementwise / GEMM / softmax /
    /// layer_norm since the code path IS the scalar path pending M4+ inline
    /// asm rewrites).
    #[test]
    fn rvv_scaffold_kernels_match_scalar_bit_exact() {
        let a: Vec<f32> = (0..17).map(|i| (i as f32) * 0.125 - 1.0).collect();
        let b: Vec<f32> = (0..17).map(|i| (i as f32) * -0.25 + 0.5).collect();

        let mut out_scalar = vec![0.0f32; a.len()];
        let mut out_rvv = vec![0.0f32; a.len()];

        scalar::mul(&a, &b, &mut out_scalar);
        mul(&a, &b, &mut out_rvv);
        assert_eq!(out_scalar, out_rvv, "mul: scalar-delegate scaffold");

        scalar::relu(&a, &mut out_scalar);
        relu(&a, &mut out_rvv);
        assert_eq!(out_scalar, out_rvv, "relu: scalar-delegate scaffold");
    }

    /// Elementwise `add`: on any host this test runs on, it must exercise the
    /// scalar delegate path (the inline-asm path is behind
    /// `cfg(target_arch = "riscv64")`, so on x86-64 / aarch64 the compiler
    /// still needs a working shape check + bit-exact match against
    /// `scalar::add`).
    #[cfg(target_arch = "riscv64")]
    #[test]
    fn rvv_add_matches_scalar_on_riscv64() {
        // Runs the inline-asm RVV path on a riscv64 host (real hart or qemu).
        // On an RVV-less hart the dispatch layer never selects Rvv, so this
        // test itself can only be exercised on a hart where `rvv_v = true`.
        let a: Vec<f32> = (0..37).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32) * -0.125 + 1.0).collect();
        let mut out_scalar = vec![0.0f32; a.len()];
        let mut out_rvv = vec![0.0f32; a.len()];
        scalar::add(&a, &b, &mut out_scalar);
        add(&a, &b, &mut out_rvv);
        // Elementwise f32 add: bit-exact between the scalar left-to-right
        // reduction and the RVV vector add (each output is a single f32
        // add — no reduction rounding order to worry about).
        assert_eq!(
            out_scalar, out_rvv,
            "rvv::add must match scalar::add bit-exactly"
        );
    }

    /// Empty-input safety for the inline-asm `add`. The `while i < n` guard
    /// must prevent us from touching the vector unit when `n == 0`, so this
    /// call must be safe on RVV-less harts too (no vsetvli executed) —
    /// mostly this proves the runtime-empty branch under miri.
    #[cfg(target_arch = "riscv64")]
    #[test]
    fn rvv_add_zero_len_is_no_op() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        add(&a, &b, &mut out); // must not touch vsetvli
    }
}
