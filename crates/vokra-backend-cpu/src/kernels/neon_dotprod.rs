//! ARM64 dotprod (SDOT) INT8 group-sum core (M4-17-T15).
//!
//! ARMv8.2-DotProd (`sdot`): Cortex-A55/A75 2017 initial cores, Apple A13+
//! (FR-BE-01) — the ARM INT8 main path, mirroring the x86 VNNI tiers. This
//! module computes the exact per-16-element-group integer sums
//! `isum[g] = Σ_{t<16} q[16g+t] · x[16g+t]` (i32, exact) that
//! [`super::kquant`]'s shared scalar combine turns into f32, so the dotprod
//! path is bit-identical to the scalar-int8 reference (ADR M4-17 §(e)).
//!
//! # Why inline asm (ADR M4-17 §(g))
//!
//! The `vdotq_s32` intrinsic is unstable on rustc 1.95
//! (`stdarch_neon_dotprod`, rust-lang/rust#117224 — verified by compile
//! probe), so this uses the M3-13 RVV precedent: a `core::arch::asm!` block
//! fenced by `.arch_extension dotprod` … `.arch_extension nodotprod`, keeping
//! the surrounding binary at the ARMv8-A NEON baseline. The `sdot` encoding
//! executes only after `CpuFeatures::detect().neon_dotprod` confirmed the
//! extension (the caller's `supports` gate = the SIGILL guard; NFR-RL-05
//! JIT-free — this is a statically assembled instruction).

/// Per-group SDOT sums: `sums[g] = Σ_{t<16} q[16g+t] · x[16g+t]`.
///
/// `q` holds the unsigned K-quant values (0..=63, always non-negative so the
/// signed×signed `sdot` is exact); `x` the Q8 activations. Lengths must be
/// equal multiples of 16 with `sums.len() * 16 == q.len()` (asserted).
///
/// The weight bytes are staged through an `i8` reinterpretation: values
/// 0..=63 are identical in `u8` and `i8`, so no conversion is involved.
pub(crate) fn dotprod_group_sums(q: &[u8], x: &[i8], sums: &mut [i32]) {
    assert_eq!(q.len(), x.len(), "dotprod_group_sums length mismatch");
    assert_eq!(q.len() % 16, 0, "dotprod_group_sums needs 16-byte groups");
    assert_eq!(sums.len() * 16, q.len(), "dotprod_group_sums sums mismatch");
    for (g, s) in sums.iter_mut().enumerate() {
        let qp = q[16 * g..].as_ptr();
        let xp = x[16 * g..].as_ptr();
        let out: i32;
        // SAFETY: `16 * g + 16 <= q.len() == x.len()` by the asserts above,
        // so both 16-byte `ld1` loads stay inside their slices. The `sdot`
        // encoding is reached only after the caller's
        // `CpuFeatures::supports(IsaPath::NeonDotprod)` gate confirmed
        // ARMv8.2-DotProd on this host (dispatch invariant — same
        // SIGILL-guard structure as the RVV kernel). The
        // `.arch_extension` fence scopes the extension to this block; v0-v2
        // are declared clobbered; no stack or memory beyond the loads is
        // touched (`readonly`).
        unsafe {
            core::arch::asm!(
                ".arch_extension dotprod",
                "ld1 {{v0.16b}}, [{qp}]",
                "ld1 {{v1.16b}}, [{xp}]",
                "movi v2.4s, #0",
                "sdot v2.4s, v0.16b, v1.16b",
                "addv s2, v2.4s",
                "fmov {out:w}, s2",
                ".arch_extension nodotprod",
                qp = in(reg) qp,
                xp = in(reg) xp,
                out = lateout(reg) out,
                out("v0") _, out("v1") _, out("v2") _,
                options(nostack, readonly),
            );
        }
        *s = out;
    }
}
