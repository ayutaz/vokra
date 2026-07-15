//! ARM64 fp16 arithmetic GEMM core (M4-17-T14).
//!
//! ARMv8.2 fp16 arithmetic (Cortex-A75+ 2018+, FR-BE-01) — genuine
//! half-precision compute: the accumulator chain is fp16 `fmla .8h` (fused
//! multiply-add, one rounding per step), NOT an fp16-storage/f32-accumulate
//! hybrid. This is the **opt-in** tier `kernels::gemm_fp16_on` exposes
//! (fp16's 10-bit mantissa is avoided for f32-precision ops — ADR M4-17
//! §(b)-2); its parity oracle is the structurally identical scalar emulation
//! in [`super::kquant`] (fp16 FMA emulated through exact f64 products, ±2
//! ulp band — ADR M4-17 §(f)).
//!
//! # Why inline asm (ADR M4-17 §(g))
//!
//! `float16x8_t` arithmetic intrinsics are unstable on rustc 1.95 (verified
//! by compile probe) → `core::arch::asm!` with an `.arch_extension fp16`
//! fence (M3-13 RVV precedent). The `fmla .8h` encoding executes only after
//! `CpuFeatures::supports(IsaPath::NeonFp16)` confirmed ARMv8.2 fp16 on this
//! host (SIGILL guard). This Apple M1 dev machine supports fp16, so the
//! differential runs for real locally (M4-17 spec T14).

/// One output-row strip: `acc[0..8] (f16) += a_scalar (f16) * b[l*ldb + j..+8]
/// (f16)` chained over `l = 0..k` with fp16 FMA per lane.
///
/// `acc` is an 8-lane f16 accumulator (bit patterns in `u16`), `a_col` the
/// column of broadcast scalars `a[i, 0..k]` (f16), `b_strip` the base of the
/// f16 `b` matrix strip at column `j` (row stride `ldb` **elements**).
/// `k >= 1` (guarded by the caller).
pub(crate) fn fp16_fma_row_strip(
    acc: &mut [u16; 8],
    a_col: &[u16],
    b_strip: *const u16,
    ldb: usize,
    k: usize,
) {
    debug_assert!(k >= 1);
    debug_assert_eq!(a_col.len(), k);
    let byte_stride = ldb * 2;
    // SAFETY: the caller (`super::kquant::gemm_fp16_on`) guarantees the
    // padded f16 `b` buffer holds `k` rows of `ldb` elements starting at
    // `b_strip` with 8 in-bounds lanes per row (columns are zero-padded to a
    // multiple of 8), and `a_col` has exactly `k` elements — so the `k`
    // iterations of `ld1 {v2.8h}` (advancing by `byte_stride`) and `ld1r`
    // (advancing by 2) stay in bounds; `acc` is an 8-lane local for the
    // bracketing `ld1`/`st1`. The fp16 encodings are reached only after the
    // `CpuFeatures::supports(IsaPath::NeonFp16)` gate (SIGILL guard); the
    // `.arch_extension` fence scopes the extension; v0-v2 are clobbered.
    unsafe {
        core::arch::asm!(
            ".arch_extension fp16",
            "ld1 {{v0.8h}}, [{accp}]",
            "2:",
            "ld1r {{v1.8h}}, [{ap}], #2",
            "ld1 {{v2.8h}}, [{bp}]",
            "add {bp}, {bp}, {bstride}",
            "fmla v0.8h, v1.8h, v2.8h",
            "subs {kc}, {kc}, #1",
            "b.ne 2b",
            "st1 {{v0.8h}}, [{accp}]",
            ".arch_extension nofp16",
            accp = in(reg) acc.as_mut_ptr(),
            ap = inout(reg) a_col.as_ptr() => _,
            bp = inout(reg) b_strip => _,
            bstride = in(reg) byte_stride,
            kc = inout(reg) k => _,
            out("v0") _, out("v1") _, out("v2") _,
            options(nostack),
        );
    }
}
