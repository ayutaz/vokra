//! ARM64 bf16 (BFMMLA) matmul core (M4-17-T17).
//!
//! ARMv8.6 bf16 — **this dev machine (Apple M1) cannot execute it** (bf16 is
//! Apple M2+ / Graviton3-class), so the differential runs under
//! runtime-detect skip locally and on owner silicon for real (M4-17-T24; no
//! fabricated green). `bfmmla` computes `C[2][2] += A[2][4] · B[2][4]ᵀ` over
//! bf16 inputs with f32 accumulation.
//!
//! # Numerics posture (ADR M4-17 §(f))
//!
//! The exact internal pair-accumulation rounding of `bfmmla` (FPCR.EBF
//! modes) is NOT asserted by the parity tests — no local bf16 silicon and no
//! primary-source verification here, so the tests bound this kernel by the
//! **architectural bf16 band** (mantissa-8-bit input rounding × reduction
//! length, input-derived) against both the f32 GEMM and the scalar bf16
//! emulation. Tightening is deferred to the owner silicon run.
//!
//! # Why inline asm (ADR M4-17 §(g))
//!
//! `bfloat16x8_t` does not exist on rustc 1.95 stable (verified by compile
//! probe) → `core::arch::asm!` with an `.arch_extension bf16` fence. Reached
//! only after the `CpuFeatures::supports(IsaPath::NeonBf16)` gate.

/// One 2x2 output tile: `c[0..4] (f32, lanes [c00, c01, c10, c11]) +=
/// Σ_chunks A[2][4]·B[2][4]ᵀ` over `k4 >= 1` prepacked 4-element bf16
/// chunks.
///
/// `a_tiles` / `b_tiles` hold `k4` interleaved 16-byte tiles: chunk `c` of
/// `a_tiles` is `[a_row0[4c..4c+4], a_row1[4c..4c+4]]` (8 bf16 = 16 bytes),
/// and of `b_tiles` `[x_col0[4c..4c+4], x_col1[4c..4c+4]]`.
pub(crate) fn bfmmla_tile(c: &mut [f32; 4], a_tiles: &[u16], b_tiles: &[u16], k4: usize) {
    debug_assert!(k4 >= 1);
    debug_assert_eq!(a_tiles.len(), 8 * k4);
    debug_assert_eq!(b_tiles.len(), 8 * k4);
    // SAFETY: `a_tiles` / `b_tiles` each hold `8 * k4` u16 = `16 * k4`
    // bytes, so the `k4` post-incremented 16-byte `ld1` loads stay in
    // bounds; `c` is a 16-byte local for the bracketing `ld1`/`st1`. The
    // `bfmmla` encoding is reached only after the caller's
    // `CpuFeatures::supports(IsaPath::NeonBf16)` gate confirmed ARMv8.6 bf16
    // (SIGILL guard); the `.arch_extension` fence scopes the extension;
    // v0-v2 are clobbered.
    unsafe {
        core::arch::asm!(
            ".arch_extension bf16",
            "ld1 {{v0.4s}}, [{cp}]",
            "3:",
            "ld1 {{v1.8h}}, [{ap}], #16",
            "ld1 {{v2.8h}}, [{bp}], #16",
            "bfmmla v0.4s, v1.8h, v2.8h",
            "subs {kc}, {kc}, #1",
            "b.ne 3b",
            "st1 {{v0.4s}}, [{cp}]",
            ".arch_extension nobf16",
            cp = in(reg) c.as_mut_ptr(),
            ap = inout(reg) a_tiles.as_ptr() => _,
            bp = inout(reg) b_tiles.as_ptr() => _,
            kc = inout(reg) k4 => _,
            out("v0") _, out("v1") _, out("v2") _,
            options(nostack),
        );
    }
}
