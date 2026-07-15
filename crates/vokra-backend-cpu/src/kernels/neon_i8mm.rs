//! ARM64 i8mm (SMMLA) INT8 2x2-tile matmul core (M4-17-T16).
//!
//! ARMv8.6 i8mm — Apple M2+ per FR-BE-01; **this dev machine (Apple M1)
//! cannot execute it**, so the differential runs under runtime-detect skip
//! locally and on owner silicon for real (M4-17-T24, ADR M4-17 §(f); no
//! fabricated green). `smmla` computes `C[2][2] += A[2][8] · B[2][8]ᵀ` over
//! signed bytes; two chained `smmla` cover one 16-element scale group, giving
//! the exact i32 group sums for a (2 weight rows × 2 activation vectors)
//! tile that [`super::kquant`]'s shared combine consumes — bit-identical to
//! the scalar-int8 reference by integer exactness (ADR M4-17 §(e)).
//!
//! # Why inline asm (ADR M4-17 §(g))
//!
//! `vmmlaq_s32` is unstable on rustc 1.95 (`stdarch_neon_i8mm`,
//! rust-lang/rust#117223 — verified by compile probe) → `core::arch::asm!`
//! with an `.arch_extension i8mm` fence (M3-13 RVV precedent). Reached only
//! after the `CpuFeatures::supports(IsaPath::NeonI8mm)` gate (SIGILL guard).

/// One SMMLA group step: given the two weight rows' bytes for one
/// 16-element group (`a0`, `a1`, each 16 bytes = the group's quants for row
/// 0 / row 1) and the two activation vectors' bytes for the same group
/// (`x0`, `x1`), writes the four exact i32 dot sums
/// `[r0·x0, r0·x1, r1·x0, r1·x1]` into `out`.
///
/// Operand packing: `smmla`'s `A` operand is a row-major 2x8 tile, so the
/// group is processed as two 8-byte halves — `A = [a_row0[h], a_row1[h]]`,
/// `B = [x0[h], x1[h]]` for half `h ∈ {0, 1}` — accumulated into one `.4s`
/// register.
pub(crate) fn smmla_group_tile(
    a0: &[u8; 16],
    a1: &[u8; 16],
    x0: &[i8; 16],
    x1: &[i8; 16],
    out: &mut [i32; 4],
) {
    // Stage the two row-major 2x8 tiles (weights are 0..=63, identical in u8
    // and i8, so the byte copy is the whole "conversion").
    let mut a_tile = [0u8; 32];
    let mut b_tile = [0i8; 32];
    for h in 0..2 {
        a_tile[16 * h..16 * h + 8].copy_from_slice(&a0[8 * h..8 * h + 8]);
        a_tile[16 * h + 8..16 * h + 16].copy_from_slice(&a1[8 * h..8 * h + 8]);
        b_tile[16 * h..16 * h + 8].copy_from_slice(&x0[8 * h..8 * h + 8]);
        b_tile[16 * h + 8..16 * h + 16].copy_from_slice(&x1[8 * h..8 * h + 8]);
    }
    // SAFETY: `a_tile` / `b_tile` are 32-byte locals, so both pairs of
    // 16-byte `ld1` loads are in bounds, and `out` is a 16-byte local array
    // for the single `st1` store. The `smmla` encoding is reached only after
    // the caller's `CpuFeatures::supports(IsaPath::NeonI8mm)` gate confirmed
    // ARMv8.6 i8mm on this host (dispatch invariant / SIGILL guard). The
    // `.arch_extension` fence scopes the extension to this block; v0-v2 are
    // declared clobbered.
    unsafe {
        core::arch::asm!(
            ".arch_extension i8mm",
            "movi v2.4s, #0",
            "ld1 {{v0.16b}}, [{ap}]",
            "ld1 {{v1.16b}}, [{bp}]",
            "smmla v2.4s, v0.16b, v1.16b",
            "ld1 {{v0.16b}}, [{ap2}]",
            "ld1 {{v1.16b}}, [{bp2}]",
            "smmla v2.4s, v0.16b, v1.16b",
            "st1 {{v2.4s}}, [{op}]",
            ".arch_extension noi8mm",
            ap = in(reg) a_tile.as_ptr(),
            bp = in(reg) b_tile.as_ptr(),
            ap2 = in(reg) a_tile[16..].as_ptr(),
            bp2 = in(reg) b_tile[16..].as_ptr(),
            op = in(reg) out.as_mut_ptr(),
            out("v0") _, out("v1") _, out("v2") _,
            options(nostack),
        );
    }
}
