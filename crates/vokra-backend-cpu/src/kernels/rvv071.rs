//! RISC-V RVV **draft 0.7.1** kernels (M4-08-T07/T08) — the T-Head XuanTie
//! C910/C906 fallback tier (LicheePi 4A TH1520 = Tier 1, Milk-V Duo Sophgo
//! CV1800B = Tier 2 / Silero VAD only, NFR-PT-03).
//!
//! Compiled only on `target_arch = "riscv64"`. This is a **runtime dispatch
//! scaffold** in the exact mould of [`super::rvv`] (M3-13): **`add` is the
//! demonstrator that emits real RVV 0.7.1 instructions**; the remaining
//! kernels forward to the portable scalar reference in [`super::scalar`]
//! pending M4+/M5 follow-up rewrites. The runtime dispatch table routes
//! through this module only on a host that probed as a 0.7.1 hart
//! (`CpuFeatures::rvv_071`, ADR M4-08 §b), which the CI cross-build +
//! host-side differential tests verify. Execution of the 0.7.1 words is
//! only possible on real T-Head silicon — upstream qemu has no
//! xtheadvector emulation (ADR M4-08 §T01) — so on-device validation is
//! the owner track (M4-08-T15/T16).
//!
//! # Why a separate file from `rvv.rs`?
//!
//! RVV 0.7.1 and ratified RVV 1.0 are **instruction-encoding incompatible**
//! (riscv-v-spec tag 0.7.1 vs 1.0): the 0.7.1 `vtype` has no ta/ma bits and
//! packs `vsew` at bits 4:2 / `vlmul` at bits 1:0, unit-stride SEW loads
//! use width=111 (`vle.v`) where 1.0 encodes `vle64.v`, and so on. No
//! instruction bytes can be shared, so the tiers live in separate files and
//! the dispatch layer treats them as peers (ADR M4-08 §d).
//!
//! # Why `.insn` raw words instead of mnemonics or `.option arch`?
//!
//! Probed on the pinned stable rustc (ADR M4-08 §T01, 2026-07-15): LLVM's
//! integrated assembler has **no `xtheadvector` target feature** (only the
//! non-vector `xthead*` set plus `xtheadvdot`), rejects
//! `.option arch, +xtheadvector`, and does not recognise `th.`-prefixed
//! vector mnemonics. The only zero-dep, stable-rustc emit path is the
//! standard `.insn <length>, <value>` directive with the instruction words
//! encoded by the `const fn encode_*` helpers below, each transcribed
//! field-by-field from the riscv-v-spec tag 0.7.1 format diagrams
//! (`vcfg-format.adoc` / `vmem-format.adoc` / `inst-table.adoc` /
//! `vtype-format.adoc`). Compile-time `const _: () = assert!(…)` pins every
//! word, so the CI riscv64 cross-build re-verifies the encodings on every
//! run. No T-Head sample code was copied (NFR-LC-04).
//!
//! # V register clobber caveat (0.7.1 edition)
//!
//! stable rustc has no `vreg` register class, so the `asm!` block below
//! cannot explicitly clobber v0..v11. This is sound as long as the
//! **surrounding compilation stays at the rv64gc baseline** — no
//! `-C target-feature=+v` (or any 0.7.1-era equivalent) in a global
//! `RUSTFLAGS` / `.cargo/config*`. At baseline the register allocator
//! cannot spill values into V registers, so the vector unit is invisible to
//! Rust and cannot be corrupted by our raw-word sequences. Unlike `rvv.rs`
//! this module does not even use `.option arch` — the 0.7.1 encodings are
//! sealed inside `.insn` words, so the assembler subset never changes
//! either. The CI `riscv-cross-build` job asserts the baseline stays
//! unpolluted (M4-08-T11).

use crate::kernels::scalar;

// ---------------------------------------------------------------------------
// RVV 0.7.1 instruction words (M4-08-T08).
//
// Field layouts transcribed from riscv-v-spec tag 0.7.1:
//
//   vsetvli (vcfg-format.adoc):
//     31    30:20        19:15  14:12  11:7  6:0
//     0   | zimm[10:0] | rs1  | 111  | rd  | 1010111
//   vtype/vtypei (vtype-format.adoc): vlmul[1:0]=bits 1:0, vsew[2:0]=bits
//     4:2, vediv[1:0]=bits 6:5 (EDIV extension, 0 in the base spec). There
//     are NO ta/ma bits in 0.7.1 — those are 0.9+/1.0 additions.
//   unit-stride load (vmem-format.adoc, LOAD-FP):
//     31:29  28:26  25   24:20   19:15  14:12   11:7  6:0
//     nf   | mop  | vm | lumop | rs1  | width | vd  | 0000111
//   unit-stride store (vmem-format.adoc, STORE-FP):
//     nf   | mop  | vm | sumop | rs1  | width | vs3 | 0100111
//     width (v-spec §7.3): 111 = SEW-sized ("VxE" = vle.v / vse.v);
//     mop (v-spec §7.4): 000 = zero-extended unit-stride (VLE/VSE);
//     lumop/sumop 00000 = plain unit-stride; vm=1 = unmasked.
//   OP-V arithmetic (inst-table.adoc):
//     31:26    25   24:20  19:15  14:12    11:7  6:0
//     funct6 | vm | vs2  | vs1  | funct3 | vd  | 1010111
//     vfadd.vv: funct6=000000, funct3=OPFVV=001.
//
// The 1.0-vs-0.7.1 contrast that forbids reusing `rvv.rs` words: 0.7.1
// `vle.v` (width=111) would decode as `vle64.v` on a 1.0 hart, and the
// vsetvli vtypei for e32,m4 is 0x0A here vs 0xD2 (e32,m4,ta,ma) in 1.0.
// ---------------------------------------------------------------------------

/// RISC-V standard opcodes shared by both vector generations (base ISA).
const OPCODE_OP_V: u32 = 0b101_0111; // OP-V major opcode (vsetvli + arithmetic)
const OPCODE_LOAD_FP: u32 = 0b000_0111;
const OPCODE_STORE_FP: u32 = 0b010_0111;

/// 0.7.1 vtype field values for the demonstrator config `e32, m4`
/// (v-spec 0.7.1 "Vector standard element width" / "Vector Register
/// Grouping" tables: SEW=32 → vsew=010, LMUL=4 → vlmul=10).
const VSEW_E32: u32 = 0b010;
const VLMUL_M4: u32 = 0b10;

/// ABI GPR indices used by the pinned-register asm block below
/// (RISC-V psABI: a0=x10, a1=x11, a2=x12, a3=x13, t0=x5).
const GPR_A0: u32 = 10;
const GPR_A1: u32 = 11;
const GPR_A2: u32 = 12;
const GPR_A3: u32 = 13;
const GPR_T0: u32 = 5;

/// Vector register operands (LMUL=4 ⇒ groups of 4: v0..v3 / v4..v7 /
/// v8..v11 — same allocation as the `rvv.rs` 1.0 demonstrator).
const VREG_V0: u32 = 0;
const VREG_V4: u32 = 4;
const VREG_V8: u32 = 8;

/// Encodes the 0.7.1 `vsetvli rd, rs1, vtypei` word. `vtypei` is built from
/// the 0.7.1 vtype layout (vsew at bits 4:2, vlmul at bits 1:0; vediv left
/// 0 = d1) — NOT the 1.0 layout, which has vsew at 5:3 and ta/ma at 6/7.
const fn encode_vsetvli_071(rd: u32, rs1: u32, vsew: u32, vlmul: u32) -> u32 {
    let vtypei = (vsew << 2) | vlmul;
    // bit 31 = 0 selects the immediate (vsetvli) form; funct3 = 111.
    (vtypei << 20) | (rs1 << 15) | (0b111 << 12) | (rd << 7) | OPCODE_OP_V
}

/// Encodes the 0.7.1 unmasked unit-stride SEW load `vle.v vd, (rs1)`
/// (nf=000, mop=000 zero-extended unit-stride, vm=1, lumop=00000,
/// width=111 = SEW-sized).
const fn encode_vle_v_071(vd: u32, rs1: u32) -> u32 {
    (1 << 25) | (rs1 << 15) | (0b111 << 12) | (vd << 7) | OPCODE_LOAD_FP
}

/// Encodes the 0.7.1 unmasked unit-stride SEW store `vse.v vs3, (rs1)`
/// (nf=000, mop=000 unit-stride, vm=1, sumop=00000, width=111).
const fn encode_vse_v_071(vs3: u32, rs1: u32) -> u32 {
    (1 << 25) | (rs1 << 15) | (0b111 << 12) | (vs3 << 7) | OPCODE_STORE_FP
}

/// Encodes the 0.7.1 unmasked `vfadd.vv vd, vs2, vs1` (funct6=000000,
/// vm=1, funct3=OPFVV=001).
const fn encode_vfadd_vv_071(vd: u32, vs2: u32, vs1: u32) -> u32 {
    (1 << 25) | (vs2 << 20) | (vs1 << 15) | (0b001 << 12) | (vd << 7) | OPCODE_OP_V
}

/// `vsetvli t0, a0, e32, m4` (0.7.1 vtypei = 0x0A — contrast 1.0's 0xD2).
const VSETVLI_T0_A0_E32_M4: u32 = encode_vsetvli_071(GPR_T0, GPR_A0, VSEW_E32, VLMUL_M4);
/// `vle.v v0, (a1)` — loads the `a` operand tile.
const VLE_V_V0_A1: u32 = encode_vle_v_071(VREG_V0, GPR_A1);
/// `vle.v v4, (a2)` — loads the `b` operand tile.
const VLE_V_V4_A2: u32 = encode_vle_v_071(VREG_V4, GPR_A2);
/// `vfadd.vv v8, v0, v4`.
const VFADD_VV_V8_V0_V4: u32 = encode_vfadd_vv_071(VREG_V8, VREG_V0, VREG_V4);
/// `vse.v v8, (a3)` — stores the result tile.
const VSE_V_V8_A3: u32 = encode_vse_v_071(VREG_V8, GPR_A3);

// Compile-time pins of the exact instruction words (hand-assembled once
// from the 0.7.1 format diagrams above). Evaluated by every build of this
// module — including the CI riscv64 cross-build — so an encode_* regression
// can never reach a binary (M4-08-T08 completion gate).
const _: () = assert!(VSETVLI_T0_A0_E32_M4 == 0x00A5_72D7);
const _: () = assert!(VLE_V_V0_A1 == 0x0205_F007);
const _: () = assert!(VLE_V_V4_A2 == 0x0206_7207);
const _: () = assert!(VFADD_VV_V8_V0_V4 == 0x0202_1457);
const _: () = assert!(VSE_V_V8_A3 == 0x0206_F427);

/// Element-wise `out[i] = a[i] + b[i]` via RVV **0.7.1** vector float-add.
///
/// Precondition (checked by the caller in `super::kernels`): `a.len() ==
/// b.len() == out.len()`.
///
/// # RVV 0.7.1 body
///
/// The inner loop is the 0.7.1 rendition of the canonical vsetvli / load /
/// fadd / store stream: `vsetvli t0, a0, e32, m4` → `vle.v v0,(a1)` /
/// `vle.v v4,(a2)` → `vfadd.vv v8, v0, v4` → `vse.v v8,(a3)`, emitted as
/// `.insn` raw words (module docs). Per the 0.7.1 `vl` rule (v-spec §6.1:
/// `vl = AVL` when `AVL ≤ VLMAX`; `vl ≥ ceil(AVL/2)` when `AVL < 2·VLMAX`;
/// `vl = VLMAX` otherwise) the returned `vl` may be *smaller* than
/// `min(AVL, VLMAX)` on the middle band — the loop therefore advances by
/// exactly the `vl` the hart reports, which is correct under both the 0.7.1
/// and 1.0 rules, and `vl ≥ 1` is guaranteed whenever `AVL ≥ 1`.
///
/// # Safety
///
/// The dispatch layer only ever routes this kernel on a host where
/// [`crate::features::CpuFeatures::detect`] reports `rvv_071 = true` (the
/// `xtheadvector` isa token or the vendor-kernel `cpu-vector : 0.7.1`
/// signal, ADR M4-08 §b). On any other riscv64 hart these raw words are
/// illegal (or worse, 1.0-decoded) instructions; that path is unreachable
/// in normal use because [`crate::dispatch::table_for`] rejects
/// `IsaPath::Rvv071` via `CpuFeatures::supports` with an explicit
/// `BackendUnavailable` error (FR-EX-08 principle), and `best_isa` only
/// auto-selects the tier on the kernel-managed signal (`rvv_071_auto`).
pub(crate) fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "rvv071::add: length mismatch (checked upstream)"
    );
    debug_assert_eq!(
        a.len(),
        out.len(),
        "rvv071::add: length mismatch (checked upstream)"
    );

    let n = a.len();
    if n == 0 {
        return;
    }
    let mut i: usize = 0;
    while i < n {
        let remaining = n - i;
        let vl: usize;
        // SAFETY: (a) `a`, `b`, `out` are non-overlapping (Rust aliasing
        // rules on the parameter references); (b) `a.as_ptr().add(i)` /
        // `b.as_ptr().add(i)` / `out.as_mut_ptr().add(i)` stay within their
        // allocations because `i < n = a.len() = b.len() = out.len()` is the
        // `while` guard, and the hart never touches more than `vl ≤
        // remaining = n - i` elements per step (vsetvli's AVL contract);
        // (c) the dispatch layer reaches this fn only on a host that probed
        // `rvv_071 = true` (`table_for` validator + `supports` gate = the
        // SIGILL-guard invariant; see the fn docs) so the raw 0.7.1 words
        // are executable; (d) the words are sealed inside `.insn`
        // directives — no `.option arch`, no global target-feature — so the
        // surrounding rv64gc baseline (and the V-clobber caveat premise in
        // the module docs) is preserved. `vsetvli` writes back the actual
        // VL executed, so subsequent iterations advance by the very number
        // of lanes processed — no element is skipped or double-processed.
        unsafe {
            core::arch::asm!(
                // vsetvli t0, a0, e32, m4   (0.7.1 vtypei layout)
                ".insn 4, {vsetvli}",
                // vle.v v0, (a1) ; vle.v v4, (a2)   (unit-stride SEW load)
                ".insn 4, {vle_a}",
                ".insn 4, {vle_b}",
                // vfadd.vv v8, v0, v4
                ".insn 4, {vfadd}",
                // vse.v v8, (a3)   (unit-stride SEW store)
                ".insn 4, {vse_o}",
                vsetvli = const VSETVLI_T0_A0_E32_M4,
                vle_a = const VLE_V_V0_A1,
                vle_b = const VLE_V_V4_A2,
                vfadd = const VFADD_VV_V8_V0_V4,
                vse_o = const VSE_V_V8_A3,
                in("a0") remaining,
                in("a1") a.as_ptr().add(i),
                in("a2") b.as_ptr().add(i),
                in("a3") out.as_mut_ptr().add(i),
                lateout("t0") vl,
                options(nostack),
            );
        }
        // 0.7.1 §6.1: vl = 0 only when AVL = 0, which the `while i < n`
        // guard excludes — vl >= 1 holds even under the 0.7.1
        // even-distribution allowance (vl >= ceil(AVL/2) >= 1).
        debug_assert!(vl >= 1, "vsetvli reported vl=0 on a non-empty AVL");
        i += vl;
    }
}

/// Element-wise `out[i] = a[i] * b[i]`. Scaffold — delegates to
/// [`scalar::mul`] pending a 0.7.1 rewrite (M4+/M5 follow-up).
pub(crate) fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    scalar::mul(a, b, out);
}

/// Element-wise ReLU. Scaffold — delegates to [`scalar::relu`] (M4+/M5).
pub(crate) fn relu(x: &[f32], out: &mut [f32]) {
    scalar::relu(x, out);
}

/// Element-wise sigmoid. Scaffold — delegates to [`scalar::sigmoid`]
/// (M4+/M5).
pub(crate) fn sigmoid(x: &[f32], out: &mut [f32]) {
    scalar::sigmoid(x, out);
}

/// Element-wise tanh. Scaffold — delegates to [`scalar::tanh`] (M4+/M5).
pub(crate) fn tanh(x: &[f32], out: &mut [f32]) {
    scalar::tanh(x, out);
}

/// Element-wise exact (erf-based) GELU. Scaffold — delegates to
/// [`scalar::gelu`] (M4+/M5).
pub(crate) fn gelu(x: &[f32], out: &mut [f32]) {
    scalar::gelu(x, out);
}

/// Row-major GEMM with optional per-column bias. Scaffold — delegates to
/// [`scalar::gemm`] pending a 0.7.1 rewrite (M4+/M5 follow-up).
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
/// delegates to [`scalar::gemv`] (M4+/M5).
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
/// [`scalar::softmax`] (M4+/M5).
pub(crate) fn softmax(input: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    scalar::softmax(input, out, rows, cols);
}

/// Row-wise layer normalisation. Scaffold — delegates to
/// [`scalar::layer_norm`] (M4+/M5).
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
/// — matches the scalar reference used by
/// `super::dispatch::scalar_fused_logmel` so the Rvv071 dispatch table entry
/// is bit-identical to the scalar oracle pending a 0.7.1 rewrite (M4+/M5).
/// Kept as an explicit duplicate (rather than a re-export) so the future
/// swap is a single-file change — same policy as [`super::rvv`].
pub(crate) fn fused_logmel(
    weights: &[f32],
    power: &[f32],
    n_mels: usize,
    n_bins: usize,
    floor: f32,
    out_log: &mut [f32],
) {
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

    /// The encode_* helpers must reproduce the hand-assembled 0.7.1 words
    /// (also pinned at compile time by the `const _` asserts above — this
    /// runtime duplicate keeps the values visible in test output and
    /// documents the field decomposition).
    #[test]
    fn rvv071_instruction_words_match_hand_assembly() {
        // vsetvli t0(x5), a0(x10), e32,m4:
        //   zimm = vsew(010)<<2 | vlmul(10) = 0x0A
        //   word = 0x0A<<20 | 10<<15 | 0b111<<12 | 5<<7 | 0b1010111
        assert_eq!(VSETVLI_T0_A0_E32_M4, 0x00A5_72D7);
        // vle.v v0,(a1): vm=1<<25 | 11<<15 | width(111)<<12 | 0<<7 | 0000111
        assert_eq!(VLE_V_V0_A1, 0x0205_F007);
        // vle.v v4,(a2)
        assert_eq!(VLE_V_V4_A2, 0x0206_7207);
        // vfadd.vv v8,v0,v4: vm=1<<25 | vs2(0)<<20 | vs1(4)<<15 |
        //   OPFVV(001)<<12 | vd(8)<<7 | 1010111
        assert_eq!(VFADD_VV_V8_V0_V4, 0x0202_1457);
        // vse.v v8,(a3)
        assert_eq!(VSE_V_V8_A3, 0x0206_F427);
    }

    /// The 0.7.1 words must differ from their RVV 1.0 counterparts — the
    /// whole reason this tier exists. 1.0 reference words computed from the
    /// ratified-spec layout (vtypei e32,m4,ta,ma = 0xD2; unit-stride 32-bit
    /// load/store width = 110): reusing `rvv.rs` encodings here would be a
    /// silent SIGILL/misdecode on a 0.7.1 hart.
    #[test]
    fn rvv071_words_differ_from_rvv10_encodings() {
        // 1.0 vsetvli t0, a0, e32, m4, ta, ma (vtypei = 0xD2).
        let vsetvli_10 = (0xD2u32 << 20) | (10 << 15) | (0b111 << 12) | (5 << 7) | 0b101_0111;
        assert_ne!(VSETVLI_T0_A0_E32_M4, vsetvli_10);
        // 1.0 vle32.v v0,(a1) uses width=110 (not 111 = SEW in 0.7.1).
        let vle32_10 = (1u32 << 25) | (11 << 15) | (0b110 << 12) | 0b000_0111;
        assert_ne!(VLE_V_V0_A1, vle32_10);
        // In fact the 0.7.1 vle.v word IS the 1.0 vle64.v encoding — the
        // starkest incompatibility witness (same bits, different meaning).
        let vle64_10 = (1u32 << 25) | (11 << 15) | (0b111 << 12) | 0b000_0111;
        assert_eq!(VLE_V_V0_A1, vle64_10);
        // vfadd.vv happens to share funct6/funct3 across generations.
        let vfadd_10 = (1u32 << 25) | (4 << 15) | (0b001 << 12) | (8 << 7) | 0b101_0111;
        assert_eq!(VFADD_VV_V8_V0_V4, vfadd_10);
    }

    /// The scalar-delegate scaffolds must produce the same result as
    /// `super::scalar::*` — bit-exact, since the code path IS the scalar
    /// path pending the 0.7.1 rewrites (M4-08-T07 differential baseline,
    /// mirroring `rvv_scaffold_kernels_match_scalar_bit_exact`).
    #[test]
    fn rvv071_scaffold_kernels_match_scalar_bit_exact() {
        let a: Vec<f32> = (0..17).map(|i| (i as f32) * 0.125 - 1.0).collect();
        let b: Vec<f32> = (0..17).map(|i| (i as f32) * -0.25 + 0.5).collect();

        let mut out_scalar = vec![0.0f32; a.len()];
        let mut out_rvv071 = vec![0.0f32; a.len()];

        scalar::mul(&a, &b, &mut out_scalar);
        mul(&a, &b, &mut out_rvv071);
        assert_eq!(out_scalar, out_rvv071, "mul: scalar-delegate scaffold");

        scalar::relu(&a, &mut out_scalar);
        relu(&a, &mut out_rvv071);
        assert_eq!(out_scalar, out_rvv071, "relu: scalar-delegate scaffold");

        scalar::sigmoid(&a, &mut out_scalar);
        sigmoid(&a, &mut out_rvv071);
        assert_eq!(out_scalar, out_rvv071, "sigmoid: scalar-delegate scaffold");

        scalar::tanh(&a, &mut out_scalar);
        tanh(&a, &mut out_rvv071);
        assert_eq!(out_scalar, out_rvv071, "tanh: scalar-delegate scaffold");

        scalar::gelu(&a, &mut out_scalar);
        gelu(&a, &mut out_rvv071);
        assert_eq!(out_scalar, out_rvv071, "gelu: scalar-delegate scaffold");

        let (m, n, k) = (3usize, 5usize, 4usize);
        let ga: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let gb: Vec<f32> = (0..k * n).map(|i| (i as f32) * -0.05 + 0.2).collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let mut og = vec![0.0f32; m * n];
        let mut rg = vec![0.0f32; m * n];
        scalar::gemm(m, n, k, &ga, &gb, Some(&bias), &mut og);
        gemm(m, n, k, &ga, &gb, Some(&bias), &mut rg);
        assert_eq!(og, rg, "gemm: scalar-delegate scaffold");

        let mut ov = vec![0.0f32; m];
        let mut rv = vec![0.0f32; m];
        let x: Vec<f32> = (0..k).map(|i| i as f32 * 0.3 - 0.4).collect();
        scalar::gemv(m, k, &ga, &x, None, &mut ov);
        gemv(m, k, &ga, &x, None, &mut rv);
        assert_eq!(ov, rv, "gemv: scalar-delegate scaffold");

        let sm: Vec<f32> = (0..2 * 7).map(|i| (i as f32) * 0.2 - 1.0).collect();
        let mut os = vec![0.0f32; sm.len()];
        let mut rs = vec![0.0f32; sm.len()];
        scalar::softmax(&sm, &mut os, 2, 7);
        softmax(&sm, &mut rs, 2, 7);
        assert_eq!(os, rs, "softmax: scalar-delegate scaffold");

        let ln: Vec<f32> = (0..2 * 5).map(|i| (i as f32) * 0.15 - 0.6).collect();
        let gamma = vec![1.0f32; 5];
        let beta = vec![0.0f32; 5];
        let mut ol = vec![0.0f32; ln.len()];
        let mut rl = vec![0.0f32; ln.len()];
        scalar::layer_norm(&ln, &mut ol, 2, 5, &gamma, &beta, 1e-5);
        layer_norm(&ln, &mut rl, 2, 5, &gamma, &beta, 1e-5);
        assert_eq!(ol, rl, "layer_norm: scalar-delegate scaffold");

        // fused_logmel duplicate vs a hand value (1 band × 3 unit weights
        // over [1, 2, 4] → log10(7); same fixture as the dispatch tests).
        let weights = [1.0f32, 1.0, 1.0];
        let power = [1.0f32, 2.0, 4.0];
        let mut out = [0.0f32; 1];
        fused_logmel(&weights, &power, 1, 3, 1e-10, &mut out);
        assert!((out[0] - 7.0f32.log10()).abs() < 1e-6);
    }

    /// Runs the raw-word 0.7.1 path on a real T-Head hart. Execution is
    /// only possible on real silicon (no upstream qemu support — ADR M4-08
    /// §T01); in CI this test compiles under the riscv64 cross-build and is
    /// exercised on-device by the owner track (M4-08-T15).
    #[test]
    fn rvv071_add_matches_scalar_on_riscv64() {
        // Off a probed 0.7.1 hart the dispatch layer never selects Rvv071,
        // and calling the raw words on a non-0.7.1 hart would be undefined —
        // so gate the body on the runtime probe, reporting a clean skip
        // (never a fabricated pass).
        let feats = crate::features::CpuFeatures::detect();
        if !feats.rvv_071 {
            eprintln!(
                "skip: rvv071_add_matches_scalar_on_riscv64 requires a probed \
                 RVV 0.7.1 hart (rvv_071 = false on this host)"
            );
            return;
        }
        let a: Vec<f32> = (0..37).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32) * -0.125 + 1.0).collect();
        let mut out_scalar = vec![0.0f32; a.len()];
        let mut out_rvv071 = vec![0.0f32; a.len()];
        scalar::add(&a, &b, &mut out_scalar);
        add(&a, &b, &mut out_rvv071);
        // Elementwise f32 add: bit-exact (each output is a single f32 add —
        // no reduction rounding order).
        assert_eq!(
            out_scalar, out_rvv071,
            "rvv071::add must match scalar::add bit-exactly"
        );
    }

    /// Empty-input safety for the raw-word `add`: the `while i < n` guard
    /// must prevent any vector instruction from executing when `n == 0`, so
    /// this call is safe on every hart (no vsetvli word reached).
    #[test]
    fn rvv071_add_zero_len_is_no_op() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        add(&a, &b, &mut out); // must not touch the vector unit
    }
}
