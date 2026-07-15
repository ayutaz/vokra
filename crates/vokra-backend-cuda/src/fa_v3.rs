//! FlashAttention **v3** (Hopper WGMMA, `sm_90a`) — M4-07, the ONLY work
//! package where FA v3 code is legal (design constraint §5-(7); the
//! containment red-line is machine-checked by
//! `scripts/check-fa-v3-confinement.sh`, which supersedes the M3-01 ADR
//! §2-(b) "no FA v3 anywhere" grep gate).
//!
//! Everything Hopper-specific lives in this module + the `launch_flash_attn_v3`
//! arm in `context.rs`, and **nothing here is compiled into the shared
//! [`crate::context`] NVRTC program**: WGMMA (`wgmma.mma_async` family) is an
//! arch-conditional PTX instruction set that requires the NVRTC target
//! `compute_90a`, so appending it to `KERNELS_CUDA` (pinned to `compute_89`
//! for every device, M3-01-T07) would break kernel compilation for every
//! non-Hopper GPU. Instead [`KERNELS_CUDA_FA_V3`] is a **separate program**,
//! lazily compiled only when the device probe reports SM ≥ 9.0
//! (`CudaContext::fa_v3_slot`, M4-07-T03).
//!
//! Design record: `docs/adr/M4-07-fa-v3-hopper.md` (gitignored local ADR).
//! Primary sources for the transcribed instruction forms: NVIDIA PTX ISA
//! ("Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions" —
//! wgmma shapes / matrix descriptor format / fence-commit-wait protocol,
//! "Data Movement" — cp.async, fence.proxy.async) and the FlashAttention-3
//! paper (Shah et al. 2024) for the algorithm structure. Values that could
//! not be verified on real hardware from this CUDA-less authoring machine are
//! marked `OWNER-VERIFY` and enumerated in the ADR's hotspot list.

use core::ffi::c_int;

use crate::sys::{CUfunction, CUmodule};

/// NVRTC `--gpu-architecture` flag for the FA v3 program. **Fixed** — the
/// `VOKRA_NVRTC_GPU_ARCH` escape hatch that A/B-tests [`super::context`]'s
/// shared program does **not** apply here: WGMMA/TMA instructions are only
/// enabled by the arch-specific `compute_90a` target (plain `compute_90`
/// does not enable them), so honoring the env var could only produce a
/// broken compile. Locked by `fa_v3_gencode_is_fixed_90a` below.
pub(crate) const FA_V3_GENCODE_FLAG: &str = "--gpu-architecture=compute_90a";

/// Minimum compute-capability major for the FA v3 path (Hopper). The lazy
/// compile is **not attempted at all** below this (no startup cost on
/// non-Hopper devices, and no structurally-impossible `compute_90a` PTX ever
/// reaches `cuModuleLoadData` on them).
pub(crate) const FA_V3_MIN_CC_MAJOR: c_int = 9;

/// Dynamic shared-memory budget of `vokra_flash_attn_v3_causal_f32` in bytes:
/// 1 024 B alignment slack (the kernel rounds the dynamic-smem base up to
/// 1 024) + five 16 KiB tiles — Q (core-matrix-tiled tf32), K (same), V_raw
/// (row-major f32 staging), V_t (core-matrix-tiled tf32, transposed), P
/// (core-matrix-tiled tf32) = `1024 + 5 * 64 * 64 * 4 = 82 944`.
///
/// This exceeds the 48 KiB default per-block cap, so the lazy module init
/// must opt the kernel in via
/// `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)`, and
/// the device must report at least this much in
/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` (ordinal 97, the
/// same probe FA v2 uses; H100 reports ~227 KiB). The budget is probed at
/// runtime, never assumed (ADR M4-07 kernel design record).
pub(crate) const FLASH_ATTN_V3_MIN_SHARED_BYTES: c_int = 1024 + 5 * 64 * 64 * 4;

/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` (cuda.h
/// `CUfunction_attribute` ordinal 8). Named locally, like the ordinal-97
/// device attribute in `context.rs` — a single-caller constant is not worth
/// widening `sys`.
pub(crate) const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

/// The FA v3 lazy-compile slot state (M4-07-T03), held in a
/// `std::cell::OnceCell` on [`crate::CudaContext`]. `Disabled` carries the
/// human-readable reason that was also printed to stderr when the decision
/// was made — FA v3 being unavailable is **never silent** (FR-EX-08), but it
/// is also never an error by itself: the FA v2 / decomposed GPU paths are
/// unaffected (GPU-internal path selection, not a CPU fallback).
pub(crate) enum FaV3Slot {
    /// Compiled + loaded + shared-memory opt-in applied; `kernel` is
    /// `vokra_flash_attn_v3_causal_f32` resolved from `module`.
    Ready(FaV3Module),
    /// FA v3 is not available on this device/driver; the string says why.
    Disabled(String),
}

/// Owned FA v3 module + kernel handle (unloaded by `CudaContext::drop`).
pub(crate) struct FaV3Module {
    pub(crate) module: CUmodule,
    #[allow(dead_code)] // read by the T05-T08 launcher commit of this WP
    pub(crate) kernel: CUfunction,
}

// Compile-time bounds: the budget must exceed the 48 KiB default per-block
// cap (that is WHY the cuFuncSetAttribute opt-in exists) and stay under
// H100's ~227 KiB opt-in ceiling.
const _: () = assert!(FLASH_ATTN_V3_MIN_SHARED_BYTES > 48 * 1024);
const _: () = assert!(FLASH_ATTN_V3_MIN_SHARED_BYTES < 227 * 1024);

/// Whether the lazy FA v3 compile should be attempted at all for a device
/// reporting `compute_capability_major` (M4-07-T03: on `false` the NVRTC
/// compile is **not tried** — structurally excluding a `compute_90a` PTX from
/// ever reaching a non-Hopper `cuModuleLoadData`, and adding zero startup
/// cost on Ada/Ampere hosts).
pub(crate) fn fa_v3_should_attempt_compile(compute_capability_major: c_int) -> bool {
    compute_capability_major >= FA_V3_MIN_CC_MAJOR
}

// ---------------------------------------------------------------------------
// Env toggles (M4-07-T09/T10/T11). All three follow the presence-based
// convention `VOKRA_CUDA_DISABLE_FA_V2` established (context.rs: `var_os(..)
// .is_some()`): the variable being SET — with any value — flips the toggle.
// Each is a pure function of the read env value so the branch logic is unit
// tested CUDA-less and without process-global env mutation (parallel-test
// safe); call sites pass `std::env::var_os(..).as_deref()`.
// ---------------------------------------------------------------------------

/// `VOKRA_CUDA_DISABLE_FA_V3` (T11): set → the FA v3 path is not selected
/// (session probe + encoder opt-in both honor it). The harness `--fa-mode v2`
/// leg injects this to measure FA v2 with v3 compiled-but-unselected.
#[allow(dead_code)] // consumed by the T09/T11 dispatch commit of this WP
pub(crate) fn fa_v3_env_disabled(v: Option<&std::ffi::OsStr>) -> bool {
    v.is_some()
}

/// `VOKRA_CUDA_FORCE_FA_V3` (T09): set → FA v3 must be *available* or the
/// session / opt-in construction fails with an **explicit error** (FR-EX-08;
/// symmetric with the M2-03-followup §D9 force-flash-attn precedent). It
/// does NOT override the `FA_V3_MIN_TQ` runtime gate — "force" means "fail
/// loudly instead of degrading", not "dispatch on wasted tiles".
#[allow(dead_code)] // consumed by the T09/T11 dispatch commit of this WP
pub(crate) fn fa_v3_env_force(v: Option<&std::ffi::OsStr>) -> bool {
    v.is_some()
}

/// `VOKRA_CUDA_FA_V3_ENCODER` (T10): set → opt the encoder
/// (`encode_prenorm_stack`) attention chains into FA v3 **when available**
/// (non-Hopper hosts keep the decomposed chain + an explicit stderr note —
/// opt-in is a preference, not a force). Default off: the encoder stays
/// byte-for-byte decomposed and every existing parity suite is unaffected.
pub(crate) fn fa_v3_env_encoder_opt_in(v: Option<&std::ffi::OsStr>) -> bool {
    v.is_some()
}

/// Host-side mirror of the kernel's `vokra_fa3_tile_elem` shared-memory
/// indexing (no-swizzle core-matrix tiling; PTX ISA "Shared Memory Matrix
/// Layout"): a 64×64 tf32 tile is stored as 8×4-element **core matrices**
/// ("atoms"), each a contiguous 128-byte block; atoms adjacent in the K
/// dimension sit 128 B apart and atoms adjacent in the M/N dimension 2 048 B
/// apart. Returns the **element** index (multiply by 4 for bytes).
///
/// Kept in Rust purely so the layout algebra is locked by CUDA-less unit
/// tests (bijectivity + atom contiguity below); the kernel carries the same
/// function in CUDA C.
#[allow(dead_code)] // host-side mirror: consumed only by the layout unit tests below
pub(crate) fn fa3_tile_elem(r: usize, c: usize) -> usize {
    debug_assert!(r < 64 && c < 64);
    (r >> 3) * 512 + (c >> 2) * 32 + (r & 7) * 4 + (c & 3)
}

/// Host-side mirror of the kernel's `vokra_fa3_desc` wgmma **matrix
/// descriptor** packing (PTX ISA "Matrix Descriptor Format"): bits [13:0] =
/// smem start address >> 4, [29:16] = leading-dimension byte offset >> 4,
/// [45:32] = stride-dimension byte offset >> 4, [51:49] = base offset (0
/// here), [63:62] = swizzle mode (0 = none). The **field packing** is locked
/// by the unit tests below; the LBO/SBO *values* the kernel feeds it
/// (`FA3_DESC_LBO_BYTES = 128`, `FA3_DESC_SBO_BYTES = 2048`) are an
/// OWNER-VERIFY hotspot (ADR M4-07 hotspot #1) — if the leading/stride field
/// assignment turns out inverted on real Hopper hardware, the fix is swapping
/// those two constants in [`KERNELS_CUDA_FA_V3`].
#[allow(dead_code)] // host-side mirror: consumed only by the descriptor unit tests below
pub(crate) fn fa3_matrix_desc(smem_addr: u32, lbo_bytes: u32, sbo_bytes: u32) -> u64 {
    let mut d: u64 = 0;
    d |= u64::from((smem_addr & 0x3FFFF) >> 4);
    d |= u64::from((lbo_bytes & 0x3FFFF) >> 4) << 16;
    d |= u64::from((sbo_bytes & 0x3FFFF) >> 4) << 32;
    d
}

/// Minimal WGMMA/`compute_90a` feasibility probe source (M4-07-T02). This is
/// **not** a runtime kernel — it exists so the NVRTC boundary questions can
/// be answered in isolation on a CUDA-toolkit host *before* debugging the
/// full kernel:
///
/// (i)  does NVRTC accept `--gpu-architecture=compute_90a`?
/// (ii) does inline-PTX `wgmma.mma_async` pass NVRTC?
/// (iii) at which stage does the same source fail under `compute_89` —
///       NVRTC compile, or deferred to `cuModuleLoadData`?
///
/// Exercised by `tests/fa_v3_nvrtc_feasibility.rs` (clean skip where NVRTC is
/// absent — e.g. the Apple-silicon authoring machine). Kept permanently as
/// the smallest reproducer for NVRTC/WGMMA issues.
pub const FA_V3_FEASIBILITY_SNIPPET: &str = r#"
// M4-07-T02 feasibility probe: one wgmma.mma_async m64n64k8 tf32 tile GEMM
// with the full fence/commit/wait protocol (PTX ISA, "Asynchronous Warpgroup
// Level Matrix Multiply-Accumulate Instructions"). Compile-only artifact.
extern "C" __global__ void vokra_fa_v3_feasibility_probe(
    const float* in,
    float* out)
{
    // A (64x8 tf32) + B (8x64 tf32) staged as no-swizzle core matrices:
    // 16 atoms of 128 B each per operand = 2 * 2048 B.
    __shared__ __align__(1024) unsigned int smem[1024];
    int tid = (int)threadIdx.x;

    for (int i = tid; i < 1024; i += (int)blockDim.x) {
        float v = in[i];
        unsigned int t;
        asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(t) : "f"(v));
        smem[i] = t;
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    unsigned int a_addr;
    unsigned int b_addr;
    asm volatile(
        "{\n.reg .u64 t;\ncvta.to.shared.u64 t, %1;\ncvt.u32.u64 %0, t;\n}\n"
        : "=r"(a_addr) : "l"(smem));
    asm volatile(
        "{\n.reg .u64 t;\ncvta.to.shared.u64 t, %1;\ncvt.u32.u64 %0, t;\n}\n"
        : "=r"(b_addr) : "l"(smem + 512));

    // Matrix descriptor (PTX ISA "Matrix Descriptor Format"): [13:0]
    // addr>>4, [29:16] LBO>>4, [45:32] SBO>>4, [63:62] swizzle = 0 (none).
    // 64x8 / 8x64 operands: K-adjacent atoms 128 B apart, M/N-adjacent 256 B.
    unsigned long long desc_a = 0ull;
    desc_a |= (unsigned long long)((a_addr & 0x3FFFFu) >> 4);
    desc_a |= (unsigned long long)((128u & 0x3FFFFu) >> 4) << 16;
    desc_a |= (unsigned long long)((256u & 0x3FFFFu) >> 4) << 32;
    unsigned long long desc_b = 0ull;
    desc_b |= (unsigned long long)((b_addr & 0x3FFFFu) >> 4);
    desc_b |= (unsigned long long)((128u & 0x3FFFFu) >> 4) << 16;
    desc_b |= (unsigned long long)((256u & 0x3FFFFu) >> 4) << 32;

    float d[32];
    for (int i = 0; i < 32; ++i) {
        d[i] = 0.0f;
    }
    int scale_d = 0; // first (only) k-step: D = A*B
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
        " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31},\n"
        " %32, %33, p, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
          "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
          "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
          "+f"(d[30]), "+f"(d[31])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

    float acc = 0.0f;
    for (int i = 0; i < 32; ++i) {
        acc += d[i];
    }
    out[tid] = acc;
}
"#;

/// The FA v3 NVRTC program (M4-07-T03 separation; kernel body lands with
/// T05–T08). Compiled **only** for `compute_90a` and **only** when the device
/// probe reports SM ≥ 9.0 — never part of [`super::context`]'s shared
/// `KERNELS_CUDA` program (see the module docs for why).
///
/// Placeholder state (T03): the program currently carries the feasibility
/// probe entry point so the lazy-compile plumbing is exercisable end-to-end;
/// `vokra_flash_attn_v3_causal_f32` is filled in by T05–T08.
pub const KERNELS_CUDA_FA_V3: &str = FA_V3_FEASIBILITY_SNIPPET;

/// The `extern "C"` kernel symbol the lazy loader resolves from
/// [`KERNELS_CUDA_FA_V3`].
pub(crate) const FA_V3_KERNEL_SYMBOL: &core::ffi::CStr = c"vokra_fa_v3_feasibility_probe";

#[cfg(test)]
mod tests {
    use super::*;

    /// The FA v3 gencode pin is `compute_90a` and is not assembled from any
    /// env var: `compute_90` (no `a` suffix) would not enable WGMMA, so the
    /// `VOKRA_NVRTC_GPU_ARCH` escape hatch must not leak into this program
    /// (M4-07-T03). The constant being a `const` (not a function reading the
    /// environment) is the structural guarantee; this test pins the value.
    #[test]
    fn fa_v3_gencode_is_fixed_90a() {
        assert_eq!(FA_V3_GENCODE_FLAG, "--gpu-architecture=compute_90a");
        assert!(
            FA_V3_GENCODE_FLAG.ends_with("90a"),
            "must be the arch-conditional target, not plain compute_90"
        );
    }

    /// SM 9.0 (Hopper) and above attempt the lazy compile; everything below
    /// (Ada 8.9, Ampere 8.6, …) must not even try (M4-07-T03 completion
    /// condition: no compile attempt on non-Hopper).
    #[test]
    fn fa_v3_compile_attempt_gate_is_sm_9_0() {
        assert!(!fa_v3_should_attempt_compile(8)); // Ada / Ampere major
        assert!(fa_v3_should_attempt_compile(9)); // Hopper
        assert!(fa_v3_should_attempt_compile(10)); // future majors included
        assert!(!fa_v3_should_attempt_compile(0));
        assert!(!fa_v3_should_attempt_compile(-1)); // corrupt probe value
    }

    /// All three env toggles are presence-based, mirroring the
    /// `VOKRA_CUDA_DISABLE_FA_V2` convention (`var_os(..).is_some()`): any
    /// value — even `"0"` or empty — counts as set. Pure functions of the
    /// read value, so no process-global env mutation in tests.
    #[test]
    fn fa_v3_env_toggles_are_presence_based() {
        use std::ffi::OsStr;
        for f in [
            fa_v3_env_disabled,
            fa_v3_env_force,
            fa_v3_env_encoder_opt_in,
        ] {
            assert!(!f(None), "unset must be off");
            assert!(f(Some(OsStr::new("1"))));
            assert!(f(Some(OsStr::new(""))), "presence-based: empty counts");
            assert!(f(Some(OsStr::new("0"))), "presence-based: '0' counts");
        }
    }

    /// The shared-memory budget constant equals the layout arithmetic it
    /// documents (1 KiB alignment slack + five 64×64×4-byte tiles).
    #[test]
    fn fa_v3_shared_bytes_matches_layout() {
        assert_eq!(FLASH_ATTN_V3_MIN_SHARED_BYTES, 82_944);
        assert_eq!(FLASH_ATTN_V3_MIN_SHARED_BYTES, 1024 + 5 * (64 * 64 * 4));
        // (48 KiB < budget < 227 KiB bounds are compile-time `const _`
        // asserts at module scope.)
    }

    /// Core-matrix tiling: the map (r, c) → element offset is a bijection of
    /// the 64×64 tile onto [0, 4096).
    #[test]
    fn fa3_tile_elem_is_a_bijection() {
        let mut seen = vec![false; 64 * 64];
        for r in 0..64 {
            for c in 0..64 {
                let e = fa3_tile_elem(r, c);
                assert!(e < 4096, "offset out of range at ({r}, {c})");
                assert!(!seen[e], "collision at ({r}, {c}) -> {e}");
                seen[e] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    /// Each 8-row × 4-column atom occupies one **contiguous** 128-byte block
    /// (32 consecutive elements) — the wgmma core-matrix requirement that a
    /// plain row-major [64][64] buffer would violate (ADR M4-07 §(e)-(ii)).
    #[test]
    fn fa3_tile_atoms_are_contiguous_128_byte_blocks() {
        for mb in 0..8 {
            for kb in 0..16 {
                let base = fa3_tile_elem(mb * 8, kb * 4);
                assert_eq!(base % 32, 0, "atom base must be 128-B aligned");
                let mut offs: Vec<usize> = Vec::with_capacity(32);
                for ri in 0..8 {
                    for ci in 0..4 {
                        offs.push(fa3_tile_elem(mb * 8 + ri, kb * 4 + ci));
                    }
                }
                offs.sort_unstable();
                let expect: Vec<usize> = (base..base + 32).collect();
                assert_eq!(offs, expect, "atom ({mb}, {kb}) is not contiguous");
            }
        }
    }

    /// Atom adjacency distances match the descriptor constants the kernel
    /// will encode: K-adjacent atoms 128 B apart, M/N-adjacent 2 048 B apart.
    #[test]
    fn fa3_tile_atom_strides_match_descriptor_constants() {
        let e00 = fa3_tile_elem(0, 0);
        let e01 = fa3_tile_elem(0, 4); // next atom in K
        let e10 = fa3_tile_elem(8, 0); // next atom in M/N
        assert_eq!((e01 - e00) * 4, 128, "K-adjacent atom stride");
        assert_eq!((e10 - e00) * 4, 2048, "M/N-adjacent atom stride");
        // One wgmma k8 step (8 tf32 = 2 atoms) advances the descriptor start
        // by 256 bytes.
        let e_k8 = fa3_tile_elem(0, 8);
        assert_eq!((e_k8 - e00) * 4, 256, "k8-step descriptor advance");
    }

    /// Descriptor bit packing (PTX ISA "Matrix Descriptor Format"): each
    /// field lands in its documented bit range, 16-byte-encoded.
    #[test]
    fn fa3_matrix_desc_bit_packing() {
        // addr 0x1040 (>>4 = 0x104), LBO 128 (>>4 = 8), SBO 2048 (>>4 = 128).
        let d = fa3_matrix_desc(0x1040, 128, 2048);
        assert_eq!(d & 0x3FFF, 0x104, "start address field [13:0]");
        assert_eq!((d >> 16) & 0x3FFF, 8, "leading byte offset field [29:16]");
        assert_eq!((d >> 32) & 0x3FFF, 128, "stride byte offset field [45:32]");
        assert_eq!((d >> 49) & 0x7, 0, "base offset field [51:49] is zero");
        assert_eq!((d >> 62) & 0x3, 0, "swizzle mode [63:62] is none");
        // The 18-bit smem window is masked before encoding.
        let d2 = fa3_matrix_desc(0xFFFF_FFFF, 0, 0);
        assert_eq!(d2 & 0x3FFF, 0x3FFF, "address is masked to 18 bits >> 4");
        assert_eq!(d2 >> 14 & 0x3, 0, "no spill past the address field");
    }

    /// String-level source checks (the honest verification level available on
    /// a CUDA-less host): the feasibility snippet carries the exact
    /// PTX-ISA-transcribed mnemonics and the full sync protocol, and no
    /// fast-math `__expf` sneaks anywhere near the program.
    #[test]
    fn fa_v3_snippet_carries_wgmma_protocol() {
        for needle in [
            "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32",
            "wgmma.fence.sync.aligned",
            "wgmma.commit_group.sync.aligned",
            "wgmma.wait_group.sync.aligned 0",
            "cvt.rna.tf32.f32",
            "fence.proxy.async.shared::cta",
            "cvta.to.shared.u64",
            "extern \"C\" __global__ void vokra_fa_v3_feasibility_probe",
        ] {
            assert!(
                FA_V3_FEASIBILITY_SNIPPET.contains(needle),
                "feasibility snippet must contain `{needle}`"
            );
        }
        assert!(
            !KERNELS_CUDA_FA_V3.contains("__expf"),
            "fast-math __expf is forbidden (M2-03-followup §D3 contract)"
        );
        assert!(
            !KERNELS_CUDA_FA_V3.contains('\0'),
            "interior NUL would break the NVRTC CString boundary"
        );
    }
}
