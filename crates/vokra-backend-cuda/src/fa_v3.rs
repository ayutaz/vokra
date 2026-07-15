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

/// The FA v3 NVRTC program (M4-07-T05..T08). Compiled **only** for
/// `compute_90a` and **only** when the device probe reports SM ≥ 9.0 — never
/// part of [`super::context`]'s shared `KERNELS_CUDA` program (see the module
/// docs for why).
///
/// # Numeric contract (rounding points — the FA_V3 parity-atol material)
///
/// Identical online-softmax semantics to `vokra_flash_attn_v2_causal_f32`
/// (running max `m` / running exp-sum `l` / output accumulator `O` in fp32,
/// log-sum-exp trick, standard `expf` — never fast-math `__expf`; causal mask
/// `k_abs > q_offset + q_row_abs` → −INF; `scale` multiplied into the scores
/// before masking), with these tf32 rounding points (ADR M4-07 §(d)):
/// Q, K, V and P are rounded f32 → tf32 (`cvt.rna.tf32.f32`, one rounding
/// each) before entering the two WGMMA matmuls; both matmuls accumulate in
/// fp32; softmax state, rescale and the final `O/l` stay fp32.
///
/// # Structure (ADR M4-07 kernel design record)
///
/// One warpgroup (128 threads) per block; BR3 = BC3 = d_head = 64. Per-head
/// host loop (`launch_flash_attn_v3`), grid = (⌈t_q/64⌉, 1, 1), kernel is
/// single-head — mirroring the FA v2 launcher exactly. K/V tiles stream in
/// via 16-byte `cp.async` into a **no-swizzle core-matrix-tiled** layout
/// (8-row × 16-byte atoms, contiguous 128-B blocks; K-adjacent 128 B apart,
/// M/N-adjacent 2 048 B apart — `FA3_DESC_LBO_BYTES` / `FA3_DESC_SBO_BYTES`,
/// the ADR OWNER-VERIFY hotspot #1: if the descriptor field assignment is
/// inverted on real Hopper silicon, swap those two defines). Synchronous
/// single-buffer pipeline — correctness first; TMA bulk copies +
/// warp-specialisation are the post-T18 optimisation (ADR §(e)).
///
/// Ragged tails: Q/K/V tail rows are zero-filled (never garbage — a NaN in a
/// dead V row would poison `P·V` through `0 × NaN`), score columns past
/// `bc_eff` are masked to −INF (P = 0 exactly), and output rows past
/// `br_eff` are never stored. Fully-masked causal tail tiles are skipped
/// (they contribute exactly nothing: all P = 0, α = 1).
pub const KERNELS_CUDA_FA_V3: &str = r#"
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// Tile geometry (host launcher + FLASH_ATTN_V3_MIN_SHARED_BYTES must agree).
#define FA3_BR 64
#define FA3_BC 64
#define FA3_D  64
// wgmma matrix-descriptor strides for the core-matrix tiling below
// (OWNER-VERIFY hotspot #1: swap these two if the PTX ISA leading/stride
// field assignment proves inverted on real Hopper silicon).
#define FA3_DESC_LBO_BYTES 128u
#define FA3_DESC_SBO_BYTES 2048u

// Generic pointer -> shared-window u32 address (pure PTX so no dependence on
// NVRTC exposing __cvta_generic_to_shared).
__device__ __forceinline__ unsigned int vokra_fa3_smem_u32(const void* p)
{
    unsigned int s;
    asm volatile(
        "{\n.reg .u64 t;\ncvta.to.shared.u64 t, %1;\ncvt.u32.u64 %0, t;\n}\n"
        : "=r"(s) : "l"(p));
    return s;
}

// No-swizzle core-matrix tiling (PTX ISA "Shared Memory Matrix Layout"):
// atom = 8 rows x 4 tf32 columns, one contiguous 128-byte block; atoms
// K-adjacent 128 B apart, M/N-adjacent 2048 B apart. Returns the ELEMENT
// index inside a 64x64 tile (byte offset = 4x). Mirrored by the Rust
// fa3_tile_elem + its bijectivity/contiguity unit tests.
__device__ __forceinline__ unsigned int vokra_fa3_tile_elem(unsigned int r, unsigned int c)
{
    return (r >> 3) * 512u + (c >> 2) * 32u + (r & 7u) * 4u + (c & 3u);
}

// wgmma shared-memory matrix descriptor (PTX ISA "Matrix Descriptor
// Format"): [13:0] addr>>4, [29:16] LBO>>4, [45:32] SBO>>4, [51:49] base
// offset = 0, [63:62] swizzle = 0 (none). Mirrored by Rust fa3_matrix_desc.
__device__ __forceinline__ unsigned long long vokra_fa3_desc(unsigned int smem_addr)
{
    unsigned long long d = 0ull;
    d |= (unsigned long long)((smem_addr & 0x3FFFFu) >> 4);
    d |= (unsigned long long)((FA3_DESC_LBO_BYTES & 0x3FFFFu) >> 4) << 16;
    d |= (unsigned long long)((FA3_DESC_SBO_BYTES & 0x3FFFFu) >> 4) << 32;
    return d;
}

// Explicit f32 -> tf32 rounding (round-to-nearest, cvt.rna — the documented
// conversion; keeps the rounding point explicit for the parity-atol
// derivation instead of relying on implicit tensor-core truncation).
__device__ __forceinline__ unsigned int vokra_fa3_f32_to_tf32(float v)
{
    unsigned int t;
    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(t) : "f"(v));
    return t;
}

// Butterfly shuffle over the 4-lane quad that shares one fragment row
// (bit-cast through b32; all 32 lanes participate, mask 0xffffffff).
__device__ __forceinline__ float vokra_fa3_shfl_bfly(float v, int lane_mask)
{
    unsigned int u = (unsigned int)__float_as_int(v);
    unsigned int r;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;\n"
                 : "=r"(r) : "r"(u), "r"(lane_mask));
    return __int_as_float((int)r);
}

// 16-byte global -> shared async copy (sm_80+ cp.async; the K/V streaming
// mechanism of this kernel — TMA bulk-tensor is the post-T18 follow-up).
__device__ __forceinline__ void vokra_fa3_cp_async_16(unsigned int dst_smem, const float* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst_smem), "l"(src) : "memory");
}

__device__ __forceinline__ void vokra_fa3_cp_async_wait_all()
{
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
}

// Makes generic-proxy shared-memory writes visible to the async proxy the
// wgmma reads through (PTX ISA fence.proxy.async).
__device__ __forceinline__ void vokra_fa3_fence_proxy_async()
{
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void vokra_fa3_wgmma_fence()
{
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void vokra_fa3_wgmma_commit()
{
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void vokra_fa3_wgmma_wait0()
{
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// One m64n64k8 tf32 warpgroup MMA with fp32 accumulator d[32] (both operands
// from shared memory via matrix descriptors). scale_d = 0 zero-initialises
// the accumulator (D = A*B), 1 accumulates (D = A*B + D). Reference-to-array
// keeps d register-resident (all call sites index it with unrolled constant
// indices). Instruction form transcribed from the PTX ISA wgmma chapter; the
// tf32 shape takes no transpose immediates.
__device__ __forceinline__ void vokra_fa3_wgmma_m64n64k8_tf32(
    float (&d)[32], unsigned long long desc_a, unsigned long long desc_b, int scale_d)
{
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
}

// ---- FlashAttention-v3 causal, tf32 x tf32 -> fp32 (M4-07, Hopper) ---------
//
// Same argument contract as vokra_flash_attn_v2_causal_f32: row-major
// head-relative Q[t_q, 64], K[t_kv, 64], V[t_kv, 64], O[t_q, 64]; the host
// launcher folds the attention scale into the qh gather and passes
// scale = 1.0 (the kernel still multiplies, preserving contract identity).
//
// Fragment map (PTX ISA wgmma f32 d-fragment; OWNER-VERIFY hotspot #2):
// warp w owns rows 16w..16w+15; lane owns rows {16w + lane/4, +8} and, per
// 8-column group g, columns {8g + 2*(lane%4), +1} — registers 4g..4g+3.
extern "C" __global__ void __launch_bounds__(128)
vokra_flash_attn_v3_causal_f32(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int t_q,
    int t_kv,
    int d_head,
    int q_offset,
    bool causal,
    float scale)
{
    // Host launcher validates d_head == 64; this is a uniform defensive exit
    // (before any wgmma, so warpgroup convergence is unaffected).
    if (d_head != FA3_D) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int q_row_base = (int)blockIdx.x * FA3_BR;
    int br_eff = t_q - q_row_base;
    if (br_eff > FA3_BR) {
        br_eff = FA3_BR;
    }
    if (br_eff <= 0) {
        return; // uniform for the whole block
    }

    // Dynamic shared memory, base rounded up to 1024 B so every region and
    // every atom is descriptor- and cp.async-aligned. Region layout (bytes
    // from base): Q_t 0 | K_t 16384 | V_raw 32768 | V_t 49152 | P_t 65536.
    extern __shared__ float vokra_fa3_smem[];
    char* raw = (char*)vokra_fa3_smem;
    unsigned int raw_s = vokra_fa3_smem_u32(raw);
    unsigned int base_s = (raw_s + 1023u) & ~1023u;
    char* base = raw + (base_s - raw_s);

    unsigned int q_s = base_s;
    unsigned int k_s = base_s + 16384u;
    unsigned int vraw_s = base_s + 32768u;
    unsigned int vt_s = base_s + 49152u;
    unsigned int p_s = base_s + 65536u;
    unsigned int* q_u = (unsigned int*)base;
    unsigned int* k_u = (unsigned int*)(base + 16384);
    float* v_raw = (float*)(base + 32768);
    unsigned int* vt_u = (unsigned int*)(base + 49152);
    unsigned int* p_u = (unsigned int*)(base + 65536);

    // ---- Prologue: Q tile, loaded once per block ------------------------
    // 64 rows x 16 sixteen-byte chunks into the core-matrix tiling; ragged
    // rows past br_eff are zero-filled.
    for (int idx = tid; idx < FA3_BR * (FA3_D / 4); idx += (int)blockDim.x) {
        int r = idx >> 4;
        int kb = idx & 15;
        unsigned int dst = q_s + vokra_fa3_tile_elem((unsigned int)r, (unsigned int)(kb * 4)) * 4u;
        if (r < br_eff) {
            vokra_fa3_cp_async_16(dst, Q + (unsigned long long)(q_row_base + r) * FA3_D + kb * 4);
        } else {
            float* z = (float*)(base + (dst - base_s));
            z[0] = 0.0f;
            z[1] = 0.0f;
            z[2] = 0.0f;
            z[3] = 0.0f;
        }
    }
    vokra_fa3_cp_async_wait_all();
    __syncthreads();
    // In-place f32 -> tf32 over the whole Q region (the tiling is a
    // bijection onto the region, so a linear sweep touches every element
    // exactly once).
    for (int i = tid; i < FA3_BR * FA3_D; i += (int)blockDim.x) {
        q_u[i] = vokra_fa3_f32_to_tf32(((float*)q_u)[i]);
    }

    // Per-thread fragment rows.
    const int r1 = warp * 16 + (lane >> 2);
    const int r2 = r1 + 8;

    // Online softmax state (fp32) + fp32 output accumulator.
    float m1 = -INFINITY;
    float l1 = 0.0f;
    float m2 = -INFINITY;
    float l2 = 0.0f;
    float o[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        o[i] = 0.0f;
    }

    int n_kv_tiles = (t_kv + FA3_BC - 1) / FA3_BC;
    if (causal) {
        // Tiles entirely in this block's causal future contribute exactly
        // nothing (every score masked -> P = 0, alpha = 1); skip them.
        int last_k = q_offset + q_row_base + br_eff - 1;
        int lim = (last_k / FA3_BC) + 1;
        if (lim < n_kv_tiles) {
            n_kv_tiles = lim;
        }
    }

    for (int kt = 0; kt < n_kv_tiles; ++kt) {
        int k_col_base = kt * FA3_BC;
        int bc_eff = t_kv - k_col_base;
        if (bc_eff > FA3_BC) {
            bc_eff = FA3_BC;
        }

        // ---- K tile (core-tiled) + V tile (row-major staging) -----------
        for (int idx = tid; idx < FA3_BC * (FA3_D / 4); idx += (int)blockDim.x) {
            int j = idx >> 4;
            int kb = idx & 15;
            unsigned int kdst =
                k_s + vokra_fa3_tile_elem((unsigned int)j, (unsigned int)(kb * 4)) * 4u;
            unsigned int vdst = vraw_s + ((unsigned int)j * FA3_D + (unsigned int)(kb * 4)) * 4u;
            if (j < bc_eff) {
                const float* krow = K + (unsigned long long)(k_col_base + j) * FA3_D + kb * 4;
                const float* vrow = V + (unsigned long long)(k_col_base + j) * FA3_D + kb * 4;
                vokra_fa3_cp_async_16(kdst, krow);
                vokra_fa3_cp_async_16(vdst, vrow);
            } else {
                // Zero-fill the ragged tail (a garbage NaN in a dead V row
                // would poison P.V through 0 x NaN).
                float* zk = (float*)(base + (kdst - base_s));
                zk[0] = 0.0f;
                zk[1] = 0.0f;
                zk[2] = 0.0f;
                zk[3] = 0.0f;
                float* zv = (float*)(base + (vdst - base_s));
                zv[0] = 0.0f;
                zv[1] = 0.0f;
                zv[2] = 0.0f;
                zv[3] = 0.0f;
            }
        }
        vokra_fa3_cp_async_wait_all();
        __syncthreads();
        // K: in-place tf32. V: transpose (j, c) -> V_t(c, j) with tf32 —
        // the P.V matmul needs V K-major (kv contiguous per d_head column).
        for (int i = tid; i < FA3_BC * FA3_D; i += (int)blockDim.x) {
            k_u[i] = vokra_fa3_f32_to_tf32(((float*)k_u)[i]);
        }
        for (int i = tid; i < FA3_BC * FA3_D; i += (int)blockDim.x) {
            int c = i >> 6;
            int j = i & 63;
            vt_u[vokra_fa3_tile_elem((unsigned int)c, (unsigned int)j)] =
                vokra_fa3_f32_to_tf32(v_raw[j * FA3_D + c]);
        }
        __syncthreads();
        vokra_fa3_fence_proxy_async();

        // ---- S = Q . K^T (8 k8-steps; first zero-initialises) ------------
        float s[32];
        vokra_fa3_wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < FA3_D / 8; ++ks) {
            vokra_fa3_wgmma_m64n64k8_tf32(
                s, vokra_fa3_desc(q_s + (unsigned int)ks * 256u),
                vokra_fa3_desc(k_s + (unsigned int)ks * 256u), ks == 0 ? 0 : 1);
        }
        vokra_fa3_wgmma_commit();
        vokra_fa3_wgmma_wait0();

        // ---- scale -> mask -> online softmax (fp32, standard expf) -------
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            s[i] *= scale;
        }
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            int c0 = g * 8 + (lane & 3) * 2;
            int ka = k_col_base + c0;
            if (c0 >= bc_eff || (causal && ka > q_offset + q_row_base + r1)) {
                s[4 * g] = -INFINITY;
            }
            if (c0 + 1 >= bc_eff || (causal && ka + 1 > q_offset + q_row_base + r1)) {
                s[4 * g + 1] = -INFINITY;
            }
            if (c0 >= bc_eff || (causal && ka > q_offset + q_row_base + r2)) {
                s[4 * g + 2] = -INFINITY;
            }
            if (c0 + 1 >= bc_eff || (causal && ka + 1 > q_offset + q_row_base + r2)) {
                s[4 * g + 3] = -INFINITY;
            }
        }

        float mt1 = -INFINITY;
        float mt2 = -INFINITY;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            mt1 = fmaxf(mt1, fmaxf(s[4 * g], s[4 * g + 1]));
            mt2 = fmaxf(mt2, fmaxf(s[4 * g + 2], s[4 * g + 3]));
        }
        mt1 = fmaxf(mt1, vokra_fa3_shfl_bfly(mt1, 1));
        mt1 = fmaxf(mt1, vokra_fa3_shfl_bfly(mt1, 2));
        mt2 = fmaxf(mt2, vokra_fa3_shfl_bfly(mt2, 1));
        mt2 = fmaxf(mt2, vokra_fa3_shfl_bfly(mt2, 2));

        float mn1 = fmaxf(m1, mt1);
        float mn2 = fmaxf(m2, mt2);
        float a1 = (m1 == -INFINITY) ? 0.0f : expf(m1 - mn1);
        float a2 = (m2 == -INFINITY) ? 0.0f : expf(m2 - mn2);
        float lt1 = 0.0f;
        float lt2 = 0.0f;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            int c0 = g * 8 + (lane & 3) * 2;
            float p00 = (s[4 * g] == -INFINITY) ? 0.0f : expf(s[4 * g] - mn1);
            float p01 = (s[4 * g + 1] == -INFINITY) ? 0.0f : expf(s[4 * g + 1] - mn1);
            float p10 = (s[4 * g + 2] == -INFINITY) ? 0.0f : expf(s[4 * g + 2] - mn2);
            float p11 = (s[4 * g + 3] == -INFINITY) ? 0.0f : expf(s[4 * g + 3] - mn2);
            lt1 += p00 + p01;
            lt2 += p10 + p11;
            p_u[vokra_fa3_tile_elem((unsigned int)r1, (unsigned int)c0)] =
                vokra_fa3_f32_to_tf32(p00);
            p_u[vokra_fa3_tile_elem((unsigned int)r1, (unsigned int)(c0 + 1))] =
                vokra_fa3_f32_to_tf32(p01);
            p_u[vokra_fa3_tile_elem((unsigned int)r2, (unsigned int)c0)] =
                vokra_fa3_f32_to_tf32(p10);
            p_u[vokra_fa3_tile_elem((unsigned int)r2, (unsigned int)(c0 + 1))] =
                vokra_fa3_f32_to_tf32(p11);
        }
        lt1 += vokra_fa3_shfl_bfly(lt1, 1);
        lt1 += vokra_fa3_shfl_bfly(lt1, 2);
        lt2 += vokra_fa3_shfl_bfly(lt2, 1);
        lt2 += vokra_fa3_shfl_bfly(lt2, 2);
        l1 = l1 * a1 + lt1;
        l2 = l2 * a2 + lt2;
        m1 = mn1;
        m2 = mn2;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            o[4 * g] *= a1;
            o[4 * g + 1] *= a1;
            o[4 * g + 2] *= a2;
            o[4 * g + 3] *= a2;
        }

        __syncthreads();
        vokra_fa3_fence_proxy_async();

        // ---- O += P . V (8 k8-steps over the kv tile) --------------------
        vokra_fa3_wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < FA3_BC / 8; ++ks) {
            vokra_fa3_wgmma_m64n64k8_tf32(
                o, vokra_fa3_desc(p_s + (unsigned int)ks * 256u),
                vokra_fa3_desc(vt_s + (unsigned int)ks * 256u), 1);
        }
        vokra_fa3_wgmma_commit();
        vokra_fa3_wgmma_wait0();
        // K/V/P smem is reused by the next tile; the wait above retired the
        // wgmma reads, this barrier aligns the threads before overwriting.
        __syncthreads();
    }

    // ---- Epilogue: O = O / l (guarded against the ragged q tail) --------
    float inv1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
    float inv2 = (l2 > 0.0f) ? (1.0f / l2) : 0.0f;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        int c0 = g * 8 + (lane & 3) * 2;
        if (r1 < br_eff) {
            O[(unsigned long long)(q_row_base + r1) * FA3_D + c0] = o[4 * g] * inv1;
            O[(unsigned long long)(q_row_base + r1) * FA3_D + c0 + 1] = o[4 * g + 1] * inv1;
        }
        if (r2 < br_eff) {
            O[(unsigned long long)(q_row_base + r2) * FA3_D + c0] = o[4 * g + 2] * inv2;
            O[(unsigned long long)(q_row_base + r2) * FA3_D + c0 + 1] = o[4 * g + 3] * inv2;
        }
    }
}
"#;

/// The `extern "C"` kernel symbol the lazy loader resolves from
/// [`KERNELS_CUDA_FA_V3`].
pub(crate) const FA_V3_KERNEL_SYMBOL: &core::ffi::CStr = c"vokra_flash_attn_v3_causal_f32";

/// Scalar-geometry validation of a FlashAttention-v3 dispatch (M4-07-T08) —
/// the shape contract of `vokra_flash_attn_v3_causal_f32`, shared by the
/// `flash_attn_v3_dev` diagnostic wrapper and exercised CUDA-less by the
/// negative tests in `tests/parity_kernels_cuda.rs` (a free function so the
/// negatives stay green on hosts where no `CudaContext` can be built).
///
/// # Errors
///
/// [`vokra_core::VokraError::InvalidArgument`] when any dimension is zero,
/// `d` is not divisible by `n_head`, `d / n_head != 64` (the kernel's tile /
/// fragment design is d_head-fixed, exactly like FA v2 — no silent
/// over-allocation, FR-EX-08), or a causal window would attend past the K/V
/// range (`q_offset + t_q > t_kv`).
pub fn flash_attn_v3_validate_args(
    t_q: usize,
    t_kv: usize,
    d: usize,
    n_head: usize,
    causal: bool,
    q_offset: usize,
) -> vokra_core::Result<()> {
    use vokra_core::VokraError;
    if t_q == 0 || t_kv == 0 || d == 0 || n_head == 0 {
        return Err(VokraError::InvalidArgument(
            "flash_attn_v3 dimensions t_q, t_kv, d, n_head must all be >= 1".to_owned(),
        ));
    }
    if d % n_head != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "flash_attn_v3 d ({d}) must be divisible by n_head ({n_head})"
        )));
    }
    let hd = d / n_head;
    if hd != 64 {
        return Err(VokraError::InvalidArgument(format!(
            "flash_attn_v3 d/n_head ({hd}) must equal 64 (the FA v3 warpgroup tile / \
             fragment design is d_head-fixed; see FLASH_ATTN_V3_MIN_SHARED_BYTES)"
        )));
    }
    if causal && q_offset.saturating_add(t_q) > t_kv {
        return Err(VokraError::InvalidArgument(format!(
            "flash_attn_v3 causal=true requires q_offset + t_q <= t_kv \
             (got q_offset={q_offset}, t_q={t_q}, t_kv={t_kv})"
        )));
    }
    Ok(())
}

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

    /// String-level verification of the full kernel source (T05–T08) — the
    /// honest check level available on a CUDA-less host: the transcribed
    /// instruction forms, the sync protocol in dispatch order, the numeric
    /// contract markers and the ragged-tail guards are all present.
    #[test]
    fn fa_v3_kernel_source_carries_full_protocol() {
        let src = KERNELS_CUDA_FA_V3;
        for needle in [
            // T05: WGMMA building block.
            "extern \"C\" __global__ void __launch_bounds__(128)",
            "vokra_flash_attn_v3_causal_f32",
            "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32",
            "setp.ne.b32 p, %34, 0;",
            // T06: smem layout + async load pipeline (cp.async per ADR (e)).
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "cp.async.wait_group 0",
            "vokra_fa3_tile_elem",
            "#define FA3_DESC_LBO_BYTES 128u",
            "#define FA3_DESC_SBO_BYTES 2048u",
            "fence.proxy.async.shared::cta",
            // T07: online softmax + mask + contract identity.
            "expf(", // standard expf, never __expf (checked above)
            "-INFINITY",
            "shfl.sync.bfly.b32",
            "cvt.rna.tf32.f32",
            // T08: ragged tail + epilogue guards.
            "br_eff",
            "bc_eff",
            "__launch_bounds__(128)",
        ] {
            assert!(
                src.contains(needle),
                "kernel source must contain `{needle}`"
            );
        }
        // Sync protocol is CALLED in dispatch order inside the kernel body
        // (two sequences per tile — S and P·V: fence before the mma chain,
        // commit then wait-0 after).
        let body_at = src
            .find("vokra_flash_attn_v3_causal_f32(")
            .expect("kernel entry present");
        let body = &src[body_at..];
        let fence = body.find("vokra_fa3_wgmma_fence();").expect("fence call");
        let mma = body
            .find("vokra_fa3_wgmma_m64n64k8_tf32(")
            .expect("mma helper call");
        let commit = body.find("vokra_fa3_wgmma_commit();").expect("commit call");
        let wait = body.find("vokra_fa3_wgmma_wait0();").expect("wait call");
        assert!(
            fence < mma && mma < commit && commit < wait,
            "kernel body must call fence -> mma -> commit -> wait in order"
        );
        assert_eq!(
            body.matches("vokra_fa3_wgmma_fence();").count(),
            2,
            "one fence per wgmma sequence (S and P·V)"
        );
        assert_eq!(
            body.matches("vokra_fa3_wgmma_wait0();").count(),
            2,
            "one wait-0 per wgmma sequence (S and P·V)"
        );
        // The kernel and the placeholder-era probe are distinct programs now.
        assert!(
            !src.contains("vokra_fa_v3_feasibility_probe"),
            "the runtime program must not carry the T02 probe entry point"
        );
    }

    /// The kernel symbol constant matches the `extern "C"` entry point in the
    /// source (a rename in one place without the other must trip red).
    #[test]
    fn fa_v3_kernel_symbol_matches_source() {
        let sym = FA_V3_KERNEL_SYMBOL.to_str().expect("ascii");
        assert!(KERNELS_CUDA_FA_V3.contains(&format!("\n{sym}(")));
    }

    /// T08 negatives — the scalar-geometry validator is CUDA-less green and
    /// mirrors `flash_attn_dev`'s FA v2 checks 1:1 (zero dims / d % n_head /
    /// d_head != 64 / causal window overflow), plus the accepted geometry.
    #[test]
    fn flash_attn_v3_validate_args_negatives_and_positive() {
        use vokra_core::VokraError;
        let bad = [
            (0usize, 64usize, 256usize, 4usize, false, 0usize), // t_q = 0
            (64, 0, 256, 4, false, 0),                          // t_kv = 0
            (64, 64, 0, 4, false, 0),                           // d = 0
            (64, 64, 256, 0, false, 0),                         // n_head = 0
            (64, 64, 250, 4, false, 0),                         // d % n_head != 0
            (64, 64, 256, 2, false, 0),                         // d_head = 128 != 64
            (64, 64, 128, 4, false, 0),                         // d_head = 32 != 64
            (64, 63, 256, 4, true, 0),                          // causal window overflow
            (64, 128, 256, 4, true, 65),                        // q_offset pushes past t_kv
        ];
        for (t_q, t_kv, d, n_head, causal, q_offset) in bad {
            match flash_attn_v3_validate_args(t_q, t_kv, d, n_head, causal, q_offset) {
                Err(VokraError::InvalidArgument(_)) => {}
                other => panic!(
                    "({t_q},{t_kv},{d},{n_head},causal={causal},q_offset={q_offset}) \
                     must be InvalidArgument, got {other:?}"
                ),
            }
        }
        // Positive: Whisper-shaped prefix step and the non-causal encoder
        // shape both pass.
        flash_attn_v3_validate_args(64, 64, 256, 4, true, 0).expect("prefix step");
        flash_attn_v3_validate_args(1500, 1500, 256, 4, false, 0).expect("encoder shape");
        // usize overflow in the causal window must not panic (saturating).
        match flash_attn_v3_validate_args(usize::MAX, 64, 256, 4, true, usize::MAX) {
            Err(VokraError::InvalidArgument(_)) => {}
            other => panic!("saturating overflow must reject, got {other:?}"),
        }
    }
}
