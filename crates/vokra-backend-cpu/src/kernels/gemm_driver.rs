//! Packed, cache-blocked GEMM driver (M5-14 Wave-1 T05/T09/T10).
//!
//! # Why (Wave-0 findings, `docs/adr/M5-14-cpu-hotpath.md` D2)
//!
//! The legacy register-blocked kernels stream `b` **unpacked** with a row
//! stride of `n` floats. On power-of-two strides (whisper-medium ffn:
//! `n = 4096` ⇒ 16 KiB row stride = the M1 L1 way size) that degrades to
//! ~18-21 GFLOP/s via L1 set-aliasing — the naive triple loop beats it —
//! while a packing GEMM (ORT-MLAS) holds 91–96 GFLOP/s on the same silicon
//! for the same shape. `m == 1` calls (decoder-step projections, Mimi's
//! per-frame transformer) fall into the kernels' latency-bound row tail at
//! ~6.5 GFLOP/s where a streaming row formulation reaches ≥ 18.
//!
//! # What
//!
//! For eligible shapes this driver reorganises the SAME arithmetic into the
//! classic Goto/BLIS blocking:
//!
//! - `b` is packed per `(NC-column slab, KC-k-block)` into contiguous
//!   `NR`-column strips (`[l][NR]`, unit stride — no more `n`-strided loads,
//!   which is precisely what kills the power-of-two aliasing);
//! - `a` is packed per `(MC-row tile, KC-block)` into `MR`-row strips
//!   (`[l][MR]`);
//! - a per-ISA micro-kernel (see [`crate::dispatch::PackedGemm`]) computes an
//!   `MR × NR` output tile per strip pair, seeding from `bias` (or zero) on
//!   the first k-block and **accumulating into `out` across k-blocks in
//!   ascending k order**.
//!
//! # Parity invariant (the M5-14 red-line)
//!
//! Every output element is produced by the **identical per-element chain** as
//! the legacy kernel: one accumulator, seeded from `bias[j]` / `0.0`,
//! advanced over `l = 0..k` in ascending order with the same operation the
//! legacy kernel used for that column region — fused FMA in the vector
//! region, plain `mul`+`add` in the scalar column tail. Splitting the k loop
//! at `KC` boundaries only round-trips the f32 partial through memory, which
//! is exact; packing changes where operands are READ from, never their
//! values or order. The packed path is therefore **bit-identical** to the
//! legacy kernel for every shape, which is asserted per element by
//! `tests/gemm_packed_parity.rs` and keeps every committed model parity
//! fixture byte-stable. The same argument makes results independent of
//! thread count and scheduling (disjoint output tiles, fixed per-tile order).
//!
//! # Threading (T09)
//!
//! Within a `(slab, k-block)` round the B-pack is chunked across the pool,
//! then `MC`-row tiles are dispatched as tasks over the pool's on-demand
//! claim queue (heterogeneous P/E cores balance by claiming at different
//! rates). Tiles own disjoint `out` rows ⇒ bit-determinism by construction.
//!
//! # Scratch (allocation policy)
//!
//! Pack buffers are **thread-local, grow-only** (`RefCell<Vec<f32>>`):
//! - the dispatching thread holds the B-slab buffer (≤ `(NC + NR_MAX) * KC`
//!   floats ≈ 2.1 MiB — bounded by the blocking constants, NOT by the model);
//! - each pool worker holds its own A-tile buffer (`MC * KC` floats =
//!   128 KiB).
//!
//! Buffers are reused across calls (steady-state allocation-free after
//! warm-up, FR-EX-05 posture); `m == 1` and legacy routes use no scratch at
//! all, so the whisper decode loop's zero-alloc guarantee is untouched.
//!
//! No JIT (NFR-RL-05), no new dependencies (NFR-DS-02): plain `std`,
//! `thread_local!`, and the existing self-built pool.

use std::cell::RefCell;

use crate::dispatch::{self, GemmM1Kernel, KernelTable, PackedGemm, PackedTail};

// ---- blocking constants (empirical on Apple M1; see the Wave-1 report) ----
//
// KC sizes the packed strips: one B strip is `KC * NR * 4` bytes (32 KiB at
// NR = 8) and one A strip `KC * MR * 4` (32 KiB) — resident in the M1 P-core
// 128 KiB L1d with room for the C tile. MC bounds the per-task A tile
// (`MC * KC * 4` = 256 KiB, L2-resident) and sets the parallel task
// granularity (m = 1500 ⇒ 24 tasks ⇒ ≥ 3 claims per thread on 8 cores for
// dynamic balance). NC bounds the packed B slab (`KC * NC * 4` = 4 MiB ≪ the
// 12 MiB shared P-cluster L2). Chosen by microbenchmark sweep on the Wave-0
// hot shapes (KC ∈ {256, 512, 1024} × NC ∈ {1024, 2048} × MC ∈ {64, 128},
// M5-14 Wave-1 report): 1T GFLOP/s was flat within ~5% across the grid
// (84–88 on every shape incl. the 16 KiB-stride pathological one); KC = 1024
// measured best at 1T AND 8T (≈87 / ≈375), NC = 2048 added ≈2% at 8T but
// doubles the per-thread slab scratch to 8 MiB — not taken. These values are
// performance tuning only — by the parity invariant above they CANNOT affect
// results (asserted by the blocking-boundary shapes in the parity tests).

/// Rows per A tile = rows per parallel task (multiple of [`PACK_MR`]).
pub const MC: usize = 64;
/// k-block depth: strips stay L1-resident.
pub const KC: usize = 1024;
/// Columns per packed B slab (multiple of every micro-kernel NR).
pub const NC: usize = 1024;
/// A-panel row-strip height shared by every packed micro-kernel.
pub const PACK_MR: usize = 8;
/// Widest NR any micro-kernel uses (AVX-512); sizes the slab scratch bound.
const NR_MAX: usize = 16;

// ---- routing thresholds ----

/// Minimum m·n·k before packing pays for itself (below: the pack copies +
/// scratch bookkeeping dominate; the legacy kernel is already fine there).
/// Measured on M1 (Wave-1 sweep, 1T): at 131K MACs (16,128,64) packed is
/// ~8% BEHIND legacy; at 262K (32,128,64) it is ~20% ahead and the gap only
/// widens with size — so the gate sits at the measured break-even.
const PACKED_MIN_MACS: usize = 1 << 18;
/// Packing needs at least two column strips' worth of width to matter.
const PACKED_MIN_N: usize = 16;
/// Very shallow k has nothing to block; the legacy kernel handles it.
const PACKED_MIN_K: usize = 8;
/// m·n·k gate below which the packed path stays single-thread (mirrors the
/// legacy `pool::GEMM_MIN_MACS` so the pool-eligibility boundary is
/// unchanged by this driver).
const PAR_MIN_MACS: usize = 1 << 20;
/// n·k gate for column-splitting an `m == 1` call across the pool. The row
/// path is memory-bound, so threads only help once `b` is large enough that
/// aggregate bandwidth beats a single core (measured on M1: the win appears
/// around the whisper-medium decoder fc shapes, n·k ≈ 4M).
const M1_PAR_MIN_MACS: usize = 1 << 21;

// ---- thread-local pack scratch (grow-only, reused across calls) ----

thread_local! {
    /// B-slab pack buffer — lives on whichever thread dispatches the GEMM.
    static PACK_B_TLS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// A-tile pack buffer — one per pool worker (and the dispatcher).
    static PACK_A_TLS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

// ---- parallel-for shim (pool when available, inline otherwise) ----

#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
fn par_for<F: Fn(usize) + Sync>(ntasks: usize, f: &F) {
    if ntasks >= 2 && crate::pool::run(ntasks, f) {
        return;
    }
    for i in 0..ntasks {
        f(i);
    }
}

#[cfg(not(all(feature = "parallel", not(target_family = "wasm"))))]
fn par_for<F: Fn(usize) + Sync>(ntasks: usize, f: &F) {
    for i in 0..ntasks {
        f(i);
    }
}

#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
fn pool_participants() -> usize {
    crate::pool::participants()
}

#[cfg(not(all(feature = "parallel", not(target_family = "wasm"))))]
fn pool_participants() -> usize {
    1
}

/// Shared base pointer for disjoint-range `&mut` reconstruction inside pool
/// tasks (the `pool::OutBase` pattern).
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);

// SAFETY: `SendPtr` is only ever used to form `&mut [f32]` over ranges that
// are DISJOINT per task (column chunks of `out`, strip chunks of the pack
// buffer, row tiles of `out`), so no two tasks ever hold overlapping `&mut`;
// sharing the raw base across pool threads is race-free.
unsafe impl Send for SendPtr {}
// SAFETY: see `Send` — disjoint per-task ranges only.
unsafe impl Sync for SendPtr {}

impl SendPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

// ---- routing ----

/// Whether `(m, n, k)` takes the packed cache-blocked path on an ISA that
/// provides packed micro-kernels. Pure and shape-only, so tests can pin the
/// hot Wave-0 shapes to the packed route (`gemm_test_probe`). Thresholds are
/// performance routing only: every route computes bit-identical results.
pub fn would_use_packed(m: usize, n: usize, k: usize) -> bool {
    m >= PACK_MR
        && n >= PACKED_MIN_N
        && k >= PACKED_MIN_K
        && m.saturating_mul(n).saturating_mul(k) >= PACKED_MIN_MACS
}

/// Whether the active dispatch table carries packed micro-kernels.
pub fn active_gemm_has_packed() -> bool {
    dispatch::table().gemm_packed.is_some()
}

/// Whether the active dispatch table carries the m == 1 row kernel.
pub fn active_gemm_has_m1() -> bool {
    dispatch::table().gemm_m1.is_some()
}

/// Production GEMM entry (routes `kernels::gemm_f32`). Inputs are
/// pre-validated by the public wrapper.
#[allow(clippy::too_many_arguments)] // GEMM's intrinsic parameter set
pub(crate) fn run(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let t = dispatch::table();
    // T10: m == 1 row fast path (decoder-step projections, per-frame loops).
    if m == 1
        && let Some(m1) = t.gemm_m1
    {
        m1_path(m1, n, k, a, b, bias, out);
        return;
    }
    // T05/T08/T09: packed cache-blocked path.
    if let Some(pk) = t.gemm_packed
        && would_use_packed(m, n, k)
    {
        packed_path(&pk, m, n, k, a, b, bias, out);
        return;
    }
    legacy_path(t, m, n, k, a, b, bias, out);
}

/// The pre-Wave-1 production behaviour: pool row-split over the legacy
/// kernel (bit-identical to inline), or inline off-`parallel` / on WASM.
#[allow(clippy::too_many_arguments)] // GEMM's intrinsic parameter set
fn legacy_path(
    t: &KernelTable,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
    {
        crate::pool::parallel_gemm(t.gemm, m, n, k, a, b, bias, out);
    }
    #[cfg(not(all(feature = "parallel", not(target_family = "wasm"))))]
    {
        (t.gemm)(m, n, k, a, b, bias, out);
    }
}

// ---- T10: m == 1 row path ----

/// `m == 1` GEMM through the ISA row kernel, column-split across the pool
/// when `b` is large enough for aggregate bandwidth to win. Chunk boundaries
/// are 16-column aligned so every chunk's internal vector/scalar column
/// regions coincide with the full-width call's regions (per-element
/// bit-identity is preserved chunk-wise for every ISA lane width).
fn m1_path(
    m1: GemmM1Kernel,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let participants = pool_participants();
    if participants >= 2 && n.saturating_mul(k) >= M1_PAR_MIN_MACS && n >= 64 {
        // ~2 chunks per participant, 16-aligned, ≥ 64 cols each.
        let target = participants * 2;
        let step = (n.div_ceil(target)).next_multiple_of(16).max(64);
        let ntasks = n.div_ceil(step);
        if ntasks >= 2 {
            let out_base = SendPtr(out.as_mut_ptr());
            let job = |task: usize| {
                let j0 = task * step;
                let j1 = (j0 + step).min(n);
                if j0 >= j1 {
                    return;
                }
                let cols = j1 - j0;
                // SAFETY: `[j0, j1)` ranges partition `0..n`, so this `&mut`
                // is disjoint from every other task's and stays inside `out`
                // (`j1 <= n == out.len()`).
                let out_sub =
                    unsafe { std::slice::from_raw_parts_mut(out_base.get().add(j0), cols) };
                let bias_sub = bias.map(|bs| &bs[j0..j1]);
                m1(cols, k, n, a, &b[j0..], bias_sub, out_sub);
            };
            par_for(ntasks, &job);
            return;
        }
    }
    m1(n, k, n, a, b, bias, out);
}

// ---- T05: pack routines (shared by every ISA; plain safe Rust) ----

/// Packs the B block `b[p0..p0+kc, j0..j0+cols]` (row-major `b[k, n]`) into
/// `nr`-column strips: `out[s*kc*nr + l*nr + c] = b[(p0+l)*n + j0 + s*nr + c]`,
/// zero for columns past `cols` in the final strip. Writes every element of
/// the `ceil(cols/nr)*kc*nr` prefix of `out` (padding included — the buffer
/// is reused across calls and must not leak stale values).
#[allow(clippy::too_many_arguments)] // the pack's intrinsic parameter set
fn pack_b_block(
    b: &[f32],
    n: usize,
    p0: usize,
    kc: usize,
    j0: usize,
    cols: usize,
    nr: usize,
    out: &mut [f32],
) {
    let strips = cols.div_ceil(nr);
    for s in 0..strips {
        let c0 = j0 + s * nr;
        let valid = nr.min(j0 + cols - c0);
        let dst_strip = &mut out[s * kc * nr..(s + 1) * kc * nr];
        if valid == nr {
            for l in 0..kc {
                let src = &b[(p0 + l) * n + c0..(p0 + l) * n + c0 + nr];
                dst_strip[l * nr..l * nr + nr].copy_from_slice(src);
            }
        } else {
            for l in 0..kc {
                let src = &b[(p0 + l) * n + c0..(p0 + l) * n + c0 + valid];
                let dst = &mut dst_strip[l * nr..l * nr + nr];
                dst[..valid].copy_from_slice(src);
                dst[valid..].fill(0.0);
            }
        }
    }
}

/// Packs the A tile `a[i0..i0+rows, p0..p0+kc]` (row-major `a[m, k]`) into
/// [`PACK_MR`]-row strips: `out[rs*kc*MR + l*MR + r] = a[(i0+rs*MR+r)*k + p0+l]`,
/// zero for rows past `rows` in the final strip (padded rows are computed by
/// the micro-kernel but never stored). Writes every element of the used
/// prefix (see [`pack_b_block`] on buffer reuse).
fn pack_a_tile(a: &[f32], k: usize, i0: usize, rows: usize, p0: usize, kc: usize, out: &mut [f32]) {
    const MR: usize = PACK_MR;
    let strips = rows.div_ceil(MR);
    for rs in 0..strips {
        let r0 = rs * MR;
        let valid = MR.min(rows - r0);
        let dst_strip = &mut out[rs * kc * MR..(rs + 1) * kc * MR];
        for l in 0..kc {
            let dst = &mut dst_strip[l * MR..l * MR + MR];
            for (r, d) in dst.iter_mut().enumerate() {
                *d = if r < valid {
                    a[(i0 + r0 + r) * k + p0 + l]
                } else {
                    0.0
                };
            }
        }
    }
}

// ---- T05/T09: the packed cache-blocked path ----

/// Column coverage plan for one ISA's tail convention (see
/// [`PackedTail`]): full `nr` strips, then per-ISA tail handling chosen to
/// reproduce the legacy kernel's per-element operation for every column.
struct ColPlan {
    /// Columns covered by full `nr`-wide strips (`floor(n / nr) * nr`).
    full_end: usize,
    /// Columns covered by the packed region in total (full strips + the
    /// 4-wide NEON strip or the AVX-512 masked strip).
    packed_end: usize,
    /// Valid columns of the extra (4-wide or masked) strip; 0 = none.
    extra_cols: usize,
    /// Packed width of the extra strip (4 for `Vec4`, `nr` for `Masked`).
    extra_width: usize,
}

fn col_plan(pk: &PackedGemm, n: usize) -> ColPlan {
    let full_end = (n / pk.nr) * pk.nr;
    match pk.tail {
        // Legacy NEON covers [n8, n4) with 4-wide FMA, then scalar mul+add.
        PackedTail::Vec4 => {
            let n4 = (n / 4) * 4;
            ColPlan {
                full_end,
                packed_end: n4,
                extra_cols: n4 - full_end,
                extra_width: 4,
            }
        }
        // Legacy AVX2 goes scalar mul+add straight after the 8-wide region.
        PackedTail::Scalar => ColPlan {
            full_end,
            packed_end: full_end,
            extra_cols: 0,
            extra_width: pk.nr,
        },
        // Legacy AVX-512 is FMA everywhere: one zero-padded strip with a
        // masked store covers the final `n % nr` columns.
        PackedTail::Masked => ColPlan {
            full_end,
            packed_end: n,
            extra_cols: n - full_end,
            extra_width: pk.nr,
        },
    }
}

#[allow(clippy::too_many_arguments)] // GEMM's intrinsic parameter set
fn packed_path(
    pk: &PackedGemm,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert!(k >= 1, "routing guarantees k >= 1 on the packed path");
    let plan = col_plan(pk, n);
    let nr = pk.nr;
    let micro = pk.micro;

    /// How the packed compute is spread over the pool. Pure scheduling — by
    /// the parity invariant every mode produces identical bits.
    #[derive(PartialEq, Clone, Copy)]
    enum ParMode {
        /// Tall m: tasks own disjoint MC-row tiles (each packs its own A).
        RowTiles,
        /// Below the pool gate / no pool: everything on the calling thread.
        Seq,
    }
    let pool_eligible =
        pool_participants() >= 2 && m.saturating_mul(n).saturating_mul(k) >= PAR_MIN_MACS;
    if pool_eligible && m <= MC {
        // Thin m (the CAM++ / conv-im2col family): the row dimension cannot
        // feed the pool, so parallelise over disjoint column-strip ranges
        // instead (single dispatch; each task packs its own B sub-blocks).
        packed_thin_m(pk, &plan, m, n, k, a, b, bias, out);
        if plan.packed_end < n {
            scalar_col_tail(m, n, k, a, b, bias, out, plan.packed_end);
        }
        return;
    }
    let par_mode = if pool_eligible {
        ParMode::RowTiles
    } else {
        ParMode::Seq
    };

    let m_tasks = m.div_ceil(MC);
    let a_tile_elems = MC * KC; // MC is a multiple of PACK_MR
    let slab_buf_elems = (NC / nr) * nr * KC + NR_MAX * KC; // strips + extra strip

    PACK_B_TLS.with(|cell| {
        let mut bbuf = cell.borrow_mut();
        if bbuf.len() < slab_buf_elems {
            bbuf.resize(slab_buf_elems, 0.0);
        }
        let bbuf = &mut bbuf[..];

        // Slabs partition the full-strip region; the extra (4-wide / masked)
        // strip rides with the last slab (or forms its own when full_end is
        // a slab multiple). `NC` is a multiple of every `nr`.
        let mut j0 = 0;
        loop {
            let jend = (j0 + NC).min(plan.full_end);
            let slab_cols = jend - j0; // multiple of nr
            let is_last_slab = jend == plan.full_end;
            let extra = if is_last_slab { plan.extra_cols } else { 0 };
            if slab_cols == 0 && extra == 0 {
                break;
            }
            let n_strips = slab_cols / nr;
            let extra_off = n_strips * KC * nr; // extra strip's slab offset

            let mut p0 = 0;
            while p0 < k {
                let kc = KC.min(k - p0);
                let first_block = p0 == 0;

                // ---- pack the B slab (parallel over strip chunks) ----
                {
                    let btarget = SendPtr(bbuf.as_mut_ptr());
                    let chunk = n_strips.div_ceil(pool_participants() * 2).max(1);
                    let pack_tasks = if n_strips == 0 {
                        0
                    } else {
                        n_strips.div_ceil(chunk)
                    };
                    let pack_job = |task: usize| {
                        let s0 = task * chunk;
                        let s1 = (s0 + chunk).min(n_strips);
                        if s0 >= s1 {
                            return;
                        }
                        // SAFETY: strip ranges `[s0, s1)` partition
                        // `0..n_strips`, so each task's sub-slice
                        // `[s0*kc*nr, s1*kc*nr)` is disjoint and inside the
                        // `slab_buf_elems`-sized buffer.
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(
                                btarget.get().add(s0 * kc * nr),
                                (s1 - s0) * kc * nr,
                            )
                        };
                        pack_b_block(b, n, p0, kc, j0 + s0 * nr, (s1 - s0) * nr, nr, dst);
                    };
                    if par_mode != ParMode::Seq && pack_tasks >= 2 {
                        par_for(pack_tasks, &pack_job);
                    } else {
                        for t in 0..pack_tasks {
                            pack_job(t);
                        }
                    }
                    if extra > 0 {
                        // The 4-wide / masked tail strip (zero-padded).
                        pack_b_block(
                            b,
                            n,
                            p0,
                            kc,
                            plan.full_end,
                            extra,
                            plan.extra_width,
                            &mut bbuf[extra_off..extra_off + kc * plan.extra_width],
                        );
                    }
                }
                let bslab: &[f32] = &bbuf[..];

                // ---- compute: MC-row tiles over the pool's claim queue ----
                let out_base = SendPtr(out.as_mut_ptr());
                let tile_job = |task: usize| {
                    let i0 = task * MC;
                    let rows = MC.min(m - i0);
                    if i0 >= m {
                        return;
                    }
                    PACK_A_TLS.with(|acell| {
                        let mut abuf = acell.borrow_mut();
                        if abuf.len() < a_tile_elems {
                            abuf.resize(a_tile_elems, 0.0);
                        }
                        pack_a_tile(a, k, i0, rows, p0, kc, &mut abuf[..]);
                        let row_strips = rows.div_ceil(PACK_MR);
                        // Column strips outer / row strips inner: the B strip
                        // stays L1-resident across the row strips of the tile.
                        for s in 0..n_strips {
                            let jcol = j0 + s * nr;
                            let bstrip = &bslab[s * kc * nr..(s + 1) * kc * nr];
                            for rs in 0..row_strips {
                                let r0 = i0 + rs * PACK_MR;
                                let tile_rows = PACK_MR.min(m - r0);
                                let astrip = &abuf[rs * kc * PACK_MR..(rs + 1) * kc * PACK_MR];
                                let bias_s = if first_block {
                                    bias.map(|bs| &bs[jcol..])
                                } else {
                                    None
                                };
                                // SAFETY: the C tile origin `r0 * n + jcol`
                                // is inside `out` (r0 < m, jcol + nr <= n),
                                // this task owns rows `[i0, i0 + rows)`
                                // exclusively (tasks partition `0..m` by MC
                                // blocks), and the micro-kernel writes only
                                // `tile_rows` rows × `nr` cols from that
                                // origin. `astrip`/`bstrip` carry `kc*MR` /
                                // `kc*nr` packed elements as required.
                                unsafe {
                                    micro(
                                        kc,
                                        astrip,
                                        bstrip,
                                        out_base.get().add(r0 * n + jcol),
                                        n,
                                        tile_rows,
                                        nr,
                                        bias_s,
                                        !first_block,
                                    );
                                }
                            }
                        }
                        if extra > 0 {
                            let jcol = plan.full_end;
                            let bstrip = &bslab[extra_off..extra_off + kc * plan.extra_width];
                            for rs in 0..row_strips {
                                let r0 = i0 + rs * PACK_MR;
                                let tile_rows = PACK_MR.min(m - r0);
                                let astrip = &abuf[rs * kc * PACK_MR..(rs + 1) * kc * PACK_MR];
                                let bias_s = if first_block {
                                    bias.map(|bs| &bs[jcol..])
                                } else {
                                    None
                                };
                                // SAFETY: as above; the micro-kernel writes
                                // only `extra` (< extra_width) columns at the
                                // tile origin, which stays inside row `r0`'s
                                // `n` columns (`jcol + extra <= n`).
                                unsafe {
                                    micro(
                                        kc,
                                        astrip,
                                        bstrip,
                                        out_base.get().add(r0 * n + jcol),
                                        n,
                                        tile_rows,
                                        extra,
                                        bias_s,
                                        !first_block,
                                    );
                                }
                            }
                        }
                    });
                };
                match par_mode {
                    ParMode::RowTiles => par_for(m_tasks, &tile_job),
                    ParMode::Seq => {
                        for t in 0..m_tasks {
                            tile_job(t);
                        }
                    }
                }

                p0 += kc;
            }

            if is_last_slab {
                break;
            }
            j0 = jend;
        }
    });

    // ---- scalar column tail [packed_end, n): the legacy kernels' plain
    // mul+add chains, computed straight from the unpacked operands ----
    if plan.packed_end < n {
        scalar_col_tail(m, n, k, a, b, bias, out, plan.packed_end);
    }
}

/// Thin-m packed path (`m <= MC`, pool-eligible): the row dimension cannot
/// feed the pool (the legacy row-split had at most `m / 8` tasks here and the
/// CAM++-family im2col GEMMs ran effectively single-threaded), so tasks own
/// **disjoint column-strip ranges** of C instead.
///
/// Single pool dispatch for the whole GEMM: the dispatcher packs the FULL A
/// once (`m <= MC` keeps it ≤ `MC × k` floats), then every task walks the k
/// blocks in ascending order, packing its own B sub-blocks (bounded worker
/// scratch: [`THIN_SB`] strips × `KC`) and running the micro-kernel over its
/// strips. C columns are disjoint per task and each column's k-accumulation
/// order is fixed (`pc` ascending inside the task), so the result is
/// bit-identical to every other route regardless of scheduling.
#[allow(clippy::too_many_arguments)] // GEMM's intrinsic parameter set
fn packed_thin_m(
    pk: &PackedGemm,
    plan: &ColPlan,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    /// B sub-block width (in strips) a task packs and computes at a time —
    /// bounds the per-worker B scratch at `THIN_SB * NR_MAX * KC` floats
    /// (1 MiB) while keeping the packed sub-block L2-resident.
    const THIN_SB: usize = 16;

    let nr = pk.nr;
    let micro = pk.micro;
    let row_strips = m.div_ceil(PACK_MR);
    let strip_elems = row_strips * PACK_MR * KC; // per-k-block A pack stride
    let k_blocks = k.div_ceil(KC);
    let n_strips = plan.full_end / nr;
    let extra = plan.extra_cols;

    PACK_A_TLS.with(|acell| {
        let mut abuf = acell.borrow_mut();
        let a_elems = k_blocks * strip_elems;
        if abuf.len() < a_elems {
            abuf.resize(a_elems, 0.0);
        }
        // Pack the full A once: k-block `c` at offset `c * strip_elems`
        // (uniform KC-based stride; the final block packs its shorter kc).
        for (c, p0) in (0..k).step_by(KC).enumerate() {
            let kc = KC.min(k - p0);
            pack_a_tile(a, k, 0, m, p0, kc, &mut abuf[c * strip_elems..]);
        }
        let apack: &[f32] = &abuf[..];

        // Work units: full strips + the extra (4-wide / masked) strip last.
        let total_units = n_strips + usize::from(extra > 0);
        let chunk = total_units
            .div_ceil(pool_participants() * 2)
            .clamp(1, THIN_SB);
        let ntasks = total_units.div_ceil(chunk);
        let out_base = SendPtr(out.as_mut_ptr());
        let job = |task: usize| {
            let u0 = task * chunk;
            let u1 = (u0 + chunk).min(total_units);
            if u0 >= u1 {
                return;
            }
            PACK_B_TLS.with(|bcell| {
                let mut bbuf = bcell.borrow_mut();
                let b_elems = THIN_SB * NR_MAX * KC;
                if bbuf.len() < b_elems {
                    bbuf.resize(b_elems, 0.0);
                }
                // k blocks ascending: every C column in this task's range
                // accumulates in the same fixed order as every other route.
                for (c, p0) in (0..k).step_by(KC).enumerate() {
                    let kc = KC.min(k - p0);
                    let first_block = p0 == 0;
                    let ablock = &apack[c * strip_elems..];
                    for u in u0..u1 {
                        let (jcol, width, ncols) = if u < n_strips {
                            (u * nr, nr, nr)
                        } else {
                            (plan.full_end, plan.extra_width, extra)
                        };
                        // Per-unit region of the worker's B scratch, packed
                        // fresh for this (task, k-block) pass.
                        let bstrip = &mut bbuf[(u - u0) * NR_MAX * KC..];
                        pack_b_block(b, n, p0, kc, jcol, ncols, width, bstrip);
                        let bstrip: &[f32] = &bstrip[..kc * width];
                        for rs in 0..row_strips {
                            let r0 = rs * PACK_MR;
                            let tile_rows = PACK_MR.min(m - r0);
                            let astrip = &ablock[rs * kc * PACK_MR..(rs + 1) * kc * PACK_MR];
                            let bias_s = if first_block {
                                bias.map(|bs| &bs[jcol..])
                            } else {
                                None
                            };
                            // SAFETY: unit ranges partition the strips, so
                            // this task's columns `[jcol, jcol + ncols)` are
                            // disjoint from every other task's; the tile
                            // origin `r0 * n + jcol` is inside `out`
                            // (`r0 < m`, `jcol + ncols <= n`), and the
                            // micro-kernel touches only `tile_rows` rows ×
                            // `ncols` columns from it. `astrip` / `bstrip`
                            // carry the packed element counts the contract
                            // requires.
                            unsafe {
                                micro(
                                    kc,
                                    astrip,
                                    bstrip,
                                    out_base.get().add(r0 * n + jcol),
                                    n,
                                    tile_rows,
                                    ncols,
                                    bias_s,
                                    !first_block,
                                );
                            }
                        }
                    }
                }
            });
        };
        par_for(ntasks, &job);
    });
}

/// The legacy scalar column tail: for every row, columns `[j_from, n)` are
/// single-accumulator plain `mul`+`add` chains over ascending `l` — exactly
/// the legacy NEON / AVX2 kernels' scalar remainder (bit-identical).
#[allow(clippy::too_many_arguments)] // GEMM's intrinsic parameter set
fn scalar_col_tail(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    j_from: usize,
) {
    for i in 0..m {
        for j in j_from..n {
            let mut s = bias.map_or(0.0, |bs| bs[j]);
            for l in 0..k {
                s += a[i * k + l] * b[l * n + j];
            }
            out[i * n + j] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_b_block_strips_and_zero_pads() {
        // b is 3x5 row-major; pack cols [1, 5) (4 cols) as nr = 4 → 1 strip.
        let n = 5;
        let b: Vec<f32> = (0..15).map(|v| v as f32).collect();
        let mut out = [f32::NAN; 3 * 4];
        pack_b_block(&b, n, 0, 3, 1, 4, 4, &mut out);
        #[rustfmt::skip]
        let want = [
            1.0, 2.0, 3.0, 4.0,
            6.0, 7.0, 8.0, 9.0,
            11.0, 12.0, 13.0, 14.0,
        ];
        assert_eq!(out, want);

        // cols [3, 5) (2 valid) at nr = 4 → zero-padded final strip.
        let mut out = [f32::NAN; 3 * 4];
        pack_b_block(&b, n, 1, 2, 3, 2, 4, &mut out[..2 * 4]);
        assert_eq!(&out[..8], &[8.0, 9.0, 0.0, 0.0, 13.0, 14.0, 0.0, 0.0]);
    }

    #[test]
    fn pack_a_tile_strips_and_zero_pads_rows() {
        // a is 3x4 row-major; tile rows [0, 3), k block [1, 3) → 1 strip of
        // MR = 8 with rows 3..8 zero.
        let k = 4;
        let a: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let mut out = [f32::NAN; 2 * PACK_MR];
        pack_a_tile(&a, k, 0, 3, 1, 2, &mut out);
        // l = 0 → a[., 1] = [1, 5, 9], l = 1 → a[., 2] = [2, 6, 10].
        let want = [
            1.0, 5.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            2.0, 6.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(out, want);
    }

    #[test]
    fn col_plan_matches_legacy_regions() {
        let neon = PackedGemm {
            nr: 8,
            micro: dummy_micro,
            tail: PackedTail::Vec4,
        };
        // n = 23: full strips to 16, 4-strip [16, 20), scalar [20, 23).
        let p = col_plan(&neon, 23);
        assert_eq!((p.full_end, p.packed_end, p.extra_cols), (16, 20, 4));
        // n = 24: full strips cover everything.
        let p = col_plan(&neon, 24);
        assert_eq!((p.full_end, p.packed_end, p.extra_cols), (24, 24, 0));

        let avx2 = PackedGemm {
            nr: 8,
            micro: dummy_micro,
            tail: PackedTail::Scalar,
        };
        let p = col_plan(&avx2, 23);
        assert_eq!((p.full_end, p.packed_end, p.extra_cols), (16, 16, 0));

        let avx512 = PackedGemm {
            nr: 16,
            micro: dummy_micro,
            tail: PackedTail::Masked,
        };
        let p = col_plan(&avx512, 23);
        assert_eq!((p.full_end, p.packed_end, p.extra_cols), (16, 23, 7));
    }

    /// Type-checking stand-in for the plan tests (never called).
    #[allow(clippy::too_many_arguments)] // mirrors the micro-kernel signature
    unsafe fn dummy_micro(
        _kc: usize,
        _ap: &[f32],
        _bp: &[f32],
        _c: *mut f32,
        _ldc: usize,
        _rows: usize,
        _ncols: usize,
        _bias: Option<&[f32]>,
        _accumulate: bool,
    ) {
        unreachable!("plan-only test double");
    }
}
