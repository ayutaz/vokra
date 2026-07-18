//! Persistent CPU worker pool for row-parallel GEMM / GEMV (M1-12, feature
//! `parallel`; activates the deferred NFR-LC-03 threading decision).
//!
//! # What it is
//!
//! A process-wide pool of `std::thread` workers (created once, parked on a
//! [`Condvar`] between jobs) plus a synchronous "parallel-for over `0..ntasks`"
//! primitive ([`Pool::dispatch`]). [`parallel_gemm`] / [`parallel_gemv`] use it
//! to split a matmul over **disjoint output-row ranges**: task `t` writes rows
//! `[t*rpt, (t+1)*rpt)` of `out` and nothing else, running the *same* dispatched
//! kernel over its rows with the *same* `k`-reduction order as the
//! single-thread call. Because every output element is produced by the identical
//! FMA chain regardless of which thread runs it, the result is **bit-identical**
//! to the single-thread path (asserted by the `parallel_*_bit_identical_*`
//! differential tests) — so it cannot perturb the Whisper parity oracle.
//!
//! # Zero dependency (NFR-DS-02)
//!
//! `std` only — `std::thread` + `Mutex` + `Condvar` + `OnceLock`. No `rayon`,
//! no `crossbeam`; the workspace stays first-party-only.
//!
//! # WASM / single-core opt-out
//!
//! This module compiles only under `feature = "parallel"` **and** a non-WASM
//! target (WASM `std` has no thread spawning). `parallel` is default-on for
//! native desktop/server builds; a WASM or deliberately single-threaded build
//! disables it with `--no-default-features` (or omits it from the feature set),
//! and the kernels call the dispatched kernel inline. Even with the feature on,
//! [`global`] returns `None` when the host reports a single core
//! ([`std::thread::available_parallelism`] `== 1`), so a single-core host also
//! runs inline with no threads spawned. The `VOKRA_CPU_THREADS` env var caps the
//! pool at runtime — `1` forces the inline single-thread path (no threads
//! spawned), `N > 1` requests `N` participants — for shared/embedded hosts or a
//! single-threaded benchmark baseline.
//!
//! # Unsafe policy (NFR-RL-07, SRS §5-(1))
//!
//! Two `unsafe` bridges, each with a `// SAFETY:` justification:
//!
//! 1. **Lifetime erasure of the job closure.** A persistent (`'static`) worker
//!    must run a closure borrowed from the dispatcher's stack frame. Like
//!    `std::thread::scope`, we erase the closure to a `*const ()` + a
//!    monomorphised trampoline and guarantee — via the completion barrier — that
//!    the pointee outlives every worker's use of it. The barrier does not return
//!    until *all* tasks have incremented `finished`, and `finished` is bumped
//!    only *after* the task body returns (an RAII guard bumps it even on a
//!    panic-unwind, so a panicking task can never hang the barrier).
//! 2. **Disjoint output aliasing.** Tasks reconstruct their own `&mut [f32]`
//!    sub-slice from a shared base pointer ([`OutBase`]); the row ranges
//!    partition `0..m`, so no two tasks ever form overlapping `&mut`.
//!
//! No JIT / no runtime codegen (NFR-RL-05): the pool only *calls* the already
//! compiled dispatch-table kernels through a function pointer.

use std::num::NonZero;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

use crate::dispatch::{GemmKernel, GemvKernel};

// ---- tuning constants ----

/// Minimum multiply-accumulate count (`m*n*k`) below which a GEMM runs inline:
/// the fixed cost of waking workers + the completion barrier is only worth
/// paying once the matmul is large (the `m = 1500` Whisper-encoder GEMMs clear
/// this by ~2 orders of magnitude; the tiny per-token decode GEMMs stay inline).
const GEMM_MIN_MACS: usize = 1 << 20; // ~1.05M
/// Minimum `m*k` below which a GEMV runs inline. The tied-logits head
/// (`m = 51865`, `k = 512` ≈ 26M) clears it comfortably.
const GEMV_MIN_MACS: usize = 1 << 18; // ~262K
/// Row-chunks targeted **per participating thread**. Oversubscribing the thread
/// count lets the on-demand claim counter balance heterogeneous cores (Apple
/// Silicon P vs E, big.LITTLE): a fast core simply claims more chunks. Chunking
/// stays a pure row partition, so it never changes the numeric result.
const TASKS_PER_THREAD: usize = 4;
/// Never split below this many rows per task, so each task does enough work to
/// amortise its claim (one mutex round-trip) and the kernel's per-call prologue.
const MIN_ROWS_PER_TASK: usize = 8;

// ---- lifetime-erased job plumbing ----

/// A shared base pointer to the output buffer. Each task derives a disjoint,
/// non-overlapping `&mut [f32]` row range from it.
#[derive(Clone, Copy)]
struct OutBase(*mut f32);

// SAFETY: `OutBase` is only ever used to form `&mut [f32]` over *disjoint* row
// ranges (the tasks partition `0..m`), so it never yields two overlapping
// mutable references — sending/sharing the base pointer across the pool threads
// introduces no data race and no aliasing.
unsafe impl Send for OutBase {}
// SAFETY: see the `Send` impl — disjoint per-task ranges, no overlapping `&mut`.
unsafe impl Sync for OutBase {}

impl OutBase {
    /// The base pointer. Takes `self` by value so a job closure that calls it
    /// captures the whole `OutBase` (which is `Sync`) rather than the bare
    /// `*mut f32` field — the latter is what Rust 2021 disjoint closure capture
    /// would otherwise grab, and `*mut f32` is not `Sync`.
    fn as_ptr(self) -> *mut f32 {
        self.0
    }
}

/// Type-erased pointer to the borrowed job closure (`&F`). Not dereferenced
/// except inside [`trampoline`], and only while the dispatch barrier keeps the
/// pointee alive.
#[derive(Clone, Copy)]
struct DataPtr(*const ());

// SAFETY: the pointee is a `&F` with `F: Fn(usize) + Sync`; `F: Sync` makes the
// shared reference sound to read from any worker, and the dispatch barrier keeps
// it alive until every task has finished. The pointer is only ever turned back
// into a shared `&F` inside `trampoline`.
unsafe impl Send for DataPtr {}

/// Monomorphised trampoline: reconstruct `&F` from the erased pointer and run
/// task `i`. One instance per closure type `F`; stored as a plain function
/// pointer so the shared state carries no lifetime.
///
/// # Safety
/// `data` must point to a live `F` (guaranteed by the dispatch barrier) and `i`
/// must be a valid task index for that job.
unsafe fn trampoline<F: Fn(usize) + Sync>(data: *const (), i: usize) {
    // SAFETY: `data` was set from `f as *const F` for a `&F` that outlives this
    // call (dispatch barrier), and `F: Sync` makes the shared reference sound
    // on a worker thread.
    let f = unsafe { &*data.cast::<F>() };
    f(i);
}

/// Initial / cleared value for the job function pointer (no job posted).
///
/// # Safety
/// Never called (guarded by `active == false`); the arguments are ignored.
unsafe fn noop_run(_data: *const (), _i: usize) {}

/// Shared, mutex-protected control block plus the two condition variables.
struct Shared {
    ctl: Mutex<Ctl>,
    /// Workers wait here for a job to become available.
    work: Condvar,
    /// The dispatcher waits here for all tasks to finish.
    done: Condvar,
}

/// The current job's mutable state (guarded by [`Shared::ctl`]).
struct Ctl {
    job_data: DataPtr,
    job_run: unsafe fn(*const (), usize),
    ntasks: usize,
    /// Next task index to claim (on-demand work distribution).
    next: usize,
    /// Tasks that have finished running.
    finished: usize,
    /// A job is posted and its `job_data` pointer is live.
    active: bool,
    /// Teardown requested (set by [`Pool`]'s `Drop`).
    shutdown: bool,
}

// SAFETY: `Ctl` is only accessed under `Shared::ctl`'s mutex. Its sole
// non-`Send` field is `job_data` (a raw pointer), whose cross-thread use is made
// sound by the `DataPtr` `Send` impl above (pointee is `Sync` + kept alive by
// the barrier). This impl is what lets `Mutex<Ctl>` be `Sync`.
unsafe impl Send for Ctl {}

/// RAII: bump `finished` (and wake the dispatcher on the last task) when a task
/// body returns — **including on a panic-unwind**, so a panicking kernel can
/// never leave the completion barrier waiting forever.
struct TaskDone<'a> {
    shared: &'a Shared,
}

impl Drop for TaskDone<'_> {
    fn drop(&mut self) {
        let mut ctl = self.shared.ctl.lock().unwrap();
        ctl.finished += 1;
        if ctl.finished == ctl.ntasks {
            self.shared.done.notify_all();
        }
    }
}

/// Claim and run tasks from the current job until none remain. Called by both
/// the worker threads and the dispatcher (which participates so all cores work).
fn run_tasks(shared: &Shared) {
    loop {
        let (task, data, run) = {
            let mut ctl = shared.ctl.lock().unwrap();
            if !ctl.active || ctl.next >= ctl.ntasks {
                return;
            }
            let t = ctl.next;
            ctl.next += 1;
            (t, ctl.job_data, ctl.job_run)
        };
        // Bump `finished` even if the task body panics (prevents a barrier hang).
        let _done = TaskDone { shared };
        // SAFETY: `data`/`run` describe the job's live closure `&F`. The
        // dispatch barrier does not return until `finished == ntasks`, and each
        // task's `finished` bump happens only after `run` returns (or unwinds),
        // so the pointee outlives this deref; `F: Sync` makes the shared
        // reference sound on this thread.
        unsafe { run(data.0, task) };
    }
}

/// Long-lived worker: park on `work` until a job is available (or shutdown),
/// then help drain it.
fn worker_loop(shared: Arc<Shared>) {
    loop {
        {
            let mut ctl = shared.ctl.lock().unwrap();
            while !ctl.shutdown && (!ctl.active || ctl.next >= ctl.ntasks) {
                ctl = shared.work.wait(ctl).unwrap();
            }
            if ctl.shutdown {
                return;
            }
        }
        run_tasks(&shared);
    }
}

/// A persistent pool of worker threads plus the calling ("dispatcher") thread.
pub(crate) struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// Total participating threads = workers + the dispatcher.
    participants: usize,
    /// Serialises concurrent `dispatch` calls (the pool has a single job slot),
    /// so calling `gemm_f32` from several threads at once is safe (they run one
    /// at a time, each getting the whole pool). Uncontended in the normal
    /// single-threaded inference orchestration.
    dispatch_guard: Mutex<()>,
}

impl Pool {
    /// Create a pool with `participants` total threads (spawns `participants-1`
    /// long-lived workers; the caller is the last participant).
    fn new(participants: usize) -> Pool {
        let want = participants.max(1);
        let shared = Arc::new(Shared {
            ctl: Mutex::new(Ctl {
                job_data: DataPtr(std::ptr::null()),
                job_run: noop_run,
                ntasks: 0,
                next: 0,
                finished: 0,
                active: false,
                shutdown: false,
            }),
            work: Condvar::new(),
            done: Condvar::new(),
        });
        let mut workers = Vec::with_capacity(want - 1);
        for _ in 1..want {
            let sh = Arc::clone(&shared);
            match thread::Builder::new()
                .name("vokra-cpu".into())
                .spawn(move || worker_loop(sh))
            {
                Ok(h) => workers.push(h),
                // A spawn failure just means fewer workers; correctness is
                // unaffected (the dispatcher always participates).
                Err(_) => break,
            }
        }
        let participants = workers.len() + 1;
        Pool {
            shared,
            workers,
            participants,
            dispatch_guard: Mutex::new(()),
        }
    }

    /// Run `f(i)` for every `i` in `0..ntasks` across the pool (workers + the
    /// caller), blocking until all have completed.
    ///
    /// `f` is `Sync` (shared immutable capture); tasks coordinate disjoint
    /// mutable output through raw pointers at the call site (see [`OutBase`]).
    fn dispatch<F: Fn(usize) + Sync>(&self, ntasks: usize, f: &F) {
        if ntasks == 0 {
            return;
        }
        // One job slot → serialise concurrent dispatchers (uncontended in the
        // normal single-threaded orchestration).
        let _serial = self.dispatch_guard.lock().unwrap();
        {
            let mut ctl = self.shared.ctl.lock().unwrap();
            ctl.job_data = DataPtr((f as *const F).cast::<()>());
            ctl.job_run = trampoline::<F>;
            ctl.ntasks = ntasks;
            ctl.next = 0;
            ctl.finished = 0;
            ctl.active = true;
        }
        self.shared.work.notify_all();

        // The dispatcher participates so all cores do work.
        run_tasks(&self.shared);

        // Barrier: wait until every task (including worker-run ones) is done.
        let mut ctl = self.shared.ctl.lock().unwrap();
        while ctl.finished < ctl.ntasks {
            ctl = self.shared.done.wait(ctl).unwrap();
        }
        // Invalidate the borrowed-closure pointer before returning, so it can
        // never be dereferenced after `f`'s borrow ends.
        ctl.active = false;
        ctl.job_data = DataPtr(std::ptr::null());
        ctl.job_run = noop_run;
    }

    /// Total participating threads (workers + the dispatcher).
    fn participants(&self) -> usize {
        self.participants
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut ctl = self.shared.ctl.lock().unwrap();
            ctl.shutdown = true;
        }
        self.shared.work.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Environment variable to cap (or disable) the worker pool at runtime, e.g. on
/// shared/embedded hardware or for a single-threaded benchmark: `1` disables
/// threading (inline, no threads spawned), `N > 1` requests `N` participants.
/// Unset ⇒ the host's reported parallelism.
const ENV_THREADS: &str = "VOKRA_CPU_THREADS";

/// Desired total participant count. `VOKRA_CPU_THREADS` (when a valid `>= 1`
/// integer) overrides the host default; otherwise
/// [`std::thread::available_parallelism`].
fn desired_threads() -> usize {
    if let Ok(s) = std::env::var(ENV_THREADS)
        && let Ok(n) = s.trim().parse::<usize>()
    {
        return n.max(1);
    }
    thread::available_parallelism()
        .map(NonZero::get)
        .unwrap_or(1)
}

/// The process-wide pool, sized to [`desired_threads`]. `None` on a single-core
/// host or when `VOKRA_CPU_THREADS=1` (run inline; no threads spawned). Leaked
/// for the process lifetime (workers park until exit) — the `OnceLock` never
/// drops, matching how established thread pools manage their workers.
fn global() -> Option<&'static Pool> {
    static GLOBAL: OnceLock<Option<Pool>> = OnceLock::new();
    GLOBAL
        .get_or_init(|| {
            let n = desired_threads();
            if n <= 1 { None } else { Some(Pool::new(n)) }
        })
        .as_ref()
}

/// Rows per task: aim for [`TASKS_PER_THREAD`] chunks per participant (for
/// dynamic load balance), but never below [`MIN_ROWS_PER_TASK`].
fn rows_per_task(m: usize, participants: usize) -> usize {
    let target = participants.saturating_mul(TASKS_PER_THREAD).max(1);
    m.div_ceil(target).max(MIN_ROWS_PER_TASK)
}

/// Total participating threads of the global pool (1 when the pool is absent
/// — single-core host or `VOKRA_CPU_THREADS=1`). Used by the M5-14 packed
/// GEMM driver to size its task chunking; the value only affects scheduling
/// granularity, never results (disjoint-output determinism).
pub(crate) fn participants() -> usize {
    global().map_or(1, Pool::participants)
}

/// Run `f(i)` for every `i` in `0..ntasks` on the global pool (M5-14-T09:
/// the packed GEMM driver's generic parallel-for). Returns `false` — having
/// run nothing — when the pool is absent, so the caller can fall back to an
/// inline loop. Tasks are claimed on demand from the shared counter (the
/// same dynamic chunk queue `parallel_gemm` uses), which is what load-
/// balances heterogeneous P/E cores; correctness never depends on which
/// thread claims which task (callers write disjoint output ranges only).
pub(crate) fn run<F: Fn(usize) + Sync>(ntasks: usize, f: &F) -> bool {
    match global() {
        Some(pool) => {
            pool.dispatch(ntasks, f);
            true
        }
        None => false,
    }
}

/// Row-parallel GEMM: split `out`'s `m` rows across the pool, running `kernel`
/// (the dispatched single-thread GEMM) over each disjoint row range. Falls back
/// to a single inline `kernel` call when the pool is absent (single core) or the
/// matmul is too small to amortise threading. **Bit-identical** to the inline
/// call (same per-element FMA chain; only the row set differs per thread).
#[allow(clippy::too_many_arguments)] // mirrors the GEMM kernel signature
pub(crate) fn parallel_gemm(
    kernel: GemmKernel,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let pool = match global() {
        Some(p)
            if m >= 2
                && p.participants() >= 2
                && m.saturating_mul(n).saturating_mul(k) >= GEMM_MIN_MACS =>
        {
            p
        }
        _ => {
            kernel(m, n, k, a, b, bias, out);
            return;
        }
    };
    let rpt = rows_per_task(m, pool.participants());
    let ntasks = m.div_ceil(rpt);
    let out_base = OutBase(out.as_mut_ptr());
    let job = |task: usize| {
        let start = task * rpt;
        let end = (start + rpt).min(m);
        if start >= end {
            return;
        }
        let rows = end - start;
        let a_sub = &a[start * k..end * k];
        // SAFETY: `[start, end)` is this task's disjoint row range (tasks
        // partition `0..m`), and `end * n <= m * n == out.len()`, so this
        // `&mut [f32]` overlaps no other task's and stays inside `out`.
        let out_sub =
            unsafe { std::slice::from_raw_parts_mut(out_base.as_ptr().add(start * n), rows * n) };
        kernel(rows, n, k, a_sub, b, bias, out_sub);
    };
    pool.dispatch(ntasks, &job);
}

/// Row-parallel GEMV (the tied-logits-head fast path): split `out`'s `m` rows
/// across the pool. Per-row `bias` is sliced to the task's rows. Falls back to
/// an inline `kernel` call on a single core or a small problem. **Bit-identical**
/// to the inline call.
pub(crate) fn parallel_gemv(
    kernel: GemvKernel,
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let pool = match global() {
        Some(p) if m >= 2 && p.participants() >= 2 && m.saturating_mul(k) >= GEMV_MIN_MACS => p,
        _ => {
            kernel(m, k, a, x, bias, out);
            return;
        }
    };
    let rpt = rows_per_task(m, pool.participants());
    let ntasks = m.div_ceil(rpt);
    let out_base = OutBase(out.as_mut_ptr());
    let job = |task: usize| {
        let start = task * rpt;
        let end = (start + rpt).min(m);
        if start >= end {
            return;
        }
        let rows = end - start;
        let a_sub = &a[start * k..end * k];
        let bias_sub = bias.map(|b| &b[start..end]);
        // SAFETY: `[start, end)` is this task's disjoint row range (tasks
        // partition `0..m`) and `end <= m == out.len()`, so this `&mut [f32]`
        // overlaps no other task's and stays inside `out`.
        let out_sub = unsafe { std::slice::from_raw_parts_mut(out_base.as_ptr().add(start), rows) };
        kernel(rows, k, a_sub, x, bias_sub, out_sub);
    };
    pool.dispatch(ntasks, &job);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::scalar;

    /// Minimal xorshift PRNG (no `rand` dependency; NFR-DS-02), mirroring the
    /// differential harness.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
            (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
    }

    /// The generic parallel-for visits every index exactly once, from a mix of
    /// worker and dispatcher threads.
    #[test]
    fn dispatch_runs_every_task_exactly_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let pool = Pool::new(4);
        for &ntasks in &[0usize, 1, 3, 8, 100, 1000] {
            let hits: Vec<AtomicUsize> = (0..ntasks).map(|_| AtomicUsize::new(0)).collect();
            let job = |i: usize| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            };
            pool.dispatch(ntasks, &job);
            for (i, h) in hits.iter().enumerate() {
                assert_eq!(h.load(Ordering::Relaxed), 1, "task {i} of {ntasks}");
            }
        }
    }

    /// Row-splitting a GEMM over a real multi-worker pool is **bit-identical** to
    /// one inline `scalar::gemm`, for shapes whose row count forces several
    /// tasks (small `MIN_ROWS_PER_TASK`-sized chunks) and ragged final chunks.
    #[test]
    fn parallel_gemm_bit_identical_to_inline() {
        let pool = Pool::new(4);
        let mut rng = Rng::new(0x5AFE_0001);
        // (m, n, k) — m spans exact and ragged multiples of the chunk size.
        for &(m, n, k) in &[
            (1, 5, 4),
            (7, 3, 6),
            (64, 17, 9),
            (100, 8, 32),
            (257, 4, 40),
        ] {
            let a = rng.vec(m * k);
            let b = rng.vec(k * n);
            let bias = rng.vec(n);
            for bias_opt in [None, Some(bias.as_slice())] {
                let mut inline = vec![0.0f32; m * n];
                scalar::gemm(m, n, k, &a, &b, bias_opt, &mut inline);

                // Force the pool even for small shapes via a fixed 1-row chunk.
                let mut threaded = vec![0.0f32; m * n];
                run_gemm_forced(
                    &pool,
                    scalar::gemm,
                    1,
                    m,
                    n,
                    k,
                    &a,
                    &b,
                    bias_opt,
                    &mut threaded,
                );
                assert_eq!(
                    threaded,
                    inline,
                    "gemm {m}x{n}x{k} bias={}",
                    bias_opt.is_some()
                );
            }
        }
    }

    /// Same guarantee for the GEMV fast path (per-row bias sliced per task).
    #[test]
    fn parallel_gemv_bit_identical_to_inline() {
        let pool = Pool::new(4);
        let mut rng = Rng::new(0x5AFE_0002);
        for &(m, k) in &[(1, 7), (5, 4), (64, 33), (257, 512), (1000, 9)] {
            let a = rng.vec(m * k);
            let x = rng.vec(k);
            let bias = rng.vec(m);
            for bias_opt in [None, Some(bias.as_slice())] {
                let mut inline = vec![0.0f32; m];
                scalar::gemv(m, k, &a, &x, bias_opt, &mut inline);

                let mut threaded = vec![0.0f32; m];
                run_gemv_forced(
                    &pool,
                    scalar::gemv,
                    1,
                    m,
                    k,
                    &a,
                    &x,
                    bias_opt,
                    &mut threaded,
                );
                assert_eq!(threaded, inline, "gemv {m}x{k} bias={}", bias_opt.is_some());
            }
        }
    }

    // Test-only: split with an explicit `rpt`, always using the given pool
    // (bypasses the size threshold so small shapes still exercise threading).
    #[allow(clippy::too_many_arguments)]
    fn run_gemm_forced(
        pool: &Pool,
        kernel: GemmKernel,
        rpt: usize,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        let ntasks = m.div_ceil(rpt.max(1));
        let out_base = OutBase(out.as_mut_ptr());
        let job = |task: usize| {
            let start = task * rpt;
            let end = (start + rpt).min(m);
            if start >= end {
                return;
            }
            let rows = end - start;
            let a_sub = &a[start * k..end * k];
            // SAFETY: disjoint row range within `out` (len m*n).
            let out_sub = unsafe {
                std::slice::from_raw_parts_mut(out_base.as_ptr().add(start * n), rows * n)
            };
            kernel(rows, n, k, a_sub, b, bias, out_sub);
        };
        pool.dispatch(ntasks, &job);
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gemv_forced(
        pool: &Pool,
        kernel: GemvKernel,
        rpt: usize,
        m: usize,
        k: usize,
        a: &[f32],
        x: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        let ntasks = m.div_ceil(rpt.max(1));
        let out_base = OutBase(out.as_mut_ptr());
        let job = |task: usize| {
            let start = task * rpt;
            let end = (start + rpt).min(m);
            if start >= end {
                return;
            }
            let rows = end - start;
            let a_sub = &a[start * k..end * k];
            let bias_sub = bias.map(|b| &b[start..end]);
            // SAFETY: disjoint row range within `out` (len m).
            let out_sub =
                unsafe { std::slice::from_raw_parts_mut(out_base.as_ptr().add(start), rows) };
            kernel(rows, k, a_sub, x, bias_sub, out_sub);
        };
        pool.dispatch(ntasks, &job);
    }
}
