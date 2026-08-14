//! M4-05 T18 — allocation-count proof that the CSM streaming frame loop is
//! malloc-free (FR-EX-05): backbone step (paged KV off the pre-allocated
//! free list) → greedy sampling → depth transformer → paged RVQ decode →
//! Mimi neural decode → PCM chunk, per frame, with **zero** heap
//! allocations after the stream is opened.
//!
//! The M4-03 `aec_hot_path_alloc.rs` counting-`#[global_allocator]`
//! pattern (a global allocator is per-binary, hence a dedicated
//! integration-test target).
//!
//! Scope note (honest): the proof runs the **greedy** sampler — the M1
//! `Sampler`'s stochastic top-k draw allocates internally (a pre-existing
//! M1 property outside this WP's blast radius; noted in
//! `csm::streaming` module docs). The error paths may `format!` — errors
//! are rare and off the hot path (the M4-03 posture).

// A `GlobalAlloc` impl is inherently `unsafe`; same opt-out the M4-03
// binary uses. SAFETY comments on each block.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use vokra_core::{DialogRequest, StreamEvent};
use vokra_models::csm::{CsmEngine, CsmStreamConfig, EchoPath};

struct CountingAlloc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// Why this counts per-thread (2026-08-14)
// ---------------------------------------------------------------------------
// A `#[global_allocator]` is process-wide, but the property under test is
// about *this* loop: "the CSM frame loop must not allocate". Counting every
// thread conflated the two, and the test failed intermittently under
// full-workspace parallel load (~1 run in 6) with a varying count (3, 9, 10)
// while the loop itself ran to completion every time.
//
// Instrumenting the failure settled it. The recorded allocations were:
//
//     [0] alloc  9 bytes, frame 7, on another thread
//     [1] alloc  9 bytes, frame 7, on another thread
//     [2] alloc  9 bytes, frame 7, on another thread
//
// Not the frame loop: another thread in the process, allocating 9 bytes —
// the length of `"vokra-cpu"`, the name `vokra-backend-cpu`'s worker pool
// gives its threads (`pool.rs`). Those workers are spawned lazily when GEMM
// routing first asks for the participant count, and they allocate while
// booting; under load that boot lands inside the measured window instead of
// before it. They do no work for this fixture — its GEMMs are orders of
// magnitude below the pool's `PAR_MIN_MACS` threshold (one frame is 8
// samples), so the loop's arithmetic runs inline on this thread.
//
// So the assertion now counts allocations made *by the measuring thread*,
// which is what it always meant. Other-thread allocations are still recorded
// and printed on failure — they are excluded from the verdict, never hidden.
//
// Everything on the recording path is an atomic store into a fixed array, and
// `IS_MEASURING` is a `const`-initialised thread-local (a plain per-thread
// static, no lazy heap init), so recording cannot re-enter the allocator or
// perturb what it measures.
const REC_SLOTS: usize = 24;
#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicUsize = AtomicUsize::new(0);

/// Recording is open (the measured window).
static RECORDING: AtomicBool = AtomicBool::new(false);
/// Byte size of each recorded allocation.
static REC_SIZE: [AtomicUsize; REC_SLOTS] = [ZERO; REC_SLOTS];
/// `0` = `alloc`, `1` = `realloc`.
static REC_KIND: [AtomicUsize; REC_SLOTS] = [ZERO; REC_SLOTS];
/// `1` when the allocating thread is the one running the measured loop.
static REC_ON_TEST_THREAD: [AtomicUsize; REC_SLOTS] = [ZERO; REC_SLOTS];
/// Loop iteration the window was on when the allocation happened.
static REC_FRAME: [AtomicUsize; REC_SLOTS] = [ZERO; REC_SLOTS];
/// Next free slot (also the total count, which may exceed `REC_SLOTS`).
static REC_IDX: AtomicUsize = AtomicUsize::new(0);
/// Allocations made by the measuring thread inside the window — the verdict.
static TEST_THREAD_ALLOCS: AtomicUsize = AtomicUsize::new(0);
/// The loop iteration currently in flight.
static CUR_FRAME: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Marks the thread running the measured loop. `const`-initialised, so
    /// reading it from inside the allocator never allocates.
    static IS_MEASURING: Cell<bool> = const { Cell::new(false) };
}

/// Records one allocation and, when it came from the measuring thread, counts
/// it toward the verdict. Called from the allocator, so it must stay
/// allocation-free and non-panicking.
fn record(size: usize, kind: usize) {
    if !RECORDING.load(Ordering::Relaxed) {
        return;
    }
    // `try_with` so a thread tearing down its TLS reports `false` rather than
    // panicking inside the allocator.
    let on_test_thread = IS_MEASURING.try_with(Cell::get).unwrap_or(false);
    if on_test_thread {
        TEST_THREAD_ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
    let i = REC_IDX.fetch_add(1, Ordering::Relaxed);
    if i < REC_SLOTS {
        REC_SIZE[i].store(size, Ordering::Relaxed);
        REC_KIND[i].store(kind, Ordering::Relaxed);
        REC_ON_TEST_THREAD[i].store(usize::from(on_test_thread), Ordering::Relaxed);
        REC_FRAME[i].store(CUR_FRAME.load(Ordering::Relaxed), Ordering::Relaxed);
    }
}

/// Renders the recorded allocations for the failure message.
fn forensics() -> String {
    let total = REC_IDX.load(Ordering::SeqCst);
    let shown = total.min(REC_SLOTS);
    let mut s = String::new();
    for i in 0..shown {
        let kind = if REC_KIND[i].load(Ordering::SeqCst) == 1 {
            "realloc"
        } else {
            "alloc  "
        };
        let where_ = if REC_ON_TEST_THREAD[i].load(Ordering::SeqCst) == 1 {
            "THIS thread (real hot-path allocation)"
        } else {
            "another thread (measurement artefact)"
        };
        s.push_str(&format!(
            "\n  [{i}] {kind} {:>8} bytes, frame {}, on {where_}",
            REC_SIZE[i].load(Ordering::SeqCst),
            REC_FRAME[i].load(Ordering::SeqCst),
        ));
    }
    if total > shown {
        s.push_str(&format!("\n  ... {} more not recorded", total - shown));
    }
    s
}

// SAFETY: pure delegation to `System`; the counter is a relaxed atomic with
// no additional invariants.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        record(layout.size(), 0);
        // SAFETY: forwarding the exact layout to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding a pointer previously returned by `System.alloc`
        // with its original layout.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        record(new_size, 1);
        // SAFETY: forwarding the original pointer/layout and the new size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[test]
fn csm_frame_loop_allocates_zero_after_open() {
    let engine = CsmEngine::synthesized_fixture(77)
        .expect("fixture engine")
        .with_echo_path(EchoPath::BypassRecordedInput);
    // Deterministic → greedy sampler (the alloc-free M1 path).
    let request = DialogRequest::new("alloc-free frame loop").deterministic();
    let mut stream = engine
        .open_stream(&request, Some(CsmStreamConfig { max_frames: 24 }))
        .expect("stream opens");
    // Vec<StreamEvent> would grow (allocate); pre-reserve outside the
    // measured region so `emit` is a plain write.
    let mut sink: Vec<StreamEvent> = Vec::with_capacity(64);
    let mut collected = 0usize;

    // Warm-up: one frame (nothing in the loop allocates lazily — the
    // warm-up keeps the measurement independent of that claim, mirroring
    // the M4-03 binary).
    let first = stream.next_frame(&mut sink).expect("first frame");
    assert!(first.is_some(), "fixture must emit at least one frame");

    // Open the forensic window (see the counters above): mark this thread,
    // then start recording. Both are plain stores — no allocation.
    IS_MEASURING.with(|c| c.set(true));
    RECORDING.store(true, Ordering::SeqCst);

    let before_all = ALLOCS.load(Ordering::SeqCst);
    let mut frames = 0usize;
    for i in 0..16 {
        CUR_FRAME.store(i, Ordering::Relaxed);
        match stream.next_frame(&mut sink).expect("frame") {
            Some(pcm) => {
                assert!(!pcm.is_empty());
                collected += pcm.len();
                frames += 1;
            }
            None => break,
        }
    }
    let after_all = ALLOCS.load(Ordering::SeqCst);

    RECORDING.store(false, Ordering::SeqCst);
    IS_MEASURING.with(|c| c.set(false));

    let on_this_thread = TEST_THREAD_ALLOCS.load(Ordering::SeqCst);
    let elsewhere = (after_all - before_all) - on_this_thread;

    // `forensics()` allocates (it builds the message) — call it only after the
    // window is closed, and only to describe a failure.
    assert_eq!(
        on_this_thread,
        0,
        "the CSM frame loop must not allocate after open_stream \
         (FR-EX-05; {frames} frames / {collected} samples generated in the \
         measured region; {elsewhere} further allocation(s) came from other \
         threads and are excluded — see the header).\
         \nRecorded allocations:{}",
        forensics()
    );
}
