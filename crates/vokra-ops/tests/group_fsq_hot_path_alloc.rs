//! Allocation-count proof for the grouped-FSQ caller-owned-buffer API (#45).
//!
//! A dedicated test binary owns the global counting allocator. After one
//! warm-up decode, 1000 consecutive NanoCodec-sized single-frame decodes must
//! allocate nothing. `scripts/check-hot-path-allocs.sh` independently scans
//! the marked implementation region for allocating constructs.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use vokra_ops::group_fsq_decode_into;

struct CountingAlloc;

static TEST_THREAD_ALLOCS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    // A const-initialised TLS flag is safe to query from the allocator: it
    // needs no lazy heap allocation. Background workspace-runner allocations
    // are unrelated to the grouped-FSQ hot path.
    static MEASURE_THIS_THREAD: Cell<bool> = const { Cell::new(false) };
}

fn record_test_thread_allocation() {
    if MEASURE_THIS_THREAD.try_with(Cell::get).unwrap_or(false) {
        TEST_THREAD_ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: this wrapper delegates every allocation to `System` unchanged and
// only increments a relaxed diagnostic counter.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_test_thread_allocation();
        // SAFETY: forwarding the exact layout to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding a pointer and its original layout to `System`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_test_thread_allocation();
        // SAFETY: forwarding the original allocation and requested size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[test]
fn one_thousand_single_frame_decodes_allocate_zero_after_warmup() {
    // The largest currently released group count in #45's acceptance matrix.
    const N_GROUPS: usize = 13;
    const N_DIMS: usize = 4;
    // Official 1.78 kbps / 12.5 fps model card: G=13, FSQ levels [8,7,6,6].
    let levels = [8u32, 7, 6, 6];
    let codes = [0u32; N_GROUPS];
    let mut out = [0.0f32; N_GROUPS * N_DIMS];

    group_fsq_decode_into(&codes, 1, &levels, &mut out).expect("warm-up decode");

    // Reproduce the hosted-runner condition that motivated this regression
    // test: unrelated work allocates on another thread while the hot-path
    // window is open. A process-wide counter would report these as a false
    // positive even though group_fsq_decode_into remains allocation-free.
    let start_noise = Arc::new(AtomicBool::new(false));
    let noise_done = Arc::new(AtomicBool::new(false));
    let worker_start = Arc::clone(&start_noise);
    let worker_done = Arc::clone(&noise_done);
    let noise = thread::spawn(move || {
        while !worker_start.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        let mut allocations = Vec::with_capacity(1_024);
        for value in 0..1_024 {
            allocations.push(Box::new(value));
        }
        black_box(allocations);
        worker_done.store(true, Ordering::Release);
    });

    TEST_THREAD_ALLOCS.store(0, Ordering::SeqCst);
    MEASURE_THIS_THREAD.with(|enabled| enabled.set(true));
    start_noise.store(true, Ordering::Release);
    for _ in 0..1000 {
        group_fsq_decode_into(&codes, 1, &levels, &mut out).expect("steady-state decode");
    }
    while !noise_done.load(Ordering::Acquire) {
        std::hint::spin_loop();
    }
    MEASURE_THIS_THREAD.with(|enabled| enabled.set(false));
    noise.join().expect("background allocator-noise thread");
    let allocations = TEST_THREAD_ALLOCS.load(Ordering::SeqCst);

    assert_eq!(
        allocations, 0,
        "group_fsq_decode_into must allocate zero times across 1000 frames",
    );
}
