//! Counting-allocator proof for `StreamingResampler::process_into` (#50).

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use vokra_ops::StreamingResampler;

struct CountingAlloc;

static TEST_THREAD_ALLOCS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    // A const-initialised TLS flag is safe to query from the allocator: it
    // needs no lazy heap allocation. The workspace test runner may allocate
    // on background threads while this test is running, especially on macOS;
    // those allocations are unrelated to the resampler hot path.
    static MEASURE_THIS_THREAD: Cell<bool> = const { Cell::new(false) };
}

fn record_test_thread_allocation() {
    if MEASURE_THIS_THREAD.try_with(Cell::get).unwrap_or(false) {
        TEST_THREAD_ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: the wrapper delegates to `System` without changing pointer/layout
// contracts and only increments a diagnostic counter.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_test_thread_allocation();
        // SAFETY: forwarding the exact layout to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding a pointer with its original layout.
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
fn one_thousand_steady_state_chunks_allocate_zero() {
    let mut stream = StreamingResampler::new(22_050, 48_000, 5).unwrap();
    let input = [0.125f32; 64];
    let mut out = [0.0f32; 256];

    stream.process_into(&input, &mut out).expect("warm-up");

    // Reproduce the macOS workspace-runner condition that motivated this
    // regression test: unrelated work allocates on another thread while the
    // hot-path window is open. A process-wide counter would report these as a
    // false positive even though `process_into` itself remains allocation-free.
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
        stream
            .process_into(&input, &mut out)
            .expect("steady-state chunk");
    }
    while !noise_done.load(Ordering::Acquire) {
        std::hint::spin_loop();
    }
    MEASURE_THIS_THREAD.with(|enabled| enabled.set(false));
    noise.join().expect("background allocator-noise thread");
    let allocations = TEST_THREAD_ALLOCS.load(Ordering::SeqCst);

    assert_eq!(
        allocations, 0,
        "StreamingResampler::process_into allocated in steady state",
    );
}
