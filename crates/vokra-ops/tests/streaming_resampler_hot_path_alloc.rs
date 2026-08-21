//! Counting-allocator proof for `StreamingResampler::process_into` (#50).

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use vokra_ops::StreamingResampler;

struct CountingAlloc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: the wrapper delegates to `System` without changing pointer/layout
// contracts and only increments a diagnostic counter.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarding the exact layout to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding a pointer with its original layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
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
    let before = ALLOCS.load(Ordering::SeqCst);
    for _ in 0..1000 {
        stream
            .process_into(&input, &mut out)
            .expect("steady-state chunk");
    }
    let after = ALLOCS.load(Ordering::SeqCst);

    assert_eq!(
        after - before,
        0,
        "StreamingResampler::process_into allocated in steady state",
    );
}
