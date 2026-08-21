//! Allocation-count proof for the grouped-FSQ caller-owned-buffer API (#45).
//!
//! A dedicated test binary owns the global counting allocator. After one
//! warm-up decode, 1000 consecutive NanoCodec-sized single-frame decodes must
//! allocate nothing. `scripts/check-hot-path-allocs.sh` independently scans
//! the marked implementation region for allocating constructs.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use vokra_ops::group_fsq_decode_into;

struct CountingAlloc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: this wrapper delegates every allocation to `System` unchanged and
// only increments a relaxed diagnostic counter.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarding the exact layout to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding a pointer and its original layout to `System`.
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
fn one_thousand_single_frame_decodes_allocate_zero_after_warmup() {
    // The largest currently released group count in #45's acceptance matrix.
    const N_GROUPS: usize = 13;
    const N_DIMS: usize = 4;
    // Official 1.78 kbps / 12.5 fps model card: G=13, FSQ levels [8,7,6,6].
    let levels = [8u32, 7, 6, 6];
    let codes = [0u32; N_GROUPS];
    let mut out = [0.0f32; N_GROUPS * N_DIMS];

    group_fsq_decode_into(&codes, 1, &levels, &mut out).expect("warm-up decode");

    let before = ALLOCS.load(Ordering::SeqCst);
    for _ in 0..1000 {
        group_fsq_decode_into(&codes, 1, &levels, &mut out).expect("steady-state decode");
    }
    let after = ALLOCS.load(Ordering::SeqCst);

    assert_eq!(
        after - before,
        0,
        "group_fsq_decode_into must allocate zero times across 1000 frames",
    );
}
