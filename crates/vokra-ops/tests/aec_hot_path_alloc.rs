//! M4-03 T11 — allocation-count proof that the AEC hot path is malloc-free
//! (FR-EX-05 / NFR-RL-08: `AecRefWriter::push` → `Aec::process` must be
//! callable from an audio callback / real-time thread).
//!
//! A counting `#[global_allocator]` (its own test binary — a global
//! allocator is per-binary, which is why this is not a unit test) wraps the
//! system allocator; after a one-frame warm-up the steady-state loop of
//! push + process must perform **zero** allocations. This is the
//! allocation-count complement of the two structural guards:
//! `scripts/check-hot-path-allocs.sh` (ZERO-ALLOC marker scan over
//! `aec.rs`) and the `buffer_capacities_are_stable_across_frames` unit
//! oracle.

// A `GlobalAlloc` impl is inherently `unsafe`; this test target opts out of
// the workspace `unsafe_code = "deny"` exactly like the vokra-ops lib does
// for its SIMD kernels (crate-level policy: every block carries `// SAFETY:`).
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use vokra_core::stream::aec_ref_queue;
use vokra_ops::{Aec, AecAttrs, AecStatus};

/// System allocator wrapper that counts every allocation / reallocation.
struct CountingAlloc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: pure delegation to `System`; the counter is a relaxed atomic with
// no additional invariants.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
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
        // SAFETY: forwarding the original pointer/layout and the new size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[test]
fn steady_state_push_and_process_allocate_zero() {
    let attrs = AecAttrs {
        sample_rate: 16_000,
        frame_size: 256,
        filter_length: 2048,
    };
    let n = attrs.frame_size;
    let mut aec = Aec::new(&attrs).expect("aec builds");
    let (mut tx, mut rx) = aec_ref_queue(8 * attrs.filter_length, attrs.sample_rate).unwrap();

    // Deterministic non-trivial frames (allocated up front, outside the
    // measured region).
    let far: Vec<f32> = (0..n)
        .map(|i| ((i * 37 + 5) % 199) as f32 / 400.0 - 0.25)
        .collect();
    let mic: Vec<f32> = (0..n)
        .map(|i| ((i * 53 + 11) % 211) as f32 / 500.0 - 0.2)
        .collect();
    let mut out = vec![0.0f32; n];

    // Warm-up: one full cycle (nothing in the AEC allocates lazily, but the
    // warm-up keeps the measurement independent of that claim).
    tx.push(&far, 0).unwrap();
    aec.process(&mic, 0, &mut rx, &mut out).unwrap();

    let before = ALLOCS.load(Ordering::SeqCst);
    for f in 1..200u64 {
        let pos = f * n as u64;
        let accepted = tx.push(&far, pos).unwrap();
        assert_eq!(accepted, n);
        let status = aec.process(&mic, pos, &mut rx, &mut out).unwrap();
        assert_eq!(status, AecStatus::Cancelled);
    }
    let after = ALLOCS.load(Ordering::SeqCst);

    assert_eq!(
        after - before,
        0,
        "AEC steady state must not allocate (FR-EX-05): {} allocations in 199 frames",
        after - before
    );
}
