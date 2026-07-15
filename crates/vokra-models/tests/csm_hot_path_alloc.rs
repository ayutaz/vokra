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
use std::sync::atomic::{AtomicUsize, Ordering};

use vokra_core::{DialogRequest, StreamEvent};
use vokra_models::csm::{CsmEngine, CsmStreamConfig, EchoPath};

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

    let before = ALLOCS.load(Ordering::SeqCst);
    for _ in 0..16 {
        match stream.next_frame(&mut sink).expect("frame") {
            Some(pcm) => {
                assert!(!pcm.is_empty());
                collected += pcm.len();
            }
            None => break,
        }
    }
    let after = ALLOCS.load(Ordering::SeqCst);
    assert_eq!(
        after - before,
        0,
        "the CSM frame loop must not allocate after open_stream \
         (FR-EX-05; {collected} samples generated in the measured region)"
    );
}
