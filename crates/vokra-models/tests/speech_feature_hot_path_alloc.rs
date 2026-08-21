//! Counting-allocator proof for #49 streaming speech-feature push/pull.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use vokra_core::engines::SpeechFeatureEngine;
use vokra_models::moshi::MoshiEngine;

struct CountingAlloc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every allocation operation is forwarded unchanged to `System`; the
// wrapper only increments a diagnostic counter.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: exact layout forwarded to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: original pointer/layout forwarded to the system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: original allocation and requested size forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[test]
fn one_thousand_warmed_push_pull_pairs_allocate_zero() {
    let engine = Arc::new(MoshiEngine::synthesized_fixture(0xFEA7_0049).unwrap());
    let mut stream = engine.open_feature_stream().unwrap();
    let token_hop = stream.feature_frame_hop() * 2;
    let input = vec![0.125f32; token_hop];
    let mut output = vec![0.0f32; stream.feature_dim() * 2];

    // Warm both the retained t=2 batch scratch and a complete rolling
    // attention-window cycle. Windows CI observed one process-lifetime
    // allocation before the window saturated; measuring after every window
    // width has been exercised keeps that setup outside steady state while
    // still catching any allocation that recurs afterward.
    for _ in 0..256 {
        stream.push_pcm(&input).unwrap();
        assert_eq!(stream.pull_into(&mut output).unwrap().0, 2);
    }

    let before = ALLOCS.load(Ordering::SeqCst);
    for _ in 0..1000 {
        stream.push_pcm(&input).unwrap();
        assert_eq!(stream.pull_into(&mut output).unwrap().0, 2);
    }
    let after = ALLOCS.load(Ordering::SeqCst);
    assert_eq!(
        after - before,
        0,
        "streaming speech-feature push/pull allocated after warmup",
    );
}
