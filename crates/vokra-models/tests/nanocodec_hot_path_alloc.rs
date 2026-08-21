//! #46: counting-allocator proof for NanoCodec causal HiFi-GAN streaming.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use vokra_models::nanocodec::{
    CausalHifiGan, CausalHifiGanConfig, CausalHifiGanConv1dWeights,
    CausalHifiGanConvTranspose1dWeights, CausalHifiGanHalfSnakeWeights,
    CausalHifiGanResidualBlockWeights, CausalHifiGanStageWeights, CausalHifiGanWeights,
};

struct CountingAlloc;

static RECORDING: AtomicBool = AtomicBool::new(false);
static MEASURED_ALLOCS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static IS_MEASURING: Cell<bool> = const { Cell::new(false) };
}

fn record_allocation() {
    if RECORDING.load(Ordering::Relaxed) && IS_MEASURING.try_with(Cell::get).unwrap_or(false) {
        MEASURED_ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every operation delegates the unchanged pointer/layout contract to
// `System`; the counter uses only atomics and const-initialized TLS.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: exact delegation to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` came from the delegated allocation.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        // SAFETY: exact delegation of the original pointer/layout/new size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn conv(out_channels: usize, in_channels: usize, kernel: usize) -> CausalHifiGanConv1dWeights {
    let len = out_channels * in_channels * kernel;
    CausalHifiGanConv1dWeights {
        weight: (0..len).map(|i| (i as f32 + 1.0) * 0.003).collect(),
        bias: vec![0.001; out_channels],
    }
}

fn half_snake(channels: usize) -> CausalHifiGanHalfSnakeWeights {
    CausalHifiGanHalfSnakeWeights {
        alpha: vec![0.8; channels / 2],
        alpha_inv: vec![1.25; channels / 2],
    }
}

fn decoder() -> CausalHifiGan {
    let config = CausalHifiGanConfig {
        input_dim: 1,
        base_channels: 2,
        frame_hop: 2,
        upsample_rates: vec![2],
        input_kernel_size: 3,
        output_kernel_size: 3,
        resblock_kernel_sizes: vec![3],
        resblock_dilations: vec![1],
    };
    let residual = CausalHifiGanResidualBlockWeights {
        input_activation: half_snake(1),
        input_conv: conv(1, 1, 3),
        skip_activation: half_snake(1),
        skip_conv: conv(1, 1, 3),
        dilation: 1,
    };
    let weights = CausalHifiGanWeights {
        pre_conv: conv(2, 1, 3),
        stages: vec![CausalHifiGanStageWeights {
            activation: half_snake(2),
            // Dense expansion of groups=out_channels=1.
            upsample: CausalHifiGanConvTranspose1dWeights {
                weight: (0..8).map(|i| (i as f32 + 1.0) * 0.002).collect(),
                bias: vec![0.001],
            },
            residual_branches: vec![vec![residual]],
        }],
        post_activation: half_snake(1),
        post_conv: conv(1, 1, 3),
    };
    CausalHifiGan::new(config, weights).expect("tiny NanoCodec decoder")
}

#[test]
fn steady_state_decode_into_allocates_zero() {
    let decoder = decoder();
    let mut state = decoder.state(1).expect("state");
    let features = [0.25f32];
    let mut pcm = [0.0f32; 2];

    // Warm backend/pool initialization before opening the measurement window.
    decoder
        .decode_into(&mut state, &features, &mut pcm)
        .expect("warm-up frame");

    MEASURED_ALLOCS.store(0, Ordering::SeqCst);
    IS_MEASURING.with(|flag| flag.set(true));
    RECORDING.store(true, Ordering::SeqCst);
    for _ in 0..32 {
        let written = decoder
            .decode_into(&mut state, &features, &mut pcm)
            .expect("steady-state frame");
        assert_eq!(written, 2);
    }
    RECORDING.store(false, Ordering::SeqCst);
    IS_MEASURING.with(|flag| flag.set(false));

    assert_eq!(
        MEASURED_ALLOCS.load(Ordering::SeqCst),
        0,
        "CausalHifiGan::decode_into must not allocate after state creation"
    );
}
