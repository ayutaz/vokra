//! #48: allocation-count proof for the generic codec handle's warmed
//! `push_codes` + `pull_pcm` success path.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use vokra_core::{CodecDecoderEngine, CodecDecoderHandle};
use vokra_models::codec::{MimiCodecGguf, MimiStreamingCodec, NanoCodecStreamingCodec};
use vokra_models::mimi::{MimiNeuralConfig, MimiNeuralDecoder};
use vokra_models::nanocodec::{
    CausalHifiGan, CausalHifiGanConfig, CausalHifiGanConv1dWeights,
    CausalHifiGanConvTranspose1dWeights, CausalHifiGanHalfSnakeWeights,
    CausalHifiGanResidualBlockWeights, CausalHifiGanStageWeights, CausalHifiGanWeights,
};
use vokra_ops::{CodebookTable, MimiRvqAttrs};

struct CountingAlloc;
static MEASURED_ALLOCS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

fn count_if_measuring() {
    if MEASURING.try_with(Cell::get).unwrap_or(false) {
        MEASURED_ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every operation delegates unchanged to the system allocator; the
// extra accounting is an allocation-free atomic increment on the measuring
// thread only.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_if_measuring();
        // SAFETY: exact delegation of the requested layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: exact delegation of the pointer and its original layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_if_measuring();
        // SAFETY: exact delegation of the pointer/layout/new size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn fixture() -> Box<dyn CodecDecoderHandle + Send> {
    let cfg = MimiNeuralConfig::tiny_for_tests();
    let tables = (0..cfg.quantizer.n_q)
        .map(|cb| {
            let data = (0..cfg.quantizer.bins * cfg.seanet.dimension)
                .map(|i| ((cb * 29 + i) as f32 - 20.0) * 0.001)
                .collect();
            CodebookTable::new(cfg.quantizer.bins, cfg.seanet.dimension, data).unwrap()
        })
        .collect::<Vec<_>>();
    let codec = MimiCodecGguf {
        attrs: MimiRvqAttrs {
            n_codebooks: cfg.quantizer.n_q,
            codebook_size: cfg.quantizer.bins,
            d_model: cfg.seanet.dimension,
        },
        tables,
    };
    let neural = MimiNeuralDecoder::synthesized(&cfg, 48, false).unwrap();
    MimiStreamingCodec::new(codec, neural, cfg.sample_rate)
        .unwrap()
        .open_decoder()
        .unwrap()
}

fn nanocodec_conv(
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
) -> CausalHifiGanConv1dWeights {
    CausalHifiGanConv1dWeights {
        weight: (0..out_channels * in_channels * kernel)
            .map(|i| (i as f32 + 1.0) * 0.003)
            .collect(),
        bias: vec![0.001; out_channels],
    }
}

fn nanocodec_half_snake(channels: usize) -> CausalHifiGanHalfSnakeWeights {
    CausalHifiGanHalfSnakeWeights {
        alpha: vec![0.8; channels / 2],
        alpha_inv: vec![1.25; channels / 2],
    }
}

fn nanocodec_fixture() -> Box<dyn CodecDecoderHandle + Send> {
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
        input_activation: nanocodec_half_snake(1),
        input_conv: nanocodec_conv(1, 1, 3),
        skip_activation: nanocodec_half_snake(1),
        skip_conv: nanocodec_conv(1, 1, 3),
        dilation: 1,
    };
    let decoder = CausalHifiGan::new(
        config,
        CausalHifiGanWeights {
            pre_conv: nanocodec_conv(2, 1, 3),
            stages: vec![CausalHifiGanStageWeights {
                activation: nanocodec_half_snake(2),
                upsample: CausalHifiGanConvTranspose1dWeights {
                    weight: (0..8).map(|i| (i as f32 + 1.0) * 0.002).collect(),
                    bias: vec![0.001],
                },
                residual_branches: vec![vec![residual]],
            }],
            post_activation: nanocodec_half_snake(1),
            post_conv: nanocodec_conv(1, 1, 3),
        },
    )
    .unwrap();
    NanoCodecStreamingCodec::new(1, vec![4], decoder, 22_050)
        .unwrap()
        .open_decoder()
        .unwrap()
}

#[test]
fn warmed_push_and_pull_allocate_nothing() {
    for mut decoder in [fixture(), nanocodec_fixture()] {
        let codes = vec![1u32; decoder.n_codebooks()];
        let mut pcm = vec![0.0f32; decoder.frame_hop()];

        // Warm backend dispatch and every lazy one-time path before measuring.
        decoder.push_codes(&codes).unwrap();
        decoder.pull_pcm(&mut pcm).unwrap();

        MEASURED_ALLOCS.store(0, Ordering::SeqCst);
        MEASURING.with(|flag| flag.set(true));
        for _ in 0..16 {
            assert_eq!(decoder.push_codes(&codes).unwrap(), 1);
            assert_eq!(decoder.pull_pcm(&mut pcm).unwrap(), pcm.len());
        }
        MEASURING.with(|flag| flag.set(false));

        assert_eq!(
            MEASURED_ALLOCS.load(Ordering::SeqCst),
            0,
            "warmed generic codec push+pull must allocate nothing"
        );
    }
}
