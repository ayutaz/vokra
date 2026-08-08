//! `SbV2Decoder` (HiFi-GAN vocoder wrapper, Task 22) tests.
//!
//! Builds a small-but-valid HiFi-GAN weight/attrs bundle shaped like the
//! SBV2 JP-Extra base config's upsample ladder (`upsample_rates = [8, 8,
//! 2, 2]`, `upsample_kernel_sizes = [16, 16, 4, 4]` — the `kernel = 2 *
//! stride` convention jik876/hifi-gan's V1/V2/V3 presets all follow,
//! which makes the upsample stack's output length exactly `mel_seq_len *
//! 256`, not merely approximately — see `sbv2::decoder`'s module doc).
//! Every other shape parameter (channel counts, MRF branch count/kernel/
//! dilation) is a minimal, deliberately-small placeholder: `vokra_ops`
//! has no `HiFiGanGenerator::synthetic_for_test()` constructor (the type
//! doesn't even exist — `hifigan_generator` is a free function over three
//! separately-constructed bundles, see `sbv2::decoder`'s module doc), so
//! this file builds [`HifiGanWeights`]/[`HifiGanAttrs`] directly,
//! mirroring `vokra-ops/tests/parity_hifigan.rs`'s own
//! `parity_attrs`/`parity_weights` helpers (same crate family, same
//! construction shape, same smooth-sinusoidal deterministic weight
//! formula). Real checkpoint-driven weights land with the Task 24-27
//! converter.

use vokra_models::sbv2::SbV2Decoder;
use vokra_ops::attrs::{HifiGanAttrs, ResBlockType};
use vokra_ops::hifigan::{
    HifiGanConfig, HifiGanWeights, MrfBranchWeights, ResBlockLayer, UpsampleStageWeights,
};

/// SBV2 JP-Extra base config's total upsample factor (`8 * 8 * 2 * 2`).
const JP_EXTRA_TOTAL_UPSAMPLE: usize = 256;

/// Small-but-JP-Extra-shaped attrs: the real `upsample_rates` /
/// `upsample_kernel_sizes` ladder from the brief, minimal channel / MRF
/// counts everywhere else (this test only needs to exercise the
/// `SbV2Decoder` wrapper's plumbing, not reproduce a real checkpoint's
/// full width).
fn jp_extra_attrs() -> HifiGanAttrs {
    HifiGanAttrs {
        n_mels: 3,
        initial_channel: 6,
        upsample_rates: vec![8, 8, 2, 2],
        upsample_kernel_sizes: vec![16, 16, 4, 4], // kernel = 2*stride: exact *256 length
        resblock_kernel_sizes: vec![3],
        resblock_dilation_sizes: vec![vec![1]],
        sample_rate: 44_100,
        leaky_relu_slope: 0.1,
        // Fixture builds V2-shape single-conv layers (no c2). Real
        // SBV2 v2 ckpt uses V1 (ResBlock1) — see the from_gguf loader
        // in sbv2/mod.rs. This test only exercises SbV2Decoder wrapper
        // plumbing, so V2 keeps the minimal-fixture invariants.
        res_block_type: ResBlockType::V2,
    }
}

/// Deterministic, bounded, nonzero weight builder — mirrors
/// `vokra-ops/tests/parity_hifigan.rs`'s `parity_weights` shape (smooth
/// sinusoidal cells so the terminal `tanh` stays well inside `(-1, 1)`).
fn jp_extra_weights(attrs: &HifiGanAttrs) -> HifiGanWeights {
    let conv_pre_kernel = 7;
    let conv_post_kernel = 7;
    let mut w = HifiGanWeights {
        conv_pre_weight: Vec::new(),
        conv_pre_bias: Vec::new(),
        conv_pre_kernel,
        upsample_weights: Vec::new(),
        mrf_stage_weights: Vec::new(),
        conv_post_weight: Vec::new(),
        conv_post_bias: Vec::new(),
        cond: None,
        conv_post_kernel,
    };
    // Initial conv1d: [initial_channel, n_mels, k].
    for oc in 0..attrs.initial_channel {
        for ic in 0..attrs.n_mels {
            for k in 0..conv_pre_kernel {
                w.conv_pre_weight
                    .push(((oc + ic + k) as f32 * 0.017).sin() * 0.05);
            }
        }
    }
    w.conv_pre_bias = (0..attrs.initial_channel)
        .map(|i| (i as f32 * 0.05).cos() * 0.01)
        .collect();

    // Upsample stages: halve the channel count each stage (floor at 3).
    let mut in_ch = attrs.initial_channel;
    for stage in 0..attrs.n_upsample_stages() {
        let out_ch = (in_ch / 2).max(3);
        let kernel = attrs.upsample_kernel_sizes[stage];
        let stride = attrs.upsample_rates[stage];
        let mut weight = Vec::new();
        for ic in 0..in_ch {
            for oc in 0..out_ch {
                for k in 0..kernel {
                    weight.push(((ic + oc + k + stage) as f32 * 0.023).sin() * 0.05);
                }
            }
        }
        let bias: Vec<f32> = (0..out_ch)
            .map(|i| ((i + stage) as f32 * 0.07).cos() * 0.01)
            .collect();
        w.upsample_weights.push(UpsampleStageWeights {
            weight,
            bias,
            in_ch,
            out_ch,
            kernel,
            stride,
        });

        // MRF branches (single branch, single dilation — see this file's
        // module doc for why the fixture stays minimal here).
        let mut branches = Vec::new();
        for b in 0..attrs.n_mrf_branches() {
            let layers = attrs.resblock_dilation_sizes[b]
                .iter()
                .map(|dilation| {
                    let kernel = attrs.resblock_kernel_sizes[b];
                    let mut weight = Vec::new();
                    for oc in 0..out_ch {
                        for ic in 0..out_ch {
                            for k in 0..kernel {
                                weight.push(((oc + ic + k + dilation) as f32 * 0.031).sin() * 0.05);
                            }
                        }
                    }
                    let bias: Vec<f32> = (0..out_ch)
                        .map(|i| ((i + *dilation + b) as f32 * 0.11).cos() * 0.01)
                        .collect();
                    ResBlockLayer {
                        weight,
                        bias,
                        weight_c2: None,
                        bias_c2: None,
                        dilation: *dilation,
                        kernel,
                        channels: out_ch,
                    }
                })
                .collect();
            branches.push(MrfBranchWeights { layers });
        }
        w.mrf_stage_weights.push(branches);
        in_ch = out_ch;
    }
    // Final conv1d: [1, in_ch, k].
    for ic in 0..in_ch {
        for k in 0..conv_post_kernel {
            w.conv_post_weight
                .push(((ic + k) as f32 * 0.019).sin() * 0.05);
        }
    }
    w.conv_post_bias = vec![0.0];
    w
}

/// Builds an `SbV2Decoder` over the JP-Extra-shaped fixture, plus the
/// `HifiGanAttrs` used to build it (so callers can size their
/// `mel_hidden` input from `attrs.n_mels`).
fn make_decoder() -> (SbV2Decoder, HifiGanAttrs) {
    let attrs = jp_extra_attrs();
    let weights = jp_extra_weights(&attrs);
    let sample_rate = attrs.sample_rate;
    let decoder = SbV2Decoder::new(weights, attrs.clone(), HifiGanConfig::fp32(), sample_rate);
    (decoder, attrs)
}

/// Output length matches `mel_seq_len * upsample_ratio` — exactly 256 for
/// the JP-Extra `[8, 8, 2, 2]` ladder (not merely approximately: see
/// `sbv2::decoder`'s module doc for why `upsample_kernel_sizes = [16, 16,
/// 4, 4]` makes every stage's output length land exactly on `in_len *
/// stride`).
#[test]
fn generate_output_length_matches_upsample_ratio() {
    let (decoder, attrs) = make_decoder();
    let mel_seq_len = 3;
    let mel_hidden: Vec<f32> = (0..mel_seq_len * attrs.n_mels)
        .map(|i| (i as f32 * 0.13).sin())
        .collect();

    let out = decoder.generate(&mel_hidden, mel_seq_len);

    assert_eq!(
        out.len(),
        mel_seq_len * JP_EXTRA_TOTAL_UPSAMPLE,
        "JP-Extra ladder [8,8,2,2] must upsample by exactly 256x"
    );
}

/// Same input → same output (HiFi-GAN is pure forward, no internal RNG —
/// `hifigan_generator` itself pins this at the op level in
/// `vokra-ops/tests/parity_hifigan.rs::fp32_forward_is_bit_identical_across_runs`;
/// this test pins the identical property through the `SbV2Decoder`
/// wrapper).
#[test]
fn generate_is_deterministic() {
    let (decoder, attrs) = make_decoder();
    let mel_seq_len = 4;
    let mel_hidden: Vec<f32> = (0..mel_seq_len * attrs.n_mels)
        .map(|i| (i as f32 * 0.09).cos() * 0.5)
        .collect();

    let out_1 = decoder.generate(&mel_hidden, mel_seq_len);
    let out_2 = decoder.generate(&mel_hidden, mel_seq_len);

    assert_eq!(out_1, out_2, "same input must produce same output");
}
