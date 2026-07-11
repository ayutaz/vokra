//! Numerical-parity harness for the HiFi-GAN generator op (M3-07, FR-OP-10).
//!
//! # Reference oracle
//!
//! An external PyTorch reference (jik876/hifi-gan + a real V1/V2/V3 generator
//! checkpoint) is not fetched by this WP — the M3-07 ticket delegates real
//! checkpoint parity to the consumer WP (M3-09 CosyVoice2), where a real
//! HiFi-GAN sub-vocoder ships and its calibration dataset exists. Until then
//! the M3-07 parity harness runs **internal-oracle** checks that pin exactly
//! the invariants the ticket calls out:
//!
//! 1. **FP32 forward is bit-identical run-to-run** (T10). The generator is a
//!    pure function of `(mel, weights, attrs)`; running it twice with the same
//!    inputs must yield the same output (no hidden RNG, no rayon-style non-
//!    determinism).
//! 2. **fp16 forward matches fp32 within atol = 0.01** (T11). The mixed-precision
//!    path (FP32 accumulator + f16-representable terminal cast) rounds only the
//!    output; NFR-QL-01's atol = 0.01 gate must hold.
//! 3. **INT8 opt-in gate rejects un-calibrated / un-checked configs** (T11
//!    negative case, FR-OP-10 + FR-EX-08). The runtime function must raise
//!    `VokraError::HifiganInt8VerifyMissing` — never silently fall back to
//!    fp32 / fp16.
//!
//! A future Kyutai / HiFi-GAN dump (via `scripts/gen_parity_fixtures.py`
//! extension proposed in T03) will add an `atol = 0.01` (NFR-QL-01) check
//! against external tensors. The mimi_rvq crate follows the same phased
//! approach — see `docs/adr/M3-06-mimi-rvq.md` §D5.

use vokra_core::VokraError;
use vokra_core::ir::graph::HifiGanAttrs;
use vokra_ops::{
    CalibrationTable, HifiGanConfig, HifiGanPrecision, HifiGanWeights, MrfBranchWeights,
    ResBlockLayer, UpsampleStageWeights, hifigan_generator,
};

/// A "V3-lite" parity-shaped attribute set: two upsample stages + two MRF
/// branches. Much smaller than a real HiFi-GAN preset (which has three or four
/// upsample stages and three MRF branches), but with enough structure to
/// exercise every code path — initial conv, transposed conv, dilated MRF conv,
/// residual add, per-branch average, final conv, tanh head.
fn parity_attrs() -> HifiGanAttrs {
    HifiGanAttrs {
        n_mels: 8,
        initial_channel: 12,
        upsample_rates: vec![4, 4],
        upsample_kernel_sizes: vec![8, 8],
        resblock_kernel_sizes: vec![3, 5],
        resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5]],
        sample_rate: 24_000,
        leaky_relu_slope: 0.1,
    }
}

/// A deterministic weight builder. Every weight cell is a bounded, smooth
/// function of its indices so tanh keeps the output well inside `(-1, 1)`.
fn parity_weights(attrs: &HifiGanAttrs) -> HifiGanWeights {
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
    // Upsample stages: halve the channel count each stage.
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
        // MRF branches.
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

/// Deterministic sinusoidal mel input.
fn parity_mel(n_mels: usize, n_frames: usize) -> Vec<f32> {
    let mut mel = Vec::with_capacity(n_mels * n_frames);
    for m in 0..n_mels {
        for f in 0..n_frames {
            mel.push(((m as f32 * 0.19 + f as f32 * 0.07).sin() * 0.5).clamp(-1.0, 1.0));
        }
    }
    mel
}

// ---- T10: FP32 parity -----------------------------------------------------

#[test]
fn fp32_forward_is_bit_identical_across_runs() {
    let attrs = parity_attrs();
    let weights = parity_weights(&attrs);
    let n_frames = 6;
    let mel = parity_mel(attrs.n_mels, n_frames);
    let a = hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp32()).unwrap();
    let b = hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp32()).unwrap();
    assert_eq!(
        a, b,
        "FP32 forward must be a pure function (no RNG, no parallel non-determinism)"
    );
}

#[test]
fn fp32_output_shape_follows_upsample_formula() {
    let attrs = parity_attrs();
    let weights = parity_weights(&attrs);
    let n_frames = 6;
    let mel = parity_mel(attrs.n_mels, n_frames);
    let out = hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp32()).unwrap();
    // Recompute the exact length the stacked transposed convs must yield.
    let mut expected = n_frames;
    for stage in 0..attrs.n_upsample_stages() {
        let up = &weights.upsample_weights[stage];
        let padding = up.kernel.saturating_sub(up.stride) / 2;
        expected = (expected - 1) * up.stride + up.kernel - 2 * padding;
    }
    assert_eq!(out.len(), expected);
}

#[test]
fn fp32_output_is_bounded_by_tanh_head() {
    let attrs = parity_attrs();
    let weights = parity_weights(&attrs);
    let n_frames = 6;
    let mel = parity_mel(attrs.n_mels, n_frames);
    let out = hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp32()).unwrap();
    for (i, v) in out.iter().enumerate() {
        assert!(v.is_finite(), "sample {i} non-finite: {v}");
        assert!(*v > -1.0 && *v < 1.0, "sample {i} outside (-1,1): {v}");
    }
}

// ---- T11: fp16 parity -----------------------------------------------------

#[test]
fn fp16_forward_matches_fp32_within_atol_001() {
    let attrs = parity_attrs();
    let weights = parity_weights(&attrs);
    let n_frames = 6;
    let mel = parity_mel(attrs.n_mels, n_frames);
    let fp32 = hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp32()).unwrap();
    let cfg_fp16 = HifiGanConfig {
        precision: HifiGanPrecision::Fp16,
        int8_enabled: false,
        calibration_data: None,
        spectral_check_passed: false,
    };
    let fp16 = hifigan_generator(&mel, n_frames, &weights, &attrs, &cfg_fp16).unwrap();
    assert_eq!(fp32.len(), fp16.len());
    let atol = 0.01_f32;
    let mut max_delta = 0.0_f32;
    let mut argmax = 0;
    for (i, (a, b)) in fp32.iter().zip(fp16.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_delta {
            max_delta = d;
            argmax = i;
        }
    }
    assert!(
        max_delta < atol,
        "fp16 vs fp32 parity: max |Δ| = {max_delta} at sample {argmax} (atol = {atol})"
    );
}

// ---- T11: INT8 gating negatives ------------------------------------------

#[test]
fn int8_gate_rejects_missing_calibration() {
    let attrs = parity_attrs();
    let weights = parity_weights(&attrs);
    let n_frames = 4;
    let mel = parity_mel(attrs.n_mels, n_frames);
    let cfg = HifiGanConfig {
        precision: HifiGanPrecision::Fp32,
        int8_enabled: true,
        calibration_data: None,
        spectral_check_passed: true, // even with passed check, missing calibration must fail
    };
    let err = hifigan_generator(&mel, n_frames, &weights, &attrs, &cfg).unwrap_err();
    assert!(
        matches!(err, VokraError::HifiganInt8VerifyMissing),
        "expected HifiganInt8VerifyMissing, got: {err}"
    );
}

#[test]
fn int8_gate_rejects_missing_spectral_check() {
    let attrs = parity_attrs();
    let weights = parity_weights(&attrs);
    let n_frames = 4;
    let mel = parity_mel(attrs.n_mels, n_frames);
    let table = CalibrationTable::new(vec![1.0; 3], vec![0; 3], 3).unwrap();
    let cfg = HifiGanConfig {
        precision: HifiGanPrecision::Fp32,
        int8_enabled: true,
        calibration_data: Some(table),
        spectral_check_passed: false,
    };
    let err = hifigan_generator(&mel, n_frames, &weights, &attrs, &cfg).unwrap_err();
    assert!(
        matches!(err, VokraError::HifiganInt8VerifyMissing),
        "expected HifiganInt8VerifyMissing, got: {err}"
    );
}

#[test]
fn int8_gate_default_config_is_disabled_and_runs_fp32() {
    let attrs = parity_attrs();
    let weights = parity_weights(&attrs);
    let n_frames = 4;
    let mel = parity_mel(attrs.n_mels, n_frames);
    // Default config: INT8 disabled, precision = FP32. This must succeed —
    // proving the "default-OFF" contract of FR-OP-10 works round-trip.
    let default_cfg = HifiGanConfig::default();
    assert!(!default_cfg.int8_enabled);
    let out = hifigan_generator(&mel, n_frames, &weights, &attrs, &default_cfg).unwrap();
    assert!(!out.is_empty());
    for v in &out {
        assert!(v.is_finite());
        assert!(*v > -1.0 && *v < 1.0);
    }
}

#[test]
fn int8_gate_with_both_proofs_but_kernel_unimplemented_errors_unsupported() {
    let attrs = parity_attrs();
    let weights = parity_weights(&attrs);
    let n_frames = 4;
    let mel = parity_mel(attrs.n_mels, n_frames);
    let table = CalibrationTable::new(vec![1.0; 3], vec![0; 3], 3).unwrap();
    // Atomic constructor: opt_in + calibration + spectral check pass all at once.
    let cfg = HifiGanConfig::fp32().with_int8_opt_in(table, true);
    let err = hifigan_generator(&mel, n_frames, &weights, &attrs, &cfg).unwrap_err();
    // FR-EX-08: because the INT8 kernel is deferred, the gate is not silently
    // relaxed — an explicit UnsupportedOp is what surfaces.
    assert!(
        matches!(err, VokraError::UnsupportedOp(_)),
        "expected UnsupportedOp (INT8 kernel deferred to consumer WP), got: {err}"
    );
}
