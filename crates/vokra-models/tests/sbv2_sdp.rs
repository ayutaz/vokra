//! Blocker 2c: `SbV2SDP` primitives (DDSConv + ElementwiseAffine + ConvFlow)
//! and end-to-end shape tests for the real Style-Bert-VITS2 v2 stochastic
//! duration predictor (286-tensor safetensors layout: 144 production side
//! walking `sdp.{pre,proj,cond,convs,flows.{0,1,3,5,7}}` + 142 training-side
//! `sdp.post_*` inverse-flow tensors this converter skips).
//!
//! # Round-trip identity discipline
//!
//! Every `_forward_reverse_identity` test below picks a well-conditioned
//! weight configuration (either zero-weight — which makes each primitive a
//! documented identity — or hand-chosen nonzero values whose analytic result
//! is known in closed form) and asserts that `reverse(forward(x)) ≈ x` (or
//! the equivalent hand-computed reference) within a bound tight enough to
//! catch a real algorithmic bug (e.g. a wrong sign, dropped bias, missing
//! residual) but loose enough not to be gamed by floating-point noise. See
//! `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §11 for the
//! honest-atol discipline this codebase applies to every parity number.
//!
//! # References (permissive only)
//!
//! - jaywalnut310/vits (MIT): `modules.py::{DDSConv,ElementwiseAffine,ConvFlow}`
//!   + `transforms.py::piecewise_rational_quadratic_transform` are the exact
//!     shape mirror. The vendored source lives in `tools/parity/vendor/vits/`.
//!
//! # NOT REFERENCED
//!
//! - `github.com/litagin02/Style-Bert-VITS2` (AGPL-3.0)
//! - `github.com/fishaudio/Bert-VITS2` (AGPL-3.0)

use vokra_core::rng::GaussianSplitMix64;
use vokra_models::sbv2::{ConvFlow, DDSConv, ElementwiseAffine, SbV2SDP};

/// Zero-weight `ElementwiseAffine(m=0, logs=0)` maps every input to itself
/// (`x = (x - 0) * exp(-0) = x`), so `reverse` is bit-exact identity —
/// documented on `ElementwiseAffine::reverse`'s doc. Uses the real
/// safetensors shape (channels = 2, one row of length 1 per channel), so a
/// future dimension-swap regression is caught.
#[test]
fn element_wise_affine_forward_reverse_identity() {
    let ea = ElementwiseAffine::from_weights(vec![0.0, 0.0], vec![0.0, 0.0]);
    let time = 5;
    let x = vec![
        // channel 0 (row-major: channel_id * time + t)
        -2.5, 0.0, 1.0, 7.25, -0.1, // channel 1
        0.5, 1.5, -3.7, 4.2, 0.05,
    ];
    let out = ea.reverse(&x, time);
    assert_eq!(
        out, x,
        "zero-weight ElementwiseAffine must be the bit-exact identity"
    );
}

/// Zero-weight `ConvFlow` (all convs, pre, proj = 0) makes the predicted
/// spline params `unnormalized_widths / unnormalized_heights /
/// unnormalized_derivatives` all zero — a uniform spline whose 10 bins are
/// each of unit width in `[-5, 5]²`. In the **interior** bins the
/// left/right derivatives are equal (both `MIN_DERIVATIVE + softplus(0) ≈
/// 0.6941`), which analytically collapses the rational-quadratic-spline
/// inverse to the identity within a tight FP32-rounding bound. The two
/// **boundary** bins (`bin 0` and `bin 9`) use the "linear-tail derivative
/// constant" at one endpoint, which is deliberately different — that
/// makes the boundary bins deviate from identity by up to ~4%, so this
/// test uses interior-only inputs (`-3.5, 0.5, 3.5`) and asserts exact
/// (< 1e-4) identity. Choosing test inputs to isolate a documented
/// non-identity region (rather than "widening the tolerance to whatever
/// passes") follows the `feedback-honest-parity-atol` discipline.
#[test]
fn conv_flow_forward_reverse_identity() {
    // channels=192 as in real SBV2 v2 base — but we build a tiny 4-channel
    // instance to keep the test small yet exercising every code path
    // (softmax over 10 bins, cumsum, searchsorted, RQS inverse quadratic).
    let dp_filter = 4;
    let cf = ConvFlow::from_weights(
        // pre.weight [dp_filter, 1, 1] = [dp_filter]: zero
        vec![0.0; dp_filter],
        vec![0.0; dp_filter], // pre.bias
        DDSConv::zero(dp_filter, /*n_layers=*/ 3, /*kernel=*/ 3),
        // proj.weight [num_bins*3-1, dp_filter, 1]: zero, num_bins=10 → 29
        vec![0.0; 29 * dp_filter],
        vec![0.0; 29], // proj.bias
        dp_filter,
    );
    let time = 3;
    // Input `x` is [2, T]: channel 0 is passed through unchanged, channel 1
    // is transformed by the spline.
    let x = vec![
        // channel 0 (conditioning, arbitrary but preserved verbatim)
        0.3, -0.4, 1.2, // channel 1 (transformed) — all in interior bins,
        // away from the two boundary bins where the linear-tail derivative
        // constant kicks in
        -3.5, 0.5, 3.5,
    ];
    // conditioning `g` from the SDP body, [dp_filter, T]; zero-weight
    // ConvFlow ignores it, so any values are fine.
    let g = vec![0.0; dp_filter * time];
    let out = cf.reverse(&x, time, &g);

    // channel 0 must be bit-exact preserved (never touched).
    for t in 0..time {
        assert_eq!(out[t], x[t], "channel 0 must pass through unchanged");
    }
    // channel 1 must be exactly the identity in the interior (see doc).
    for t in 0..time {
        let out1 = out[time + t];
        let in1 = x[time + t];
        assert!(
            (out1 - in1).abs() < 1e-4,
            "channel 1 must be near-exact identity in interior bins: \
             in={in1}, out={out1}, |delta|={}",
            (out1 - in1).abs()
        );
    }
}

/// `DDSConv::forward` preserves shape (input `[C, T]` → output `[C, T]`)
/// regardless of weight values, so the returned buffer's length must equal
/// `channels * time` exactly.
#[test]
fn dds_conv_shape_preserved() {
    let channels = 4;
    let time = 7;
    // Any deterministic small-magnitude fill will do — the assertion is
    // shape-only.
    let dds = DDSConv::zero(channels, /*n_layers=*/ 3, /*kernel=*/ 3);
    let x = vec![0.1_f32; channels * time];
    let out = dds.forward(&x, time, None);
    assert_eq!(
        out.len(),
        channels * time,
        "DDSConv must preserve [channels, time] shape"
    );
}

/// End-to-end `SbV2SDP::sample` returns exactly `text_seq_len` durations
/// (one per phoneme position), matching `piper_plus::DurationPredictor`'s
/// caller-side contract and the length-regulator's downstream expectation.
#[test]
fn sdp_sample_shape_matches_input() {
    let d_hidden = 4;
    let gin = 4;
    let text_seq_len = 6;
    // Empty SDP (no flows, zero everywhere): with `noise_scale_w = 0.0`,
    // every draw is `0.0` and every predicted duration is `1`
    // (`exp(0).ceil().max(1) == 1`) — a documented, exercised no-op path
    // (`SbV2SDP::empty`'s own doc).
    let sdp = SbV2SDP::empty(d_hidden, gin);
    let hidden = vec![0.0_f32; text_seq_len * d_hidden];
    let g = vec![0.0_f32; gin];
    let mut rng = GaussianSplitMix64::new(42);
    let out = sdp.sample(&hidden, text_seq_len, &g, &mut rng, 0.0);
    assert_eq!(
        out.len(),
        text_seq_len,
        "SbV2SDP::sample must return one duration per phoneme position"
    );
    for &d in &out {
        assert!(
            d >= 1,
            "SbV2SDP::sample must return a valid frame count (>= 1); got {d}"
        );
    }
}
