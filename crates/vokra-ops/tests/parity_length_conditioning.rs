//! Parity tests for `length_conditioning` (M3-08-T06; FR-OP-71).
//!
//! The op is a closed-form scalar formula — `frames = round(seconds ·
//! sample_rate / hop_length)` (mode A) or `frames = round(ref_speech_frames ·
//! text_ratio)` (mode B) — so the "PyTorch reference" this file compares
//! against is the same closed form recomputed inline. That makes the test an
//! **internal oracle**: no fixture, no Python, no `parity/fixtures` file. The
//! WP directive requires atol=0.01 (NFR-QL-01); since frame counts are `u32`,
//! this becomes an **exact** integer equality (max|Δ| == 0), a strictly
//! stronger bar. Real-reference parity against F5-TTS / CosyVoice2 upstream
//! (a deeper coupling than the linear closed form) is a consumer-WP concern
//! and lands with M3-09.
//!
//! Both modes are driven through both the direct op function
//! (`vokra_ops::length_conditioning`) and the IR dispatch path
//! (`OpKind::LengthConditioning` → `dispatch::dispatch`) so the graph-node and
//! direct-call contracts stay identical (the M2-05-T10 "graph path == direct
//! call" pattern).

use vokra_core::OpKind;
use vokra_ops::attrs::{DurationUnit, LengthConditioningAttrs, LengthConditioningMode};
use vokra_ops::{OpValue, dispatch, length_conditioning};

/// The closed-form reference (mode A). Kept locally to be visibly identical to
/// the op's own formula: no chance of a "fabricated pass" through a
/// different derivation.
fn ref_user_specified_frames(
    duration: f32,
    unit: DurationUnit,
    sample_rate: u32,
    hop_length: u32,
) -> u32 {
    let frames_f32 = match unit {
        DurationUnit::Frames => duration,
        DurationUnit::Seconds => {
            (f64::from(duration) * f64::from(sample_rate) / f64::from(hop_length)) as f32
        }
    };
    frames_f32.round() as u32
}

/// The closed-form reference (mode B).
fn ref_ref_linear(ref_speech_frames: u32, text_ratio: f32) -> u32 {
    ((f64::from(ref_speech_frames) * f64::from(text_ratio)) as f32).round() as u32
}

fn dispatch_frames(op: OpKind) -> u32 {
    let out = dispatch(&op, &[]).expect("length_conditioning dispatch succeeds");
    let (shape, data) = out[0].as_real().expect("real output");
    assert_eq!(shape, &[1], "scalar tensor");
    assert_eq!(data.len(), 1);
    // The dispatch path encodes u32 as an f32 for OpValue::Real; the exact
    // decode below is what a consumer (M3-09) will do.
    let v = data[0];
    assert!(v.is_finite() && v >= 0.0);
    v.round() as u32
}

#[test]
fn mode_a_frames_matches_reference_over_a_grid() {
    // Explicit frame input covers integer, sub-integer, and boundary values.
    for &d in &[0.0_f32, 1.0, 41.0, 42.4, 42.5, 42.6, 199.5, 200.0, 65_535.0] {
        let attrs = LengthConditioningAttrs::user_specified_frames(d);
        let direct = length_conditioning(&attrs).unwrap();
        let via_graph = dispatch_frames(OpKind::LengthConditioning(attrs));
        let expect = ref_user_specified_frames(d, DurationUnit::Frames, 0, 0);
        assert_eq!(direct, expect, "direct @ d={d}");
        assert_eq!(via_graph, expect, "graph @ d={d}");
    }
}

#[test]
fn mode_a_seconds_matches_reference_over_a_grid() {
    // Sweep the common speech front-end rates and hop sizes; the reference is
    // the exact same closed form, so max|Δ| must be 0.
    let cases: &[(f32, u32, u32)] = &[
        (0.0, 16_000, 160),
        (0.1, 16_000, 160),
        (0.5, 16_000, 160),
        (1.0, 16_000, 160),
        (2.0, 22_050, 256),
        (3.5, 24_000, 300),
        (10.0, 44_100, 512),
    ];
    for &(sec, sr, hop) in cases {
        let attrs = LengthConditioningAttrs::user_specified_seconds(sec, sr, hop);
        let direct = length_conditioning(&attrs).unwrap();
        let via_graph = dispatch_frames(OpKind::LengthConditioning(attrs));
        let expect = ref_user_specified_frames(sec, DurationUnit::Seconds, sr, hop);
        assert_eq!(direct, expect, "direct @ sec={sec} sr={sr} hop={hop}");
        assert_eq!(via_graph, expect, "graph @ sec={sec} sr={sr} hop={hop}");
    }
}

#[test]
fn mode_b_matches_reference_over_a_grid() {
    // The WP directive's concrete case (ref=100, ratio=2.0 -> 200) plus a
    // spread of realistic ratios (compression, identity, mild expansion,
    // strong expansion). Every case checked bit-exact vs the closed form.
    let cases: &[(u32, f32)] = &[
        (100, 2.0),
        (100, 1.0),
        (100, 0.5),
        (100, 0.0),
        (157, 1.31),
        (1024, 0.87),
        (1, 100.0),
        (65_535, 1.0),
    ];
    for &(refn, ratio) in cases {
        let attrs = LengthConditioningAttrs::ref_linear(refn, ratio);
        let direct = length_conditioning(&attrs).unwrap();
        let via_graph = dispatch_frames(OpKind::LengthConditioning(attrs));
        let expect = ref_ref_linear(refn, ratio);
        assert_eq!(direct, expect, "direct @ ref={refn} ratio={ratio}");
        assert_eq!(via_graph, expect, "graph @ ref={refn} ratio={ratio}");
    }
}

#[test]
fn wp_directive_shape_case() {
    // Directly reflects the ticket's headline example: given ref_len=100,
    // target=200, output len == 200. This is the "shape correctness" that
    // makes an M3-09 Flow Matching sampler compute a 200-frame batch.
    let attrs = LengthConditioningAttrs::ref_linear(100, 2.0);
    let n = length_conditioning(&attrs).unwrap();
    assert_eq!(n, 200);

    let via_graph = dispatch_frames(OpKind::LengthConditioning(attrs));
    assert_eq!(via_graph, 200);
}

#[test]
fn dispatch_rejects_non_zero_arity() {
    // The op takes no runtime inputs; supplying one is an explicit error
    // (FR-EX-08).
    let attrs = LengthConditioningAttrs::ref_linear(100, 2.0);
    let bogus = OpValue::real(vec![1], vec![7.0]);
    let e = dispatch(&OpKind::LengthConditioning(attrs), &[bogus]).unwrap_err();
    assert!(matches!(e, vokra_core::VokraError::InvalidArgument(_)));
}

#[test]
fn dispatch_propagates_mode_a_seconds_missing_sample_rate() {
    // A graph node that specifies Seconds but leaves sample_rate=0 must
    // surface as InvalidArgument through the dispatch path — the same rule
    // the direct op function enforces, verified end-to-end.
    let bad = LengthConditioningAttrs {
        mode: LengthConditioningMode::UserSpecified { duration: 1.0 },
        unit: DurationUnit::Seconds,
        sample_rate: 0,
        hop_length: 160,
    };
    let e = dispatch(&OpKind::LengthConditioning(bad), &[]).unwrap_err();
    assert!(matches!(e, vokra_core::VokraError::InvalidArgument(_)));
}
