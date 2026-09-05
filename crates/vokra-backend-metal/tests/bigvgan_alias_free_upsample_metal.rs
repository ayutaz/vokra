//! BigVGAN alias-free `UpSample1d` Metal parity.
//!
//! This suite targets the dedicated centered/replicate/asymmetric-crop
//! primitive. It intentionally compares against a hand-computed scalar
//! fixture rather than the generic causal FIR kernel, so a causal/polyphase
//! implementation cannot pass by sharing the same mistake.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use vokra_backend_metal::MetalContext;
use vokra_core::VokraError;

const ATOL: f32 = 1e-6;

macro_rules! ctx_or_skip {
    () => {
        match MetalContext::new() {
            Ok(context) => context,
            Err(error) => {
                eprintln!("no Metal device (BigVGAN alias-free upsample); skipping: {error}");
                return;
            }
        }
    };
}

#[test]
fn centered_replicate_crop_matches_independent_fixture_and_differs_from_causal() {
    let ctx = ctx_or_skip!();
    let input = ctx.upload(&[1.0, 2.0, 3.0]).expect("input upload");
    let filter = ctx.upload(&[0.5, 0.25, 0.1, 0.05]).expect("filter upload");
    let mut output = ctx.alloc_dev(6).expect("output allocation");

    ctx.bigvgan_alias_free_upsample_dev(&mut output, &input, &filter, 2, 1, 3, 4)
        .expect("BigVGAN alias-free upsample");
    let mut got = [0.0f32; 6];
    ctx.download(&output, &mut got).expect("output download");

    // Scalar geometry: pad=1 gives padded input [1,1,2,3,3]. The grouped
    // transpose output has 12 positions; crop_left=crop_right=3 leaves core
    // positions 3..8. Walking padded positions in order yields:
    //   [0.1 + 0.5, 0.2 + 2.0, 0.1 + 1.0, 0.4 + 3.0,
    //    0.2 + 1.5, 0.6 + 3.0]
    // where every term includes the transposed-convolution scale ratio=2.
    let expected = [0.6, 2.2, 1.1, 3.4, 1.7, 3.6];
    for (index, (&actual, &reference)) in got.iter().zip(&expected).enumerate() {
        let delta = (actual - reference).abs();
        assert!(
            delta <= ATOL,
            "BigVGAN centered fixture index {index}: got {actual}, expected {reference}, delta {delta}"
        );
    }

    // The generic API is intentionally causal. Its output must not accidentally
    // become the centered BigVGAN result while the two APIs coexist.
    let mut causal = ctx.alloc_dev(6).expect("causal output allocation");
    ctx.anti_aliased_upsample_dev(&mut causal, &input, &filter, 2, 1, 3, 4)
        .expect("causal upsample");
    let mut causal_host = [0.0f32; 6];
    ctx.download(&causal, &mut causal_host)
        .expect("causal output download");
    let differs = causal_host
        .iter()
        .zip(expected)
        .any(|(&causal_value, centered_value)| (causal_value - centered_value).abs() > 0.1);
    assert!(
        differs,
        "generic causal upsample unexpectedly matches centered BigVGAN fixture: {causal_host:?}"
    );
}

#[test]
fn validates_shape_owner_and_empty_semantics() {
    let ctx = ctx_or_skip!();
    let input = ctx.upload(&[1.0, 2.0, 3.0]).expect("input upload");
    let filter = ctx.upload(&[0.5, 0.25, 0.1, 0.05]).expect("filter upload");
    let mut wrong_output = ctx.alloc_dev(5).expect("wrong output allocation");
    assert!(matches!(
        ctx.bigvgan_alias_free_upsample_dev(&mut wrong_output, &input, &filter, 2, 1, 3, 4,),
        Err(VokraError::InvalidArgument(_))
    ));
    assert_eq!(
        ctx.submission_count(),
        0,
        "shape rejection must not dispatch"
    );

    let mut bad_ratio = ctx.alloc_dev(6).expect("bad-ratio output allocation");
    assert!(matches!(
        ctx.bigvgan_alias_free_upsample_dev(&mut bad_ratio, &input, &filter, 0, 1, 3, 4,),
        Err(VokraError::InvalidArgument(_))
    ));
    assert_eq!(ctx.submission_count(), 0, "invalid ratio must not dispatch");

    let other = match MetalContext::new() {
        Ok(context) => context,
        Err(error) => {
            eprintln!("second Metal context unavailable; owner portion skipped: {error}");
            return;
        }
    };
    let foreign_input = other.upload(&[1.0, 2.0, 3.0]).expect("foreign upload");
    let mut owner_output = ctx.alloc_dev(6).expect("owner output allocation");
    assert!(matches!(
        ctx.bigvgan_alias_free_upsample_dev(
            &mut owner_output,
            &foreign_input,
            &filter,
            2,
            1,
            3,
            4,
        ),
        Err(VokraError::InvalidArgument(_))
    ));
    assert_eq!(
        ctx.submission_count(),
        0,
        "owner rejection must not dispatch"
    );

    let empty_input = ctx.upload(&[]).expect("empty input upload");
    let mut empty_output = ctx.alloc_dev(0).expect("empty output allocation");
    ctx.bigvgan_alias_free_upsample_dev(&mut empty_output, &empty_input, &filter, 2, 3, 0, 4)
        .expect("empty time must be a no-op");
    assert_eq!(ctx.submission_count(), 0, "empty shape must not dispatch");
}
