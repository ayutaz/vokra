//! Independent PyTorch-reference parity for the vocoder convolution seam.
//!
//! The binary fixtures are generated on VAST by
//! `tools/parity/vocoder_conv_dump_reference.py`; they are intentionally not
//! synthesized or generated on the maintainer machine.  Keep these tests
//! ignored until the owner recovers and commits that manifest and its bytes.

#[path = "support/vocoder_conv_fixture.rs"]
mod fixture;

use fixture::{Fixture, Kind};
use vokra_backend_cpu::kernels;

// Every fixture operand is a signed power of two, and every accumulation is
// an exact integer well below the f32 24-bit significand limit.  This is an
// exactness assertion, not a newly derived tolerance.
const VOCODER_CONV_ATOL: f32 = 0.0;

fn assert_close(got: &[f32], expected: &[f32], context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: output length");
    for (index, (&actual, &reference)) in got.iter().zip(expected).enumerate() {
        assert!(
            (actual - reference).abs() <= VOCODER_CONV_ATOL,
            "{context}: index {index}: got {actual}, PyTorch {reference}, |delta|={}",
            (actual - reference).abs()
        );
    }
}

fn assert_common(fixture: &Fixture, kind: Kind) {
    assert_eq!(fixture.kind, kind);
    assert_eq!(fixture.input_shape[0], 1);
    assert_eq!(fixture.input_shape[1], fixture.in_channels);
    assert_eq!(
        fixture.input.len(),
        fixture.input_shape[1] * fixture.input_shape[2]
    );
    match kind {
        Kind::Conv1d => {
            assert_eq!(fixture.weight_shape[0], fixture.out_channels);
            assert_eq!(fixture.weight_shape[1], fixture.in_channels);
        }
        Kind::ConvTranspose1d => {
            assert_eq!(fixture.weight_shape[0], fixture.in_channels);
            assert_eq!(fixture.weight_shape[1], fixture.out_channels);
        }
    }
    assert_eq!(fixture.weight_shape[2], fixture.kernel);
    assert_eq!(fixture.output_shape[0], 1);
    assert_eq!(fixture.output_shape[1], fixture.out_channels);
    assert_eq!(
        fixture.output_shape[2] * fixture.output_shape[1],
        fixture.output.len()
    );
    assert_eq!(fixture.bias.len(), fixture.out_channels);
}

#[test]
#[ignore = "requires VAST-generated PyTorch fixture bytes under tests/parity/vocoder_conv"]
fn conv1d_dilated_matches_pytorch_functional_reference() {
    let fixture = fixture::load("conv1d_d2_s2_p2");
    assert_common(&fixture, Kind::Conv1d);
    assert_eq!(fixture.input_shape, [1, 2, 5]);
    assert_eq!(fixture.weight_shape, [3, 2, 3]);
    assert_eq!(fixture.output_shape, [1, 3, 3]);
    assert_eq!(fixture.stride, 2);
    assert_eq!(fixture.dilation, 2);
    assert_eq!(fixture.padding, 2);
    assert_eq!(fixture.output_padding, 0);

    let mut actual = vec![f32::NAN; fixture.output.len()];
    kernels::conv1d_f32_dilated(
        &fixture.input,
        fixture.in_channels,
        fixture.input_shape[2],
        &fixture.weight,
        fixture.out_channels,
        fixture.kernel,
        Some(&fixture.bias),
        fixture.stride,
        fixture.dilation,
        fixture.padding,
        &mut actual,
    )
    .expect("CPU dilated Conv1d");
    assert_close(&actual, &fixture.output, "CPU Conv1d vs PyTorch");
}

#[test]
#[ignore = "requires VAST-generated PyTorch fixture bytes under tests/parity/vocoder_conv"]
fn conv_transpose1d_matches_pytorch_functional_reference() {
    let fixture = fixture::load("conv_transpose1d_s3_p1_op2");
    assert_common(&fixture, Kind::ConvTranspose1d);
    assert_eq!(fixture.input_shape, [1, 2, 4]);
    assert_eq!(fixture.weight_shape, [2, 3, 4]);
    assert_eq!(fixture.output_shape, [1, 3, 13]);
    assert_eq!(fixture.stride, 3);
    assert_eq!(fixture.dilation, 1);
    assert_eq!(fixture.padding, 1);
    assert_eq!(fixture.output_padding, 2);

    let mut actual = vec![f32::NAN; fixture.output.len()];
    kernels::conv_transpose1d_f32(
        &fixture.input,
        fixture.in_channels,
        fixture.input_shape[2],
        &fixture.weight,
        fixture.out_channels,
        fixture.kernel,
        Some(&fixture.bias),
        fixture.stride,
        fixture.padding,
        fixture.output_padding,
        &mut actual,
    )
    .expect("CPU ConvTranspose1d");
    assert_close(&actual, &fixture.output, "CPU ConvTranspose1d vs PyTorch");
}
