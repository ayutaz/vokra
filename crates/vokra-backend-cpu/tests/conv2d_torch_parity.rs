//! Independent PyTorch-reference parity for the Conv2d compute seams.
//!
//! Fixtures are generated remotely by
//! `tools/parity/conv2d_dump_reference.py`.  This test is intentionally
//! ignored until that directory is populated with the generated raw-f32 files
//! and pinned manifest digests; it must be run explicitly with `--ignored` on
//! the reference-generation host or a matching offline checkout.

#[path = "support/conv2d_fixture.rs"]
mod fixture;

use fixture::{Fixture, Kind};
use vokra_backend_cpu::kernels;

fn assert_equal(actual: &[f32], reference: &[f32], context: &str) {
    assert_eq!(actual.len(), reference.len(), "{context}: output length");
    for (index, (&got, expected)) in actual.iter().zip(reference).enumerate() {
        assert!(
            got.to_bits() == expected.to_bits(),
            "{context}: index {index}: got {got:?} ({:#010x}), PyTorch {expected:?} ({:#010x})",
            got.to_bits(),
            expected.to_bits()
        );
    }
}

fn assert_case_shape(fixture: &Fixture) {
    assert_eq!(fixture.input_shape[0], 1);
    assert_eq!(fixture.output_shape[0], 1);
    assert_eq!(fixture.input_shape[1], fixture.in_channels);
    assert_eq!(fixture.output_shape[1], fixture.out_channels);
    assert_eq!(fixture.bias.len(), fixture.out_channels);
    assert_eq!(fixture.input.len(), fixture.input_shape.iter().product());
    assert_eq!(fixture.output.len(), fixture.output_shape.iter().product());
    assert!(fixture.groups > 0);
    assert_eq!(fixture.in_channels % fixture.groups, 0);
    assert_eq!(fixture.out_channels % fixture.groups, 0);
    match fixture.kind {
        Kind::Conv2d => {
            assert_eq!(fixture.weight_shape[0], fixture.out_channels);
            assert_eq!(
                fixture.weight_shape[1],
                fixture.in_channels / fixture.groups
            );
        }
        Kind::ConvTranspose2d => {
            assert_eq!(fixture.weight_shape[0], fixture.in_channels);
            assert_eq!(
                fixture.weight_shape[1],
                fixture.out_channels / fixture.groups
            );
        }
    }
    assert_eq!(fixture.weight_shape[2], fixture.kernel[0]);
    assert_eq!(fixture.weight_shape[3], fixture.kernel[1]);
}

fn assert_case_contract(fixture: &Fixture) {
    match fixture.name.as_str() {
        "conv2d_grouped_d2_s21_p12" => {
            assert_eq!(fixture.kind, Kind::Conv2d);
            assert_eq!(fixture.input_shape, [1, 4, 5, 6]);
            assert_eq!(fixture.weight_shape, [6, 2, 3, 2]);
            assert_eq!(fixture.output_shape, [1, 6, 2, 9]);
            assert_eq!(fixture.kernel, [3, 2]);
            assert_eq!(fixture.stride, [2, 1]);
            assert_eq!(fixture.padding, [1, 2]);
            assert_eq!(fixture.dilation, [2, 1]);
            assert_eq!(fixture.output_padding, [0, 0]);
            assert_eq!(fixture.groups, 2);
        }
        "conv_transpose2d_grouped_d21_s23_p12_op12" => {
            assert_eq!(fixture.kind, Kind::ConvTranspose2d);
            assert_eq!(fixture.input_shape, [1, 4, 3, 4]);
            assert_eq!(fixture.weight_shape, [4, 3, 2, 3]);
            assert_eq!(fixture.output_shape, [1, 6, 6, 10]);
            assert_eq!(fixture.kernel, [2, 3]);
            assert_eq!(fixture.stride, [2, 3]);
            assert_eq!(fixture.padding, [1, 2]);
            assert_eq!(fixture.dilation, [2, 1]);
            assert_eq!(fixture.output_padding, [1, 2]);
            assert_eq!(fixture.groups, 2);
        }
        "conv_transpose2d_op1_lt_dilation" => {
            assert_eq!(fixture.kind, Kind::ConvTranspose2d);
            assert_eq!(fixture.input_shape, [1, 1, 1, 1]);
            assert_eq!(fixture.weight_shape, [1, 1, 2, 1]);
            assert_eq!(fixture.output_shape, [1, 1, 4, 1]);
            assert_eq!(fixture.kernel, [2, 1]);
            assert_eq!(fixture.stride, [1, 2]);
            assert_eq!(fixture.padding, [0, 0]);
            assert_eq!(fixture.dilation, [2, 1]);
            assert_eq!(fixture.output_padding, [1, 0]);
            assert_eq!(fixture.groups, 1);
            assert_eq!(fixture.output_padding[0], fixture.stride[0]);
            assert!(fixture.output_padding[0] < fixture.dilation[0]);
        }
        other => panic!("unexpected Conv2d fixture case {other:?}"),
    }
}

fn compare_case(fixture: &Fixture) {
    assert_case_shape(fixture);
    assert_case_contract(fixture);
    let mut actual = vec![f32::NAN; fixture.output.len()];
    match fixture.kind {
        Kind::Conv2d => kernels::conv2d_f32(
            &fixture.input,
            fixture.in_channels,
            fixture.input_shape[2],
            fixture.input_shape[3],
            &fixture.weight,
            fixture.out_channels,
            fixture.kernel[0],
            fixture.kernel[1],
            Some(&fixture.bias),
            (fixture.stride[0], fixture.stride[1]),
            (fixture.padding[0], fixture.padding[1]),
            (fixture.dilation[0], fixture.dilation[1]),
            fixture.groups,
            &mut actual,
        )
        .expect("CPU Conv2d"),
        Kind::ConvTranspose2d => kernels::conv_transpose2d_f32(
            &fixture.input,
            fixture.in_channels,
            fixture.input_shape[2],
            fixture.input_shape[3],
            &fixture.weight,
            fixture.out_channels,
            fixture.kernel[0],
            fixture.kernel[1],
            Some(&fixture.bias),
            (fixture.stride[0], fixture.stride[1]),
            (fixture.padding[0], fixture.padding[1]),
            (fixture.dilation[0], fixture.dilation[1]),
            (fixture.output_padding[0], fixture.output_padding[1]),
            fixture.groups,
            &mut actual,
        )
        .expect("CPU ConvTranspose2d"),
    }
    assert_equal(&actual, &fixture.output, &fixture.name);
}

#[test]
#[ignore = "run after remote PyTorch fixture generation; no local Torch/model execution"]
fn conv2d_and_conv_transpose2d_match_pytorch_reference() {
    for fixture in fixture::load_all() {
        compare_case(&fixture);
    }
}
