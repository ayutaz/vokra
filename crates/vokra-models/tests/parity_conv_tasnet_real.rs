//! Independent parity against official Asteroid 0.7.0 Conv-TasNet.

use vokra_core::gguf::GgufFile;
use vokra_models::conv_tasnet::ConvTasnet;

const PCM: &[u8] = include_bytes!("fixtures/conv_tasnet/pcm.f32.bin");
const ENCODER: &[u8] = include_bytes!("fixtures/conv_tasnet/encoder.f32.bin");
const BOTTLENECK: &[u8] = include_bytes!("fixtures/conv_tasnet/bottleneck.f32.bin");
const MASK: &[u8] = include_bytes!("fixtures/conv_tasnet/mask.f32.bin");
const SEPARATED: &[u8] = include_bytes!("fixtures/conv_tasnet/separated.f32.bin");

fn f32s(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn compare(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
    relative_l1_bound: f32,
) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{label} finite"
    );
    let (index, max_abs) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap();
    let mean_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>()
        / actual.len() as f32;
    let relative_l1 = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>()
        / expected
            .iter()
            .map(|value| value.abs())
            .sum::<f32>()
            .max(1e-20);
    let (scaled_index, max_scaled) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| {
            (
                index,
                (actual - expected).abs() / (atol + rtol * expected.abs()).max(1e-20),
            )
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap();
    eprintln!(
        "ConvTasNet {label}: max_abs={max_abs:.9e} at {index} (actual={:.9e}, reference={:.9e}), mean_abs={mean_abs:.9e}, relative_l1={relative_l1:.9e}, max_scaled={max_scaled:.9e} at {scaled_index}",
        actual[index], expected[index]
    );
    assert!(
        max_scaled <= 1.0,
        "{label} violates atol={atol:.3e} + rtol={rtol:.3e} at {scaled_index}: scaled={max_scaled:.9e}"
    );
    assert!(
        relative_l1 <= relative_l1_bound,
        "{label} relative_l1={relative_l1:.9e}"
    );
}

#[test]
fn committed_reference_has_pinned_shapes_and_finite_values() {
    for (label, values, expected) in [
        ("pcm", f32s(PCM), 4_096),
        ("encoder", f32s(ENCODER), 512 * 255),
        ("bottleneck", f32s(BOTTLENECK), 128 * 255),
        ("mask", f32s(MASK), 512 * 255),
        ("separated", f32s(SEPARATED), 4_096),
    ] {
        assert_eq!(values.len(), expected, "{label}");
        assert!(values.iter().all(|value| value.is_finite()), "{label}");
    }
}

#[test]
fn converted_official_checkpoint_matches_asteroid() {
    let Some(path) = std::env::var_os("VOKRA_CONV_TASNET_GGUF") else {
        eprintln!(
            "[parity_conv_tasnet_real] SKIP: set VOKRA_CONV_TASNET_GGUF to a GGUF converted from the pinned official checkpoint"
        );
        return;
    };
    let file = GgufFile::open(path).expect("open corrected Conv-TasNet GGUF");
    let model = ConvTasnet::from_gguf(&file).expect("strict Conv-TasNet bind");
    assert_eq!(model.tensor_count(), 345);
    assert_eq!(model.sample_rate(), 16_000);
    assert_eq!(model.n_out(), 1);

    let pcm = f32s(PCM);
    let expected_encoder = f32s(ENCODER);
    let expected_bottleneck = f32s(BOTTLENECK);
    let expected_mask = f32s(MASK);
    let expected_separated = f32s(SEPARATED);

    let (encoder, frames) = model.encode_features(&pcm).expect("CPU encoder");
    assert_eq!(frames, 255);
    compare(
        "CPU encoder vs Asteroid",
        &encoder,
        &expected_encoder,
        1e-4,
        0.0,
        1e-6,
    );
    let (bottleneck, _) = model.bottleneck_features(&pcm).expect("CPU bottleneck");
    compare(
        "CPU bottleneck vs Asteroid",
        &bottleneck,
        &expected_bottleneck,
        1e-3,
        0.0,
        5e-5,
    );
    let (mask, _) = model.mask_features(&pcm).expect("CPU mask");
    compare(
        "CPU mask vs Asteroid",
        &mask,
        &expected_mask,
        0.30,
        0.0,
        0.001,
    );
    let separated = model.separate(&pcm).expect("CPU separation");
    assert_eq!(separated.len(), 1);
    compare(
        "CPU waveform vs Asteroid",
        &separated[0],
        &expected_separated,
        0.40,
        0.0,
        0.001,
    );

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use vokra_core::BackendKind;

        let metal = ConvTasnet::from_gguf(&file)
            .expect("strict Conv-TasNet bind for Metal")
            .with_backend(BackendKind::Metal);
        let (metal_mask, metal_frames) = metal.mask_features(&pcm).expect("Metal mask");
        assert_eq!(metal_frames, frames);
        compare("Metal mask vs CPU", &metal_mask, &mask, 0.30, 0.0, 0.001);
        let metal_separated = metal.separate(&pcm).expect("Metal separation");
        compare(
            "Metal waveform vs CPU",
            &metal_separated[0],
            &separated[0],
            0.40,
            0.0,
            0.001,
        );
    }
}
