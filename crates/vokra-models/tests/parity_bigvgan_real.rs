//! Real-weight BigVGAN base parity against NVIDIA's upstream Python forward.
//!
//! Generate the fixture with `tools/parity/bigvgan_dump_reference.py`, convert
//! the folded safetensors through `vokra-cli convert --model
//! bigvgan-base-24khz-100band`, then set `VOKRA_BIGVGAN_BASE_GGUF`.

use std::env;

use vokra_core::gguf::GgufFile;
use vokra_models::bigvgan::{BigVGan, BigVGanVariant};

const GGUF_ENV: &str = "VOKRA_BIGVGAN_BASE_GGUF";

#[test]
fn parity_bigvgan_base_real_weight_mel_to_waveform() {
    let Some(path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping real BigVGAN parity; clean skip, not a fabricated pass"
        );
        return;
    };
    let file = GgufFile::open(&path).unwrap_or_else(|error| {
        panic!("open opted-in BigVGAN GGUF {path}: {error}");
    });
    let model = BigVGan::from_gguf(&file).expect("bind complete BigVGAN tensor manifest");
    assert_eq!(model.variant(), BigVGanVariant::BaseV1_24khz100Band);

    let fixture = include_str!("../../../tools/parity/fixtures/bigvgan_base_reference.csv");
    let mut rows = fixture.lines();
    let input_row: Vec<&str> = rows.next().expect("input row").split(',').collect();
    let output_row: Vec<&str> = rows.next().expect("output row").split(',').collect();
    assert_eq!(input_row[0], "input");
    assert_eq!(output_row[0], "output");
    assert!(rows.next().is_none());
    let mel: Vec<f32> = input_row[1..]
        .iter()
        .map(|value| value.parse::<f32>().expect("input f32"))
        .collect();
    let expected: Vec<f32> = output_row[1..]
        .iter()
        .map(|value| value.parse::<f32>().expect("output f32"))
        .collect();
    assert_eq!(mel.len(), 100);
    assert_eq!(expected.len(), 256);

    let actual = model.decode(&mel, 1).expect("native BigVGAN forward");
    assert_eq!(actual.len(), expected.len());
    let max_abs = actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (value - reference).abs())
        .fold(0.0f32, f32::max);
    eprintln!("BigVGAN base real-weight parity: samples=256, max_abs={max_abs:e}");
    assert!(
        max_abs <= 2e-5,
        "BigVGAN base max |Δ| {max_abs:e} exceeds the 2e-5 FP32 bound"
    );
}
