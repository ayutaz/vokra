//! Real-weight SpeechBrain HiFi-GAN parity against the official 1.0.3 forward.
//!
//! Generate the fixture with `tools/parity/hifigan_dump_reference.py`, prepare
//! and convert the checkpoint with `hifigan_prepare_checkpoint.py` plus
//! `vokra-cli convert --model hifigan-vocoder`, then set
//! `VOKRA_HIFIGAN_REAL_GGUF` to opt into this test.

use std::env;

use vokra_core::gguf::GgufFile;
use vokra_models::hifigan::HiFiGan;

const GGUF_ENV: &str = "VOKRA_HIFIGAN_REAL_GGUF";

#[test]
fn parity_speechbrain_hifigan_real_weight_mel_to_waveform() {
    let Some(path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping real SpeechBrain HiFi-GAN parity; clean skip, not a fabricated pass"
        );
        return;
    };
    let file = GgufFile::open(&path).unwrap_or_else(|error| {
        panic!("open opted-in SpeechBrain HiFi-GAN GGUF {path}: {error}");
    });
    let model = HiFiGan::from_gguf(&file).expect("bind complete 156-tensor folded manifest");
    assert_eq!(model.sample_rate(), 22_050);
    assert_eq!(model.attrs().n_mels, 80);
    assert_eq!(model.attrs().total_upsample_factor(), 256);

    let fixture = include_str!("../../../tools/parity/fixtures/hifigan_reference.csv");
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
    assert_eq!(mel.len(), 80 * 2);
    // SpeechBrain inference replicate-pads five frames on both sides and does
    // not crop: (2 + 10) * 256 = 3072 samples.
    assert_eq!(expected.len(), 3_072);

    let actual = model
        .decode(&mel, 2)
        .expect("native SpeechBrain HiFi-GAN forward");
    assert_eq!(actual.len(), expected.len());
    let max_abs = actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (value - reference).abs())
        .fold(0.0f32, f32::max);
    eprintln!("SpeechBrain HiFi-GAN real-weight parity: samples=3072, max_abs={max_abs:e}");
    // Initial bound matches the independently justified SpeechT5 sibling.
    // Tighten or document any change only after the first official run.
    assert!(
        max_abs <= 5e-5,
        "SpeechBrain HiFi-GAN max |Δ| {max_abs:e} exceeds the 5e-5 FP32 bound"
    );
}
