//! Real-weight SpeechT5 HiFi-GAN parity against the official Transformers forward.
//!
//! Generate the fixture with
//! `tools/parity/speecht5_hifigan_dump_reference.py`, convert the prepared
//! safetensors through `vokra-cli convert --model speecht5-hifigan`, then set
//! `VOKRA_SPEECHT5_HIFIGAN_REAL_GGUF`.

use std::env;

use vokra_core::gguf::GgufFile;
use vokra_models::hifigan::HiFiGan;

const GGUF_ENV: &str = "VOKRA_SPEECHT5_HIFIGAN_REAL_GGUF";

#[test]
fn parity_speecht5_hifigan_real_weight_mel_to_waveform() {
    let Some(path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping real SpeechT5 HiFi-GAN parity; clean skip, not a fabricated pass"
        );
        return;
    };
    let file = GgufFile::open(&path).unwrap_or_else(|error| {
        panic!("open opted-in SpeechT5 HiFi-GAN GGUF {path}: {error}");
    });
    let model = HiFiGan::from_gguf(&file).expect("bind complete 158-tensor manifest");
    assert_eq!(model.sample_rate(), 16_000);
    assert_eq!(model.attrs().n_mels, 80);
    assert_eq!(model.attrs().total_upsample_factor(), 256);

    let fixture = include_str!("../../../tools/parity/fixtures/speecht5_hifigan_reference.csv");
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
    assert_eq!(expected.len(), 512);

    let actual = model
        .decode(&mel, 2)
        .expect("native SpeechT5 HiFi-GAN forward");
    assert_eq!(actual.len(), expected.len());
    let max_abs = actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (value - reference).abs())
        .fold(0.0f32, f32::max);
    eprintln!("SpeechT5 HiFi-GAN real-weight parity: samples=512, max_abs={max_abs:e}");
    // The official PyTorch CPU convolution kernels and Vokra's scalar FP32
    // kernels accumulate products in different orders across 78 convolutions.
    // The first real run measured 2.7298927e-5; 5e-5 keeps a narrow 1.8x
    // ceiling without pretending bitwise agreement between those kernels.
    assert!(
        max_abs <= 5e-5,
        "SpeechT5 HiFi-GAN max |Δ| {max_abs:e} exceeds the 5e-5 FP32 bound"
    );

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    {
        let metal = HiFiGan::from_gguf(&file)
            .expect("rebind SpeechT5 HiFi-GAN for Metal")
            .with_backend(vokra_core::BackendKind::Metal)
            .decode(&mel, 2)
            .expect("real SpeechT5 HiFi-GAN Metal forward");
        let gpu_max_abs = actual
            .iter()
            .zip(&metal)
            .map(|(cpu, gpu)| (cpu - gpu).abs())
            .fold(0.0f32, f32::max);
        assert!(
            gpu_max_abs <= 0.01,
            "SpeechT5 HiFi-GAN CPU/Metal max |Δ| {gpu_max_abs:e} exceeds the established FP32 GPU gate"
        );
    }
}
