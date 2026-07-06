//! CI-resident round-trip test (M0-03-T17): a synthetic checkpoint is written
//! to disk, run through the public [`convert_file`] entry point, and the
//! resulting GGUF is loaded back with the runtime loader. No large real
//! checkpoint is committed; real-model E2E is a manual local run of the
//! `vokra-convert` binary.

use std::path::PathBuf;

use vokra_convert::{ModelKind, convert_file};
use vokra_core::gguf::{FrontendSpec, GgufFile};

/// A unique temp path for this test process.
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("vokra-convert-it-{tag}-{}", std::process::id()));
    p
}

/// Builds a tiny valid safetensors buffer with two F32 tensors.
fn synthetic_safetensors() -> Vec<u8> {
    let a: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let b: Vec<u8> = [5.0f32, 6.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    let header = r#"{"model.encoder.conv1.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"model.decoder.bias":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}"#;
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&a);
    out.extend_from_slice(&b);
    out
}

#[test]
fn whisper_safetensors_roundtrips_through_convert_file() {
    let input = tmp_path("whisper-in");
    let output = tmp_path("whisper-out");
    std::fs::write(&input, synthetic_safetensors()).expect("write input");

    let summary = convert_file(ModelKind::Whisper, &input, &output).expect("convert");
    assert_eq!(summary.tensor_count, 2);
    // 2 model keys + 13 frontend keys + 13 `vokra.whisper.*` hyperparameter keys
    // (M0-06-T04: n_mels, n_audio_ctx/state/head/layer, n_text_ctx/state/head/
    // layer, n_vocab, ffn_dim, eot, decoder_start_ids).
    assert_eq!(summary.metadata_count, 28);

    let file = GgufFile::open(&output).expect("load output gguf");
    assert_eq!(file.tensors().len(), 2);
    assert!(file.tensor_info("model.encoder.conv1.weight").is_some());

    let spec = FrontendSpec::from_gguf(&file).expect("frontend spec reads back");
    assert_eq!(spec.n_fft, 400);
    assert_eq!(spec.sample_rate, 16_000);

    // The first tensor's bytes survive the whole pipeline intact.
    assert_eq!(
        file.tensor_data("model.encoder.conv1.weight").unwrap(),
        [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<_>>()
            .as_slice()
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}
