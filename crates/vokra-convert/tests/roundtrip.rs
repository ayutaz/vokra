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

/// Builds a synthetic Kokoro-like safetensors buffer whose tensor names track
/// the foundation shape-driver in `models::kokoro::write_hparams` so every
/// `vokra.kokoro.*` numeric hparam derives a non-zero value from this buffer.
///
/// The payloads are all-zero (only shapes drive assertions), but the two
/// leading tensors — `voicepack` and `text_encoder.embedding.weight` — carry a
/// short LE-`f32` prefix so we can assert byte-exact round-tripping on the wire.
fn synthetic_kokoro_safetensors() -> Vec<u8> {
    // (name, shape) — element count = product; F32 payload = 4 * elems.
    let entries: &[(&str, &[u64])] = &[
        // voicepack [num_voices=2, style_dim=4] → 32 bytes.
        ("voicepack", &[2, 4]),
        // text_encoder.embedding.weight [n_sym=3, hidden=8] → 96 bytes.
        ("text_encoder.embedding.weight", &[3, 8]),
        ("text_encoder.layers.0.attn.q_proj.weight", &[1, 1]),
        ("text_encoder.layers.1.attn.q_proj.weight", &[1, 1]),
        ("decoder.generator.upsamples.0.weight", &[1, 1]),
    ];

    let mut cursor = 0usize;
    let mut header_entries = Vec::new();
    for &(name, shape) in entries {
        let elems: u64 = shape.iter().product();
        let span = elems as usize * 4;
        let begin = cursor;
        let end = cursor + span;
        cursor = end;
        let dims = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header_entries.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{dims}],"data_offsets":[{begin},{end}]}}"#
        ));
    }
    let header = format!("{{{}}}", header_entries.join(","));

    // Byte-exact fingerprints for the first two tensors so the roundtrip
    // assertion has non-trivial content to compare. Remaining tensors stay
    // all-zero (their bytes are irrelevant to the assertion).
    let voicepack_payload: Vec<u8> = (0..8)
        .map(|i| i as f32 + 0.5)
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let embedding_payload: Vec<u8> = (0..24)
        .map(|i| (i as f32) * -0.25)
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let mut payload = Vec::with_capacity(cursor);
    payload.extend_from_slice(&voicepack_payload);
    payload.extend_from_slice(&embedding_payload);
    payload.resize(cursor, 0);

    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&payload);
    out
}

#[test]
fn kokoro_safetensors_roundtrips_through_convert_file() {
    let input = tmp_path("kokoro-in");
    let output = tmp_path("kokoro-out");
    std::fs::write(&input, synthetic_kokoro_safetensors()).expect("write input");

    let summary = convert_file(ModelKind::Kokoro, &input, &output).expect("convert");
    // 5 F32 tensors in the synthetic checkpoint.
    assert_eq!(summary.tensor_count, 5);
    // 2 model keys (`vokra.model.arch` / `.name`) + 11 `vokra.kokoro.*` keys
    // (sample_rate, style_dim, num_voices, n_text_layers, n_decoder_layers,
    // hidden_dim, istft.n_fft/hop/win_length, phoneme_symbols, voice_names).
    // Kokoro deliberately does NOT emit `vokra.frontend.*` — it is a TTS
    // decoder with no runtime-controlled input front-end.
    assert_eq!(summary.metadata_count, 13);

    let file = GgufFile::open(&output).expect("load output gguf");
    assert_eq!(file.tensors().len(), 5);

    // Every `vokra.kokoro.*` key from the T06 chunk design is present in the
    // on-disk GGUF.
    let u = |k: &str| file.get(k).and_then(|v| v.as_u64());
    assert_eq!(u("vokra.kokoro.sample_rate"), Some(24_000));
    assert_eq!(u("vokra.kokoro.num_voices"), Some(2));
    assert_eq!(u("vokra.kokoro.style_dim"), Some(4));
    assert_eq!(u("vokra.kokoro.hidden_dim"), Some(8));
    assert_eq!(u("vokra.kokoro.n_text_layers"), Some(2));
    assert_eq!(u("vokra.kokoro.n_decoder_layers"), Some(1));
    // iSTFT triple: pinned to Kokoro-82M's canonical values (n_fft=20,
    // hop=5, win_length=20) as of the T02 upstream inspection captured in
    // `crates/vokra-models/src/kokoro/data/upstream_tensors_v1_0.tsv` and
    // matching the module-level `KOKORO_ISTFT_*` constants. The runtime
    // consumer still rejects `0` for FR-EX-08, but the values are no
    // longer placeholders.
    assert_eq!(u("vokra.kokoro.istft.n_fft"), Some(20));
    assert_eq!(u("vokra.kokoro.istft.hop"), Some(5));
    assert_eq!(u("vokra.kokoro.istft.win_length"), Some(20));
    assert!(file.get("vokra.kokoro.phoneme_symbols").is_some());
    assert!(file.get("vokra.kokoro.voice_names").is_some());

    // No `vokra.frontend.*` chunk (Kokoro is TTS-only, no input front-end).
    assert!(file.get("vokra.frontend.n_fft").is_none());

    // Byte-exact tensor content round-trips through the pipeline for the two
    // tensors whose synthetic payloads carry a non-zero fingerprint.
    let voicepack_expected: Vec<u8> = (0..8)
        .map(|i| i as f32 + 0.5)
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(
        file.tensor_data("voicepack").unwrap(),
        voicepack_expected.as_slice()
    );
    let embedding_expected: Vec<u8> = (0..24)
        .map(|i| (i as f32) * -0.25)
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(
        file.tensor_data("text_encoder.embedding.weight").unwrap(),
        embedding_expected.as_slice()
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}
