//! CI-resident round-trip test (M0-03-T17): a synthetic checkpoint is written
//! to disk, run through the public [`convert_file`] entry point, and the
//! resulting GGUF is loaded back with the runtime loader. No large real
//! checkpoint is committed; real-model E2E is a manual local run of the
//! `vokra-convert` binary.

use std::path::PathBuf;

use vokra_convert::{ModelKind, convert_file, convert_kokoro_file};
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

/// Builds a lean `whisper-base`-shaped safetensors buffer: `derive_name` keys
/// only off `(d_model, n_audio_layer, n_text_layer, n_mels)`, so `conv1`
/// `[512, 80, 1]` + six encoder / six decoder layer prefixes is enough to derive
/// the `whisper-base` label. `embed_tokens` is left tiny (`[8, 1]`) — n_vocab is
/// irrelevant to size detection and a small value skips the tokenizer embedding,
/// keeping the buffer to ~164 KB (the `conv1` payload).
fn synthetic_whisper_base_safetensors() -> Vec<u8> {
    let mut tensors: Vec<(String, Vec<u64>)> = vec![
        ("model.encoder.conv1.weight".to_string(), vec![512, 80, 1]),
        (
            "model.encoder.embed_positions.weight".to_string(),
            vec![1500, 1],
        ),
        (
            "model.decoder.embed_positions.weight".to_string(),
            vec![448, 1],
        ),
        ("model.decoder.embed_tokens.weight".to_string(), vec![8, 1]),
        (
            "model.encoder.layers.0.fc1.weight".to_string(),
            vec![2048, 1],
        ),
    ];
    for i in 0..6 {
        tensors.push((
            format!("model.encoder.layers.{i}.mlp.fc2.weight"),
            vec![1, 1],
        ));
    }
    for i in 0..6 {
        tensors.push((
            format!("model.decoder.layers.{i}.self_attn.q_proj.weight"),
            vec![1, 1],
        ));
    }

    let mut cursor = 0usize;
    let mut header_entries = Vec::new();
    for (name, shape) in &tensors {
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
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&vec![0u8; cursor]);
    out
}

#[test]
fn whisper_alignment_heads_roundtrip_through_convert_file() {
    // M4-20: a real-sized checkpoint's word-timestamp alignment-heads table must
    // survive convert_file -> disk -> GgufFile::open as a flat UINT32 array of
    // [layer, head] pairs. (Numeric word-timestamp accuracy vs. openai is an
    // owner verification — real audio + weights — and is not asserted here.)
    let input = tmp_path("whisper-align-in");
    let output = tmp_path("whisper-align-out");
    std::fs::write(&input, synthetic_whisper_base_safetensors()).expect("write input");

    convert_file(ModelKind::Whisper, &input, &output).expect("convert");
    let file = GgufFile::open(&output).expect("load output gguf");

    let heads: Vec<u32> = file
        .get("vokra.whisper.alignment_heads")
        .and_then(|v| v.as_array())
        .expect("vokra.whisper.alignment_heads present for a whisper-base checkpoint")
        .values
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect();

    assert!(
        !heads.is_empty() && heads.len() % 2 == 0,
        "must be [layer, head] pairs, got {heads:?}"
    );
    // Every index within the whisper-base grid (n_text_layer 6, n_text_head 8).
    for pair in heads.chunks_exact(2) {
        assert!(pair[0] < 6, "layer {} >= n_text_layer 6", pair[0]);
        assert!(pair[1] < 8, "head {} >= n_text_head 8", pair[1]);
    }

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

/// Builds a Kokoro-82M-*canonical-shaped* safetensors buffer:
///
/// - `text_encoder.embedding.weight` = `[178, 512]` (n_vocab = 178, hidden = 512)
/// - `voicepack` = `[3, 128]` (num_voices = 3, style_dim = 128 — matches the
///   canonical release's per-voice style vector width)
///
/// Payload is all-zero apart from a fingerprint on `voicepack` so a byte-exact
/// round-trip check on `voicepack` is still meaningful (mirroring the M2-07 T06
/// placeholder-path test's fingerprint pattern). Total buffer is well under
/// 1 MB (178·512·4 = 365 KB + 3·128·4 = 1.5 KB), comparable to the Whisper
/// synthetic checkpoint.
fn synthetic_kokoro_82m_shaped_safetensors() -> Vec<u8> {
    // (name, shape) — element count = product; F32 payload = 4 * elems.
    let entries: &[(&str, &[u64])] = &[
        // voicepack [num_voices=3, style_dim=128] → 1536 bytes.
        ("voicepack", &[3, 128]),
        // text_encoder.embedding.weight [n_vocab=178, hidden=512] → 364,544 bytes.
        ("text_encoder.embedding.weight", &[178, 512]),
    ];

    let mut cursor = 0usize;
    let mut header_entries = Vec::new();
    let mut sizes = Vec::new();
    for &(name, shape) in entries {
        let elems: u64 = shape.iter().product();
        let span = elems as usize * 4;
        let begin = cursor;
        let end = cursor + span;
        cursor = end;
        sizes.push(span);
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

    // Fingerprint on `voicepack` so we can assert byte-exact round-trip on the
    // voicepack tensor even though the payload is otherwise all-zero.
    let voicepack_bytes = sizes[0];
    let mut payload = vec![0u8; cursor];
    for (i, chunk) in payload[..voicepack_bytes].chunks_mut(4).enumerate() {
        let f = (i as f32) * 0.125 + 0.5;
        chunk.copy_from_slice(&f.to_le_bytes());
    }

    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&payload);
    out
}

/// Builds a synthetic Kokoro `config.json`:
///
/// - 178 phoneme symbols under the primary `vocab: {symbol: id}` shape
///   (Kokoro / misaki-style — matches upstream `KOTA` phoneme id assignment,
///   though the exact symbols here are placeholders because the upstream
///   `hexgrad/Kokoro-82M/config.json` is not accessible in this workspace);
/// - 3 voice names under the primary `voices: [str]` shape (again the exact
///   `af` / `am_michael` / `bf_emma` names are the standard hexgrad naming
///   convention but the ordering is our best-guess pending upstream access).
///
/// This is a **synthesized** config — the task instructions call this out
/// explicitly as an accepted fallback. When a real config.json becomes
/// available, this test's assertions on the exact symbol strings should stay
/// unchanged (they only check that the config's values, not the placeholder
/// `p{i}` synth, ended up in the GGUF).
fn synthetic_kokoro_config_json() -> Vec<u8> {
    // Build the vocab map: symbol → id. Symbols are `sym0`..`sym177` so the
    // assertion "no `p{i}` prefix" is meaningful (`sym` starts with `s`, not
    // `p`, and matches nothing the placeholder path emits).
    let mut vocab_pairs: Vec<String> = Vec::with_capacity(178);
    for i in 0..178 {
        vocab_pairs.push(format!(r#""sym{i}":{i}"#));
    }
    let vocab = format!("{{{}}}", vocab_pairs.join(","));
    let voices = r#"["af","am_michael","bf_emma"]"#;
    format!(r#"{{"vocab":{vocab},"voices":{voices}}}"#).into_bytes()
}

#[test]
fn kokoro_safetensors_with_config_roundtrips_through_convert_kokoro_file() {
    // M2-07-T17-fixup #3: exercise the config-driven Kokoro conversion.
    // Assumption: the exact `config.json` schema + voice-name choices are
    // synthesized because the upstream `hexgrad/Kokoro-82M/config.json` is
    // not accessible in this workspace. When a real config lands the parser
    // already accepts multiple aliases (`vocab` / `phoneme_symbols` /
    // `symbols`; `voices` / `voice_names`) so no test change should be
    // needed — only the schema documentation.

    let input = tmp_path("kokoro-cfg-in");
    let config = tmp_path("kokoro-cfg-json");
    let output = tmp_path("kokoro-cfg-out");
    std::fs::write(&input, synthetic_kokoro_82m_shaped_safetensors()).expect("write input");
    std::fs::write(&config, synthetic_kokoro_config_json()).expect("write config");

    let summary = convert_kokoro_file(&input, &config, &output).expect("convert_kokoro_file");
    // 2 F32 tensors in the synthetic checkpoint.
    assert_eq!(summary.tensor_count, 2);
    // Same 13-key surface as the placeholder path — the config path never
    // introduces new metadata keys, only replaces the *values* on
    // `phoneme_symbols` / `voice_names` / `num_voices`.
    assert_eq!(
        summary.metadata_count, 13,
        "config path emits the same 13 metadata keys as the placeholder path \
         (2 model + 11 kokoro)"
    );

    let file = GgufFile::open(&output).expect("load output gguf");
    assert_eq!(file.tensors().len(), 2);

    // Numeric hparams:
    //   - hidden_dim  = embedding rows-axis-1 = 512
    //   - style_dim   = voicepack cols-axis-1 = 128
    //   - num_voices  = config voice_names.len() (= 3), not voicepack rows
    //     (= 3 here; happens to agree, so no warning note).
    let u = |k: &str| file.get(k).and_then(|v| v.as_u64());
    assert_eq!(u("vokra.kokoro.sample_rate"), Some(24_000));
    assert_eq!(u("vokra.kokoro.style_dim"), Some(128));
    assert_eq!(u("vokra.kokoro.num_voices"), Some(3));
    assert_eq!(u("vokra.kokoro.hidden_dim"), Some(512));
    // iSTFT triple: unchanged from the placeholder path.
    assert_eq!(u("vokra.kokoro.istft.n_fft"), Some(20));
    assert_eq!(u("vokra.kokoro.istft.hop"), Some(5));
    assert_eq!(u("vokra.kokoro.istft.win_length"), Some(20));

    // `phoneme_symbols` carries the real 178-entry table from the config, not
    // the `p{i}` placeholder.
    let syms = file
        .get("vokra.kokoro.phoneme_symbols")
        .and_then(|v| v.as_array())
        .expect("phoneme_symbols present");
    assert_eq!(syms.values.len(), 178);
    // Every string starts with `sym`, not `p` — proves the placeholder path
    // did not fire. (`vokra.kokoro.*` values are `GgufMetadataValue::String`.)
    for (i, v) in syms.values.iter().enumerate() {
        let s = v.as_str().expect("string element");
        assert!(
            !s.starts_with('p') && s.starts_with("sym"),
            "phoneme_symbols[{i}] = {s:?} — expected `sym{i}` from config, \
             not the `p{i}` placeholder"
        );
    }

    // `voice_names` carries the config voice list.
    let voices = file
        .get("vokra.kokoro.voice_names")
        .and_then(|v| v.as_array())
        .expect("voice_names present");
    assert_eq!(voices.values.len(), 3);
    assert_eq!(voices.values[0].as_str(), Some("af"));
    assert_eq!(voices.values[1].as_str(), Some("am_michael"));
    assert_eq!(voices.values[2].as_str(), Some("bf_emma"));

    // No `vokra.frontend.*` chunk (Kokoro is TTS-only, no input front-end).
    assert!(file.get("vokra.frontend.n_fft").is_none());

    // `voicepack` bytes round-trip verbatim (fingerprint check).
    let voicepack_expected: Vec<u8> = (0..(3 * 128))
        .map(|i| (i as f32) * 0.125 + 0.5)
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(
        file.tensor_data("voicepack").unwrap(),
        voicepack_expected.as_slice()
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&output);
}
