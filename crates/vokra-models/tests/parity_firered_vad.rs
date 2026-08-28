//! Env-gated real-weight parity for FireRedTeam/FireRedVAD Stream-VAD.
//!
//! The reference JSON is produced by executing the official ONNX graph
//! directly from pinned upstream commit
//! `c30ec49e8cc69642b0ee65362eba11b9d11c6e54`; it is independent of the Rust
//! implementation. Both the normalized-feature forward and the complete
//! PCM → Kaldi fbank → CMVN → DFSMN path use a pre-registered `1e-4` bound.

use std::path::Path;

use vokra_core::engines::VadEngine;
use vokra_models::firered_vad::FireredVad;

const GGUF_ENV: &str = "VOKRA_FIRERED_VAD_REAL_GGUF";
const REFERENCE_ENV: &str = "VOKRA_FIRERED_VAD_REFERENCE_JSON";
const WAV_ENV: &str = "VOKRA_FIRERED_VAD_REAL_WAV";
const PROB_ATOL: f32 = 1.0e-4;

#[derive(Debug)]
struct Reference {
    sample_rate: u32,
    n_frames: usize,
    features: Vec<f32>,
    probabilities: Vec<f32>,
}

fn number(value: &vokra_core::json::JsonValue, context: &str) -> f32 {
    let value = match value {
        vokra_core::json::JsonValue::Float(value) => *value,
        vokra_core::json::JsonValue::Int(value) => *value as f64,
        other => panic!("{context}: expected number, got {other:?}"),
    };
    assert!(value.is_finite(), "{context}: non-finite number");
    value as f32
}

fn parse_reference(path: &Path) -> Reference {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read FireRedVAD reference {}: {error}", path.display()));
    let root = vokra_core::json::parse(&bytes)
        .unwrap_or_else(|error| panic!("parse FireRedVAD reference {}: {error}", path.display()));
    let integer = |key: &str| -> usize {
        root.get(key)
            .unwrap_or_else(|| panic!("reference missing `{key}`"))
            .as_u64()
            .unwrap_or_else(|| panic!("reference `{key}` must be unsigned")) as usize
    };
    let rows = root
        .get("features")
        .and_then(vokra_core::json::JsonValue::as_array)
        .expect("reference `features` must be an array");
    let mut features = Vec::with_capacity(rows.len() * 80);
    for (frame, row) in rows.iter().enumerate() {
        let row = row
            .as_array()
            .unwrap_or_else(|| panic!("features[{frame}] must be an array"));
        assert_eq!(row.len(), 80, "features[{frame}] width");
        features.extend(
            row.iter()
                .enumerate()
                .map(|(bin, value)| number(value, &format!("features[{frame}][{bin}]"))),
        );
    }
    let probabilities = root
        .get("probabilities")
        .and_then(vokra_core::json::JsonValue::as_array)
        .expect("reference `probabilities` must be an array")
        .iter()
        .enumerate()
        .map(|(index, value)| number(value, &format!("probabilities[{index}]")))
        .collect::<Vec<_>>();
    let n_frames = integer("n_frames");
    assert_eq!(rows.len(), n_frames, "feature frame count");
    assert_eq!(probabilities.len(), n_frames, "probability frame count");
    Reference {
        sample_rate: u32::try_from(integer("sample_rate")).expect("sample_rate fits u32"),
        n_frames,
        features,
        probabilities,
    }
}

fn assert_probabilities(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: output length");
    let (index, max_abs) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (&actual, &expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("reference must contain at least one frame");
    eprintln!(
        "FireRedVAD {label}: frames={}, max_abs={max_abs:.9e} at frame={index}",
        actual.len()
    );
    assert!(
        max_abs <= PROB_ATOL,
        "{label}: max_abs={max_abs:.9e} exceeds fixed PROB_ATOL={PROB_ATOL:.9e} at frame {index}"
    );
}

fn read_pcm16_wav(path: &Path) -> (Vec<f32>, u32) {
    let bytes =
        std::fs::read(path).unwrap_or_else(|error| panic!("read WAV {}: {error}", path.display()));
    assert!(bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE");
    let mut offset = 12usize;
    let mut format = None;
    let mut data = None;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let start = offset + 8;
        let end = start.checked_add(size).expect("WAV chunk size overflow");
        assert!(end <= bytes.len(), "truncated WAV chunk");
        if id == b"fmt " {
            assert!(size >= 16, "short WAV fmt chunk");
            format = Some((
                u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap()),
                u16::from_le_bytes(bytes[start + 2..start + 4].try_into().unwrap()),
                u32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap()),
                u16::from_le_bytes(bytes[start + 14..start + 16].try_into().unwrap()),
            ));
        } else if id == b"data" {
            data = Some(&bytes[start..end]);
        }
        offset = end + (size & 1);
    }
    let (encoding, channels, sample_rate, bits) = format.expect("WAV missing fmt chunk");
    assert_eq!(
        (encoding, channels, bits),
        (1, 1, 16),
        "expected mono PCM16 WAV"
    );
    let data = data.expect("WAV missing data chunk");
    assert_eq!(data.len() % 2, 0, "PCM16 data length");
    let pcm = data
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32_768.0)
        .collect();
    (pcm, sample_rate)
}

#[test]
fn firered_vad_official_feature_and_pcm_parity() {
    let Some(gguf) = std::env::var_os(GGUF_ENV) else {
        eprintln!("skip: set {GGUF_ENV}, {REFERENCE_ENV}, and {WAV_ENV}");
        return;
    };
    let reference_path = std::env::var_os(REFERENCE_ENV)
        .unwrap_or_else(|| panic!("{REFERENCE_ENV} is required when {GGUF_ENV} is set"));
    let wav_path = std::env::var_os(WAV_ENV)
        .unwrap_or_else(|| panic!("{WAV_ENV} is required when {GGUF_ENV} is set"));
    let reference = parse_reference(Path::new(&reference_path));
    assert!(reference.n_frames > 0);
    let model = FireredVad::from_path(&gguf).expect("bind official FireRedVAD GGUF");
    let feature_probs = model
        .forward_features(&reference.features)
        .expect("native feature forward");
    assert_probabilities(&feature_probs, &reference.probabilities, "feature-forward");

    let (pcm, sample_rate) = read_pcm16_wav(Path::new(&wav_path));
    assert_eq!(sample_rate, reference.sample_rate);
    let pcm_probs = model
        .speech_probabilities(&pcm, sample_rate)
        .expect("native PCM forward");
    assert_probabilities(&pcm_probs, &reference.probabilities, "pcm-forward");

    let mut stream = model.open_stream();
    let mut streamed = Vec::new();
    for chunk in pcm.chunks(173) {
        streamed.extend(
            stream
                .push_pcm(chunk, sample_rate)
                .expect("native streamed PCM forward"),
        );
    }
    assert_probabilities(&streamed, &reference.probabilities, "streaming-pcm-forward");
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn firered_vad_official_cpu_metal_parity() {
    let Some(gguf) = std::env::var_os(GGUF_ENV) else {
        eprintln!("skip: set {GGUF_ENV}, {REFERENCE_ENV}, and {WAV_ENV}");
        return;
    };
    let reference_path = std::env::var_os(REFERENCE_ENV)
        .unwrap_or_else(|| panic!("{REFERENCE_ENV} is required when {GGUF_ENV} is set"));
    let wav_path = std::env::var_os(WAV_ENV)
        .unwrap_or_else(|| panic!("{WAV_ENV} is required when {GGUF_ENV} is set"));
    let reference = parse_reference(Path::new(&reference_path));
    let cpu = FireredVad::from_path(&gguf).expect("bind FireRedVAD CPU");
    let metal = FireredVad::from_path(&gguf)
        .expect("bind FireRedVAD Metal")
        .with_backend(vokra_core::BackendKind::Metal);

    let cpu_features = cpu.forward_features(&reference.features).unwrap();
    let metal_features = metal.forward_features(&reference.features).unwrap();
    assert_probabilities(
        &cpu_features,
        &reference.probabilities,
        "CPU feature-forward",
    );
    assert_probabilities(
        &metal_features,
        &reference.probabilities,
        "Metal feature-forward",
    );
    assert_probabilities(&metal_features, &cpu_features, "CPU/Metal feature-forward");

    let (pcm, sample_rate) = read_pcm16_wav(Path::new(&wav_path));
    let cpu_pcm = cpu.speech_probabilities(&pcm, sample_rate).unwrap();
    let metal_pcm = metal.speech_probabilities(&pcm, sample_rate).unwrap();
    assert_probabilities(&cpu_pcm, &reference.probabilities, "CPU PCM");
    assert_probabilities(&metal_pcm, &reference.probabilities, "Metal PCM");
    assert_probabilities(&metal_pcm, &cpu_pcm, "CPU/Metal PCM");
}
