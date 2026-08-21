//! Env-gated real-weight parity against pinned official FunASR execution.

use vokra_core::engines::VadEngine;
use vokra_core::json::JsonValue;
use vokra_models::fsmn_vad::FsmnVadV1;

const GGUF_ENV: &str = "VOKRA_FSMN_VAD_REAL_GGUF";
const FIXTURE: &[u8] = include_bytes!("../../../tools/parity/fixtures/fsmn_vad_real.json");
// Calibrated on the pinned VAST reference run: full-posterior max_abs was
// 8.344650269e-7 and both PCM paths were 1.370906830e-6.  These tight fixed
// bounds leave only a small cross-ISA reduction-order margin.
const NETWORK_ATOL: f32 = 2.0e-6;
const PCM_ATOL: f32 = 5.0e-6;

struct Reference {
    sample_rate: u32,
    pcm: Vec<f32>,
    frames: usize,
    feature_width: usize,
    output_width: usize,
    features: Vec<f32>,
    probabilities: Vec<f32>,
    scores: Vec<f32>,
}

fn number(value: &JsonValue, label: &str) -> f32 {
    let value = match value {
        JsonValue::Int(value) => *value as f64,
        JsonValue::Float(value) => *value,
        other => panic!("{label}: expected number, got {other:?}"),
    };
    assert!(value.is_finite(), "{label}: non-finite number");
    value as f32
}

fn integer(root: &JsonValue, key: &str) -> usize {
    root.get(key)
        .unwrap_or_else(|| panic!("fixture missing `{key}`"))
        .as_u64()
        .unwrap_or_else(|| panic!("fixture `{key}` must be unsigned")) as usize
}

fn matrix(root: &JsonValue, key: &str, rows: usize, width: usize) -> Vec<f32> {
    let matrix = root
        .get(key)
        .and_then(JsonValue::as_array)
        .unwrap_or_else(|| panic!("fixture `{key}` must be an array"));
    assert_eq!(matrix.len(), rows, "fixture `{key}` row count");
    let mut output = Vec::with_capacity(rows * width);
    for (row_index, row) in matrix.iter().enumerate() {
        let row = row
            .as_array()
            .unwrap_or_else(|| panic!("fixture `{key}[{row_index}]` must be an array"));
        assert_eq!(row.len(), width, "fixture `{key}[{row_index}]` width");
        output.extend(
            row.iter()
                .enumerate()
                .map(|(column, value)| number(value, &format!("{key}[{row_index}][{column}]"))),
        );
    }
    output
}

fn parse_reference() -> Reference {
    let root = vokra_core::json::parse(FIXTURE).expect("parse committed FSMN-VAD fixture");
    let provenance = root.get("provenance").expect("fixture provenance");
    assert_eq!(
        provenance
            .get("funasr_revision")
            .and_then(JsonValue::as_str),
        Some("3c58cb56a56598232c3efffa15d313d7e82a4307")
    );
    assert_eq!(
        provenance.get("model_revision").and_then(JsonValue::as_str),
        Some("df20e6b30c653645fa4ff125cacfcabd1020a669")
    );
    assert_eq!(
        provenance.get("stream_final"),
        Some(&JsonValue::Bool(false))
    );
    let frames = integer(&root, "n_frames");
    let feature_width = integer(&root, "feature_width");
    let output_width = integer(&root, "output_width");
    let pcm = root
        .get("pcm_i16")
        .and_then(JsonValue::as_array)
        .expect("fixture pcm_i16")
        .iter()
        .enumerate()
        .map(|(index, value)| number(value, &format!("pcm_i16[{index}]")) / 32768.0)
        .collect::<Vec<_>>();
    let scores = root
        .get("speech_scores")
        .and_then(JsonValue::as_array)
        .expect("fixture speech_scores")
        .iter()
        .enumerate()
        .map(|(index, value)| number(value, &format!("speech_scores[{index}]")))
        .collect::<Vec<_>>();
    assert_eq!(scores.len(), frames);
    Reference {
        sample_rate: u32::try_from(integer(&root, "sample_rate")).unwrap(),
        pcm,
        frames,
        feature_width,
        output_width,
        features: matrix(&root, "features", frames, feature_width),
        probabilities: matrix(&root, "probabilities", frames, output_width),
        scores,
    }
}

fn max_abs(actual: &[f32], expected: &[f32]) -> (usize, f32) {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (&actual, &expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty parity vector")
}

#[test]
fn fsmn_vad_official_network_and_pcm_parity() {
    let Some(gguf) = std::env::var_os(GGUF_ENV) else {
        eprintln!("skip: set {GGUF_ENV} to the canonical official-weight GGUF");
        return;
    };
    let reference = parse_reference();
    assert_eq!(
        (
            reference.frames,
            reference.feature_width,
            reference.output_width
        ),
        (96, 400, 248)
    );
    let model = FsmnVadV1::open(gguf).expect("bind canonical FSMN-VAD GGUF");

    let probabilities = model
        .forward_features(&reference.features)
        .expect("official-feature forward");
    let (network_index, network_max) = max_abs(&probabilities, &reference.probabilities);

    let mut one_shot = model.open_stream();
    let scores = one_shot
        .push_pcm(&reference.pcm, reference.sample_rate)
        .expect("one-shot PCM forward");
    let (pcm_index, pcm_max) = max_abs(&scores, &reference.scores);

    let mut streaming = model.open_stream();
    let mut streamed = Vec::new();
    for chunk in reference.pcm.chunks(173) {
        streamed.extend(
            streaming
                .push_pcm(chunk, reference.sample_rate)
                .expect("streamed PCM forward"),
        );
    }
    let (stream_index, stream_max) = max_abs(&streamed, &reference.scores);
    eprintln!(
        "FSMN-VAD measured parity: network_max={network_max:.9e}@{network_index}, \
         pcm_max={pcm_max:.9e}@{pcm_index}, stream_max={stream_max:.9e}@{stream_index}"
    );
    assert!(
        network_max <= NETWORK_ATOL,
        "network max {network_max:.9e} exceeds {NETWORK_ATOL:.9e} at {network_index}"
    );
    assert!(
        pcm_max <= PCM_ATOL,
        "PCM max {pcm_max:.9e} exceeds {PCM_ATOL:.9e} at {pcm_index}"
    );
    assert!(
        stream_max <= PCM_ATOL,
        "stream max {stream_max:.9e} exceeds {PCM_ATOL:.9e} at {stream_index}"
    );
}
