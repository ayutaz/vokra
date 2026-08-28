//! Independent parity against TEN-VAD v1.0's official ONNX and public C ABI.
//!
//! `tools/parity/ten_vad_prepare_checkpoint.py` produces both references from
//! upstream commit `8e96899ba05a8e8c0e883ec7417e7a144bd9dec0`. The ONNX leg
//! isolates the neural graph; the C-ABI leg covers PCM, LPCNet-derived pitch,
//! log-mel context, recurrent state, and output probability end to end.

use std::path::Path;

use vokra_core::engines::VadEngine;
use vokra_models::ten_vad::TenVad;
use vokra_ops::ten_vad::{TenVadFrontend, TenVadNetworkState};

const GGUF_ENV: &str = "VOKRA_TEN_VAD_REAL_GGUF";
const REFERENCE_ENV: &str = "VOKRA_TEN_VAD_REFERENCE_JSON";
const NETWORK_ATOL: f32 = 1.0e-5;
// The released C ABI uses LPCNet's fixed-coefficient, f32 Ooura FFT. Vokra's
// independent mixed-radix FFT changes reduction order slightly; the graph-only
// gate above stays much tighter so this allowance cannot mask a weight error.
const FEATURE_ATOL: f32 = 3.0e-4;
const STREAM_ATOL: f32 = 1.0e-3;

fn number(value: &vokra_core::json::JsonValue, context: &str) -> f32 {
    let value = match value {
        vokra_core::json::JsonValue::Float(value) => *value,
        vokra_core::json::JsonValue::Int(value) => *value as f64,
        other => panic!("{context}: expected number, got {other:?}"),
    };
    assert!(value.is_finite(), "{context}: non-finite number");
    value as f32
}

fn array<'a>(
    value: &'a vokra_core::json::JsonValue,
    key: &str,
) -> &'a [vokra_core::json::JsonValue] {
    value
        .get(key)
        .and_then(vokra_core::json::JsonValue::as_array)
        .unwrap_or_else(|| panic!("reference `{key}` must be an array"))
}

fn numeric_array(value: &vokra_core::json::JsonValue, context: &str) -> Vec<f32> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{context} must be an array"))
        .iter()
        .enumerate()
        .map(|(index, value)| number(value, &format!("{context}[{index}]")))
        .collect()
}

fn max_abs(actual: &[f32], expected: &[f32], label: &str) -> f32 {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn official_network_and_stream_parity() {
    let Some(gguf) = std::env::var_os(GGUF_ENV) else {
        eprintln!("skip: set {GGUF_ENV} and {REFERENCE_ENV}");
        return;
    };
    let reference_path = std::env::var_os(REFERENCE_ENV)
        .unwrap_or_else(|| panic!("{REFERENCE_ENV} is required when {GGUF_ENV} is set"));
    let bytes = std::fs::read(Path::new(&reference_path)).expect("read TEN-VAD reference");
    let root = vokra_core::json::parse(&bytes).expect("parse TEN-VAD reference");
    let model = TenVad::open(Path::new(&gguf)).expect("bind canonical TEN-VAD GGUF");

    let network = root.get("network").expect("reference missing network");
    let feature_steps = array(network, "features");
    let expected_probabilities = numeric_array(
        network.get("probabilities").expect("network probabilities"),
        "network.probabilities",
    );
    let mut state = TenVadNetworkState::default();
    let mut actual_probabilities = Vec::new();
    for (step, context) in feature_steps.iter().enumerate() {
        let rows = context
            .as_array()
            .unwrap_or_else(|| panic!("network.features[{step}] must be an array"));
        assert_eq!(rows.len(), 3, "network feature context length");
        let mut features = Vec::with_capacity(123);
        for (row_index, row) in rows.iter().enumerate() {
            let values = numeric_array(row, &format!("network.features[{step}][{row_index}]"));
            assert_eq!(values.len(), 41, "network feature width");
            features.extend(values);
        }
        actual_probabilities.push(
            model
                .predict_features(&features, &mut state)
                .expect("native TEN-VAD network forward"),
        );
    }
    let network_error = max_abs(
        &actual_probabilities,
        &expected_probabilities,
        "network probabilities",
    );
    eprintln!("TEN-VAD official ONNX network max_abs={network_error:.9e}");
    assert!(
        network_error <= NETWORK_ATOL,
        "network max_abs={network_error:.9e} exceeds fixed NETWORK_ATOL={NETWORK_ATOL:.9e}"
    );

    let stream = root.get("stream").expect("reference missing stream");
    let pcm = numeric_array(stream.get("pcm_i16").expect("stream PCM"), "stream.pcm_i16")
        .into_iter()
        .map(|sample| sample / 32768.0)
        .collect::<Vec<_>>();
    let expected_stream = numeric_array(
        stream.get("probabilities").expect("stream probabilities"),
        "stream.probabilities",
    );
    let expected_features = array(stream, "features")
        .iter()
        .enumerate()
        .map(|(index, value)| numeric_array(value, &format!("stream.features[{index}]")))
        .collect::<Vec<_>>();
    let mut frontend = TenVadFrontend::new();
    let mut feature_error = 0.0f32;
    let mut feature_frame = 0usize;
    let mut feature_bin = 0usize;
    for (frame, (pcm_frame, expected)) in pcm.chunks_exact(256).zip(&expected_features).enumerate()
    {
        let actual = frontend
            .process_frame(pcm_frame)
            .expect("native TEN-VAD frontend");
        assert_eq!(expected.len(), 123, "official feature context width");
        for (bin, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            if error > feature_error {
                feature_error = error;
                feature_frame = frame;
                feature_bin = bin;
            }
        }
    }
    eprintln!(
        "TEN-VAD official frontend max_abs={feature_error:.9e} at frame={feature_frame} bin={feature_bin}"
    );
    assert!(
        feature_error <= FEATURE_ATOL,
        "frontend max_abs={feature_error:.9e} exceeds fixed FEATURE_ATOL={FEATURE_ATOL:.9e}"
    );
    let mut handle = model.open_stream();
    let mut actual_stream = Vec::new();
    for chunk in pcm.chunks(173) {
        actual_stream.extend(
            handle
                .push_pcm(chunk, 16_000)
                .expect("native TEN-VAD streamed forward"),
        );
    }
    let stream_error = max_abs(&actual_stream, &expected_stream, "stream probabilities");
    let (stream_index, _) = actual_stream
        .iter()
        .zip(&expected_stream)
        .enumerate()
        .map(|(index, (&actual, &expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("stream reference is non-empty");
    eprintln!(
        "TEN-VAD official C ABI stream max_abs={stream_error:.9e} at frame={stream_index} \
         rust={:.9e} official={:.9e}",
        actual_stream[stream_index], expected_stream[stream_index]
    );
    assert!(
        stream_error <= STREAM_ATOL,
        "stream max_abs={stream_error:.9e} exceeds fixed STREAM_ATOL={STREAM_ATOL:.9e}"
    );
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn official_network_and_stream_cpu_metal_parity() {
    let Some(gguf) = std::env::var_os(GGUF_ENV) else {
        eprintln!("skip: set {GGUF_ENV} and {REFERENCE_ENV}");
        return;
    };
    let reference_path = std::env::var_os(REFERENCE_ENV)
        .unwrap_or_else(|| panic!("{REFERENCE_ENV} is required when {GGUF_ENV} is set"));
    let bytes = std::fs::read(Path::new(&reference_path)).expect("read TEN-VAD reference");
    let root = vokra_core::json::parse(&bytes).expect("parse TEN-VAD reference");
    let network = root.get("network").expect("reference missing network");
    let feature_steps = array(network, "features");
    let expected = numeric_array(
        network.get("probabilities").expect("network probabilities"),
        "network.probabilities",
    );
    let cpu = TenVad::open(Path::new(&gguf)).expect("bind TEN-VAD CPU");
    let metal = TenVad::open(Path::new(&gguf))
        .expect("bind TEN-VAD Metal")
        .with_backend(vokra_core::BackendKind::Metal);
    let mut cpu_state = TenVadNetworkState::default();
    let mut metal_state = TenVadNetworkState::default();
    let mut cpu_probabilities = Vec::new();
    let mut metal_probabilities = Vec::new();
    for (step, context) in feature_steps.iter().enumerate() {
        let mut features = Vec::with_capacity(123);
        for (row_index, row) in context
            .as_array()
            .unwrap_or_else(|| panic!("network.features[{step}] must be an array"))
            .iter()
            .enumerate()
        {
            features.extend(numeric_array(
                row,
                &format!("network.features[{step}][{row_index}]"),
            ));
        }
        cpu_probabilities.push(cpu.predict_features(&features, &mut cpu_state).unwrap());
        metal_probabilities.push(metal.predict_features(&features, &mut metal_state).unwrap());
    }
    assert!(max_abs(&cpu_probabilities, &expected, "CPU network") <= NETWORK_ATOL);
    assert!(max_abs(&metal_probabilities, &expected, "Metal network") <= 0.01);
    assert!(
        max_abs(
            &metal_probabilities,
            &cpu_probabilities,
            "CPU/Metal network"
        ) <= 0.01
    );

    let stream = root.get("stream").expect("reference missing stream");
    let pcm = numeric_array(stream.get("pcm_i16").expect("stream PCM"), "stream.pcm_i16")
        .into_iter()
        .map(|sample| sample / 32768.0)
        .collect::<Vec<_>>();
    let expected_stream = numeric_array(
        stream.get("probabilities").expect("stream probabilities"),
        "stream.probabilities",
    );
    let mut cpu_handle = cpu.open_stream();
    let mut metal_handle = metal.open_stream();
    let cpu_stream = cpu_handle.push_pcm(&pcm, 16_000).unwrap();
    let metal_stream = metal_handle.push_pcm(&pcm, 16_000).unwrap();
    assert!(max_abs(&cpu_stream, &expected_stream, "CPU stream") <= STREAM_ATOL);
    assert!(max_abs(&metal_stream, &expected_stream, "Metal stream") <= 0.01);
    assert!(max_abs(&metal_stream, &cpu_stream, "CPU/Metal stream") <= 0.01);
}
