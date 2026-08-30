//! VAST-only real-weight parity for the authenticated GigaAM Multilingual CTC route.
//!
//! The reference directory is produced by the pinned official remote-code
//! dumper. This test is ignored by default and must never be interpreted as a
//! synthetic or self-mirror parity fixture.

use std::path::{Path, PathBuf};

use vokra_core::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_models::gigaam::GigaamMultilingual;

// These are the repository FP32 model-parity bounds. They are fixed before a
// VAST run; a failure requires diagnosis, not widening the bounds.
const ENCODED_MAX_ABS_BOUND: f32 = 1.0e-2;
const ENCODED_MEAN_ABS_BOUND: f32 = 1.0e-3;
const LOGITS_MAX_ABS_BOUND: f32 = 1.0e-2;
const LOGITS_MEAN_ABS_BOUND: f32 = 1.0e-3;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read f32 reference artifact");
    assert_eq!(
        bytes.len() % 4,
        0,
        "f32 artifact is truncated: {}",
        path.display()
    );
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte f32")))
        .collect::<Vec<_>>();
    assert!(
        !values.is_empty(),
        "f32 artifact is empty: {}",
        path.display()
    );
    assert!(
        values.iter().all(|value| value.is_finite()),
        "non-finite f32 artifact: {}",
        path.display()
    );
    assert!(
        values.iter().any(|value| *value != 0.0),
        "all-zero f32 artifact: {}",
        path.display()
    );
    values
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).expect("read u32 reference artifact");
    assert_eq!(
        bytes.len() % 4,
        0,
        "u32 artifact is truncated: {}",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("4-byte u32")))
        .collect()
}

fn compare(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    max_bound: f32,
    mean_bound: f32,
) -> (f32, f32) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let (index, max_abs) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (left, right))| (index, (left - right).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty parity vector");
    let mean_abs = actual
        .iter()
        .zip(expected)
        .map(|(left, right)| f64::from((*left - *right).abs()))
        .sum::<f64>()
        / actual.len() as f64;
    let mean_abs = mean_abs as f32;
    eprintln!(
        "GIGAAM_MULTILINGUAL_PARITY {label} max_abs={max_abs:.9e} index={index} mean_abs={mean_abs:.9e}"
    );
    assert!(
        max_abs <= max_bound,
        "{label} max_abs {max_abs} > {max_bound}"
    );
    assert!(
        mean_abs <= mean_bound,
        "{label} mean_abs {mean_abs} > {mean_bound}"
    );
    (max_abs, mean_abs)
}

#[test]
#[ignore = "real checkpoint parity runs only on a disposable VAST worker"]
fn real_gigaam_multilingual_trace_matches_official() {
    let gguf =
        PathBuf::from(std::env::var("GIGAAM_MULTILINGUAL_GGUF").expect("GIGAAM_MULTILINGUAL_GGUF"));
    let reference = PathBuf::from(
        std::env::var("GIGAAM_MULTILINGUAL_REFERENCE_DIR")
            .expect("GIGAAM_MULTILINGUAL_REFERENCE_DIR"),
    );
    let report = std::env::var_os("GIGAAM_MULTILINGUAL_PARITY_REPORT").map(PathBuf::from);
    if let Some(path) = &report {
        assert!(
            !path.exists() && !path.is_symlink(),
            "parity report must be absent"
        );
    }

    let file = GgufFile::open(&gguf).expect("open converted GigaAM GGUF");
    let backend = match std::env::var("GIGAAM_BACKEND").as_deref() {
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok("metal") => BackendKind::Metal,
        Ok(other) => panic!("unsupported GIGAAM_BACKEND={other:?}"),
    };
    let model = GigaamMultilingual::from_gguf(&file)
        .expect("strict GigaAM Multilingual bind")
        .with_backend(backend)
        .expect("authenticated GigaAM Multilingual backend preflight");
    let pcm = read_f32(&reference.join("pcm.f32le"));
    let expected_encoded = read_f32(&reference.join("encoded.f32le"));
    let expected_logits = read_f32(&reference.join("logits.f32le"));
    let expected_raw_argmax = read_u32(&reference.join("raw_argmax.u32le"));
    let expected_token_ids = read_u32(&reference.join("token_ids.u32le"));

    let trace = model.parity_trace_pcm(&pcm).expect("native GigaAM trace");
    assert_eq!(trace.encoded_frames, expected_raw_argmax.len());
    assert_eq!(trace.encoded.len(), trace.encoded_frames * 768);
    let (encoded_max_abs, encoded_mean_abs) = compare(
        "encoded",
        &trace.encoded,
        &expected_encoded,
        ENCODED_MAX_ABS_BOUND,
        ENCODED_MEAN_ABS_BOUND,
    );
    let (logits_max_abs, logits_mean_abs) = compare(
        "logits",
        &trace.logits,
        &expected_logits,
        LOGITS_MAX_ABS_BOUND,
        LOGITS_MEAN_ABS_BOUND,
    );
    assert_eq!(trace.raw_argmax, expected_raw_argmax, "raw CTC argmax IDs");
    assert_eq!(
        trace.token_ids, expected_token_ids,
        "collapsed CTC token IDs"
    );
    eprintln!("GIGAAM_MULTILINGUAL_PARITY token_ids=exact PASS");
    eprintln!("GIGAAM_MULTILINGUAL_PARITY backend={backend:?} PASS");

    if let Some(path) = report {
        let body = format!(
            "{{\n  \"format\": \"vokra-gigaam-multilingual-parity-v1\",\n  \"status\": \"PASS\",\n  \"encoded_max_abs\": {encoded_max_abs:.9e},\n  \"encoded_mean_abs\": {encoded_mean_abs:.9e},\n  \"logits_max_abs\": {logits_max_abs:.9e},\n  \"logits_mean_abs\": {logits_mean_abs:.9e},\n  \"raw_argmax\": \"EXACT\",\n  \"token_ids\": \"EXACT\"\n}}\n"
        );
        std::fs::write(path, body).expect("write parity report");
    }
}
