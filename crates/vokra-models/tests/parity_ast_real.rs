//! Gated real-public-GGUF parity against the pinned official Hugging Face AST.
//!
//! The independent fixture is produced by `tools/parity/ast/dump_reference.py`.
//! Set `VOKRA_AST_GGUF` to the audited public GGUF. `VOKRA_AST_BACKEND=metal`
//! selects Apple Metal; absent or `cpu` selects CPU. An unset GGUF is a
//! documented skip, never a fabricated pass.

use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_models::ast::{AstAudioSet, MAX_LENGTH, NUM_LABELS, SAMPLE_RATE, extract_features};
use vokra_models::silero_vad::wav::read_wav_f32;

const NUM_MELS: usize = 128;

// The initial 2e-5 max-only gate was intentionally registered before the first
// Vokra run. The official TorchAudio float32 frontend itself differs from the
// independent NumPy float64 Kaldi-equation cross-check by max 2.40257e-4 at a
// near-floor mel bin (RMSE 5.54365e-6). Keep a strict distribution gate while
// allowing that measured, numerically unavoidable tail.
const FEATURE_MAX_ABS: f32 = 5.0e-4;
const FEATURE_RMSE: f64 = 1.0e-5;
const FEATURE_P99: f32 = 2.0e-5;

// Pre-registered before the first Vokra logit execution.
const LOGIT_MAX_ABS: f32 = 1.0e-2;
const LOGIT_RMSE: f64 = 2.0e-3;
const LOGIT_COSINE_MIN: f64 = 0.999_99;

fn repo_fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures")
        .join(path)
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read AST fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn selected_backend() -> BackendKind {
    match std::env::var("VOKRA_AST_BACKEND").as_deref() {
        Ok("metal") => BackendKind::Metal,
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok(other) => panic!("VOKRA_AST_BACKEND must be cpu or metal, got {other:?}"),
    }
}

fn max_abs(actual: &[f32], expected: &[f32]) -> (usize, f32) {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty AST tensor")
}

fn top_indices(values: &[f32], count: usize) -> Vec<usize> {
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]));
    indices.truncate(count);
    indices
}

#[test]
fn official_reference_fixture_is_pinned() {
    let features = read_f32(&repo_fixture("ast/input_values.f32le"));
    let logits = read_f32(&repo_fixture("ast/logits.f32le"));
    assert_eq!(features.len(), MAX_LENGTH * NUM_MELS);
    assert_eq!(logits.len(), NUM_LABELS);
    let manifest = include_str!("../../../tests/fixtures/ast/manifest.json");
    for required in [
        "vokra.ast.official-parity.v1",
        "f826b80d28226b62986cc218e5cec390b1096902",
        "ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d",
        "e951795c5c1787b0552ad7dd67f0dcee5bc73219cb585dad61e5a0e380f96602",
        "dd03ffc605ee8d9230eeaa7fc30b4db5bee555dd385cbc9bd373496f7a6a4eed",
    ] {
        assert!(
            manifest.contains(required),
            "AST manifest lost pin {required}"
        );
    }
}

#[test]
fn real_ast_frontend_and_logits_match_official() {
    let Some(gguf_path) = std::env::var_os("VOKRA_AST_GGUF") else {
        eprintln!(
            "[parity_ast_real] SKIP: set VOKRA_AST_GGUF to vokra/ast-finetuned-audioset/ast.gguf"
        );
        return;
    };
    let expected_features = read_f32(&repo_fixture("ast/input_values.f32le"));
    let expected_logits = read_f32(&repo_fixture("ast/logits.f32le"));
    assert_eq!(expected_features.len(), MAX_LENGTH * NUM_MELS);
    assert_eq!(expected_logits.len(), NUM_LABELS);

    let wav = read_wav_f32(repo_fixture("audio/jfk-30s.wav")).expect("read pinned AST WAV");
    assert_eq!(wav.sample_rate, SAMPLE_RATE);
    let actual_features = extract_features(&wav.samples, wav.sample_rate)
        .expect("extract native AST frontend features");
    if let Some(path) = std::env::var_os("VOKRA_AST_DUMP_FEATURES") {
        let bytes: Vec<u8> = actual_features
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(&path, bytes).unwrap_or_else(|error| {
            panic!(
                "write AST diagnostic features {}: {error}",
                Path::new(&path).display()
            )
        });
    }
    let (feature_index, feature_delta) = max_abs(&actual_features, &expected_features);
    let mut feature_deltas: Vec<f32> = actual_features
        .iter()
        .zip(&expected_features)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .collect();
    let feature_rmse = (feature_deltas
        .iter()
        .map(|&delta| f64::from(delta).powi(2))
        .sum::<f64>()
        / actual_features.len() as f64)
        .sqrt();
    feature_deltas.sort_unstable_by(f32::total_cmp);
    let p99_index = (feature_deltas.len() * 99).div_ceil(100) - 1;
    let feature_p99 = feature_deltas[p99_index];
    eprintln!(
        "[parity_ast_real] frontend max_abs={feature_delta:.9e} at {feature_index} \
         (actual={:.9e}, reference={:.9e}), rmse={feature_rmse:.9e}, \
         p99={feature_p99:.9e}",
        actual_features[feature_index], expected_features[feature_index]
    );
    assert!(
        feature_delta <= FEATURE_MAX_ABS,
        "AST frontend max_abs {feature_delta:.9e} exceeds {FEATURE_MAX_ABS:.9e}"
    );
    assert!(
        feature_rmse <= FEATURE_RMSE,
        "AST frontend RMSE {feature_rmse:.9e} exceeds {FEATURE_RMSE:.9e}"
    );
    assert!(
        feature_p99 <= FEATURE_P99,
        "AST frontend p99 {feature_p99:.9e} exceeds {FEATURE_P99:.9e}"
    );

    let backend = selected_backend();
    let model = AstAudioSet::open(&gguf_path)
        .expect("strict public AST bind")
        .with_backend(backend);
    let actual_logits = model
        .classify_pcm(&wav.samples, wav.sample_rate)
        .expect("native AST classify");
    let (logit_index, logit_delta) = max_abs(&actual_logits, &expected_logits);
    let mut squared_error = 0.0f64;
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    for (&actual, &expected) in actual_logits.iter().zip(&expected_logits) {
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        squared_error += (actual - expected).powi(2);
        dot += actual * expected;
        actual_norm += actual * actual;
        expected_norm += expected * expected;
    }
    let rmse = (squared_error / actual_logits.len() as f64).sqrt();
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
    let actual_top5 = top_indices(&actual_logits, 5);
    let expected_top5 = top_indices(&expected_logits, 5);
    eprintln!(
        "[parity_ast_real] {backend:?}: max_abs={logit_delta:.9e} at {logit_index}, \
         rmse={rmse:.9e}, cosine={cosine:.12}, top5={actual_top5:?}"
    );
    assert!(
        logit_delta <= LOGIT_MAX_ABS,
        "AST {backend:?} logit max_abs {logit_delta:.9e} exceeds {LOGIT_MAX_ABS:.9e}"
    );
    assert!(
        rmse <= LOGIT_RMSE,
        "AST {backend:?} logit RMSE {rmse:.9e} exceeds {LOGIT_RMSE:.9e}"
    );
    assert!(
        cosine >= LOGIT_COSINE_MIN,
        "AST {backend:?} logit cosine {cosine:.12} is below {LOGIT_COSINE_MIN:.12}"
    );
    assert_eq!(actual_top5, expected_top5, "AST {backend:?} top-5 order");
}
