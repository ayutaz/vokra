//! Independent parity against the pinned upstream WeSpeaker ResNet34-LM.

use vokra_models::wespeaker::{WeSpeaker, WeSpeakerArtifactLayout};

const PCM: &[u8] = include_bytes!("fixtures/wespeaker/pcm.f32.bin");
const FEATURES: &[u8] = include_bytes!("fixtures/wespeaker/features.f32.bin");
const EMBEDDING: &[u8] = include_bytes!("fixtures/wespeaker/embedding.f32.bin");
const MANIFEST: &str = include_str!("fixtures/wespeaker/manifest.json");

fn f32s(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[derive(Debug)]
struct Metrics {
    max_abs: f32,
    max_abs_index: usize,
    mean_abs: f32,
    relative_l1: f32,
    cosine: f32,
}

fn measure(label: &str, actual: &[f32], expected: &[f32]) -> Metrics {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{label} finite"
    );
    let (max_abs_index, max_abs) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap();
    let sum_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>();
    let expected_l1 = expected.iter().map(|value| value.abs()).sum::<f32>();
    let dot = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| actual * expected)
        .sum::<f32>();
    let actual_norm = actual.iter().map(|value| value * value).sum::<f32>().sqrt();
    let expected_norm = expected
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let metrics = Metrics {
        max_abs,
        max_abs_index,
        mean_abs: sum_abs / actual.len() as f32,
        relative_l1: sum_abs / expected_l1.max(1.0e-20),
        cosine: dot / (actual_norm * expected_norm).max(1.0e-20),
    };
    eprintln!(
        "WeSpeaker {label}: max_abs={:.9e} at {} (actual={:.9e}, reference={:.9e}), mean_abs={:.9e}, relative_l1={:.9e}, cosine={:.9e}",
        metrics.max_abs,
        metrics.max_abs_index,
        actual[metrics.max_abs_index],
        expected[metrics.max_abs_index],
        metrics.mean_abs,
        metrics.relative_l1,
        metrics.cosine,
    );
    metrics
}

#[test]
fn committed_reference_has_pinned_shapes_and_provenance() {
    assert_eq!(f32s(PCM).len(), 32_000);
    assert_eq!(f32s(FEATURES).len(), 198 * 80);
    assert_eq!(f32s(EMBEDDING).len(), 256);
    assert!(MANIFEST.contains("vokra-wespeaker-reference-v1"));
    assert!(MANIFEST.contains("f0c48c298fd835726c27956a5d617bad7115627e"));
    assert!(MANIFEST.contains("45941e7cba2c3ea99e232d02bedf617fc71b0dad"));
}

#[test]
fn public_pyannote_artifact_matches_upstream_wespeaker() {
    let Some(path) = std::env::var_os("VOKRA_WESPEAKER_GGUF") else {
        eprintln!(
            "[parity_wespeaker_real] SKIP: set VOKRA_WESPEAKER_GGUF to the public pyannote WeSpeaker GGUF"
        );
        return;
    };
    let model = WeSpeaker::from_path(&path).expect("strict public WeSpeaker bind");
    assert_eq!(
        model.artifact_layout(),
        WeSpeakerArtifactLayout::PyannotePrefixed
    );
    let pcm = f32s(PCM);
    let expected_features = f32s(FEATURES);
    let expected_embedding = f32s(EMBEDDING);
    let (features, frames) = model
        .frontend_features(&pcm, 16_000)
        .expect("CPU WeSpeaker fbank");
    assert_eq!(frames, 198);
    let feature_metrics = measure("CPU frontend vs upstream", &features, &expected_features);
    let embedding = model
        .embed_features(&features, frames)
        .expect("CPU WeSpeaker network");
    let embedding_metrics = measure("CPU embedding vs upstream", &embedding, &expected_embedding);
    let end_to_end = model.embed_pcm(&pcm, 16_000).expect("CPU end-to-end");
    let end_to_end_metrics = measure(
        "CPU end-to-end vs upstream",
        &end_to_end,
        &expected_embedding,
    );

    // VAST 48565792 measured frontend max-abs 1.581e-4 / relative-L1
    // 3.945e-6 and embedding max-abs 1.338e-6 / relative-L1 8.789e-6.
    // These gates leave less than 2.3x headroom while bounding both the worst
    // element and aggregate drift.
    assert!(feature_metrics.max_abs <= 3.5e-4);
    assert!(feature_metrics.relative_l1 <= 8.0e-6);
    assert!(feature_metrics.cosine >= 0.999_999_5);
    assert!(embedding_metrics.max_abs <= 3.0e-6);
    assert!(embedding_metrics.relative_l1 <= 2.0e-5);
    assert!(embedding_metrics.cosine >= 0.999_999);
    assert!(end_to_end_metrics.max_abs <= 3.0e-6);
    assert!(end_to_end_metrics.relative_l1 <= 2.0e-5);
    assert!(end_to_end_metrics.cosine >= 0.999_999);

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use vokra_core::BackendKind;

        let metal = WeSpeaker::from_path(path)
            .expect("strict WeSpeaker bind for Metal")
            .with_backend(BackendKind::Metal);
        let metal_embedding = metal
            .embed_features(&features, frames)
            .expect("Metal WeSpeaker network");
        let metal_metrics = measure("Metal embedding vs CPU", &metal_embedding, &embedding);
        assert!(metal_metrics.relative_l1 <= 1.0e-3);
        assert!(metal_metrics.cosine >= 0.999);
    }
}

#[test]
fn mislicensed_public_canonical_artifact_is_rejected() {
    let Some(path) = std::env::var_os("VOKRA_WESPEAKER_MISLICENSED_GGUF") else {
        eprintln!(
            "[parity_wespeaker_real] SKIP: set VOKRA_WESPEAKER_MISLICENSED_GGUF to audit the old canonical public file"
        );
        return;
    };
    let error = WeSpeaker::from_path(path).expect_err("Apache-stamped CC-BY checkpoint must fail");
    assert!(error.to_string().contains("vokra.provenance.license"));
    assert!(error.to_string().contains("cc-by-4.0"));
}
