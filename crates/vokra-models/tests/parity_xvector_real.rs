//! Independent parity against SpeechBrain 1.0.3 X-vector.

use vokra_core::gguf::GgufFile;
use vokra_models::xvector::{XVector, XVectorArtifactLayout};

const PCM: &[u8] = include_bytes!("fixtures/xvector/pcm.f32.bin");
const FEATURES: &[u8] = include_bytes!("fixtures/xvector/features.f32.bin");
const EMBEDDING: &[u8] = include_bytes!("fixtures/xvector/embedding.f32.bin");
const MANIFEST: &str = include_str!("fixtures/xvector/manifest.json");

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
        "XVector {label}: max_abs={:.9e} at {} (actual={:.9e}, reference={:.9e}), mean_abs={:.9e}, relative_l1={:.9e}, cosine={:.9e}",
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
    assert_eq!(f32s(PCM).len(), 16_000);
    assert_eq!(f32s(FEATURES).len(), 101 * 24);
    assert_eq!(f32s(EMBEDDING).len(), 512);
    assert!(MANIFEST.contains("vokra-xvector-reference-v1"));
    assert!(MANIFEST.contains("56895a2df401be4150a159f3a1c653f00051d477"));
    assert!(MANIFEST.contains("speechbrain\": \"1.0.3"));
}

fn parity_for_path(path: std::ffi::OsString, expected_layout: XVectorArtifactLayout) {
    let file = GgufFile::open(path).expect("open public X-vector GGUF");
    let model = XVector::from_gguf(&file).expect("strict X-vector bind");
    assert_eq!(model.artifact_layout(), expected_layout);
    assert_eq!(
        model.tensor_count(),
        match expected_layout {
            XVectorArtifactLayout::EmbeddingOnlyBare => 32,
            XVectorArtifactLayout::CombinedPrefixed => 46,
        }
    );

    let pcm = f32s(PCM);
    let expected_features = f32s(FEATURES);
    let expected_embedding = f32s(EMBEDDING);
    let (features, frames) = model
        .frontend_features(&pcm, 16_000)
        .expect("CPU SpeechBrain fbank");
    assert_eq!(frames, 101);
    let feature_metrics = measure("CPU frontend vs SpeechBrain", &features, &expected_features);
    let embedding = model
        .embed_features(&features, frames)
        .expect("CPU X-vector network");
    let embedding_metrics = measure(
        "CPU embedding vs SpeechBrain",
        &embedding,
        &expected_embedding,
    );
    let end_to_end = model.embed_pcm(&pcm, 16_000).expect("CPU end-to-end");
    let end_to_end_metrics = measure(
        "CPU end-to-end vs SpeechBrain",
        &end_to_end,
        &expected_embedding,
    );

    // VAST 48553051 measured both public layouts at frontend relative-L1
    // 5.481e-6 and embedding relative-L1 8.009e-5 / cosine 0.999999821.
    // These gates leave less than 2x headroom and also cap the worst element.
    assert!(feature_metrics.max_abs <= 1.5e-3);
    assert!(feature_metrics.relative_l1 <= 1.0e-5);
    assert!(embedding_metrics.max_abs <= 4.0e-3);
    assert!(embedding_metrics.relative_l1 <= 1.5e-4);
    assert!(embedding_metrics.cosine >= 0.999_999_5);
    assert!(end_to_end_metrics.max_abs <= 4.0e-3);
    assert!(end_to_end_metrics.relative_l1 <= 1.5e-4);
    assert!(end_to_end_metrics.cosine >= 0.999_999_5);

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use vokra_core::BackendKind;

        let metal = XVector::from_gguf(&file)
            .expect("strict X-vector bind for Metal")
            .with_backend(BackendKind::Metal);
        let metal_embedding = metal
            .embed_features(&features, frames)
            .expect("Metal X-vector network");
        let metal_metrics = measure("Metal embedding vs CPU", &metal_embedding, &embedding);
        assert!(metal_metrics.relative_l1 <= 1.0e-4);
        assert!(metal_metrics.cosine >= 0.999_99);
    }
}

#[test]
fn public_embedding_only_artifact_matches_speechbrain() {
    let Some(path) = std::env::var_os("VOKRA_XVECTOR_GGUF") else {
        eprintln!(
            "[parity_xvector_real] SKIP: set VOKRA_XVECTOR_GGUF to the public 32-tensor X-vector GGUF"
        );
        return;
    };
    parity_for_path(path, XVectorArtifactLayout::EmbeddingOnlyBare);
}

#[test]
fn public_combined_artifact_matches_speechbrain() {
    let Some(path) = std::env::var_os("VOKRA_XVECTOR_COMBINED_GGUF") else {
        eprintln!(
            "[parity_xvector_real] SKIP: set VOKRA_XVECTOR_COMBINED_GGUF to the public 46-tensor X-vector GGUF"
        );
        return;
    };
    parity_for_path(path, XVectorArtifactLayout::CombinedPrefixed);
}
