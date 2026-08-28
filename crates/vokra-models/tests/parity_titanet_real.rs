//! Independent parity against NVIDIA NeMo TitaNet-L.

use vokra_models::titanet::TitaNet;

const PCM: &[u8] = include_bytes!("fixtures/titanet/pcm.f32.bin");
const FEATURES: &[u8] = include_bytes!("fixtures/titanet/features.f32.bin");
const EMBEDDING: &[u8] = include_bytes!("fixtures/titanet/embedding.f32.bin");
const MANIFEST: &str = include_str!("fixtures/titanet/manifest.json");

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
        "TitaNet {label}: max_abs={:.9e} at {} (actual={:.9e}, reference={:.9e}), mean_abs={:.9e}, relative_l1={:.9e}, cosine={:.9e}",
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
    assert_eq!(f32s(PCM).len(), 8_173);
    assert_eq!(f32s(FEATURES).len(), 52 * 80);
    assert_eq!(f32s(EMBEDDING).len(), 192);
    assert!(MANIFEST.contains("vokra-titanet-reference-v1"));
    assert!(MANIFEST.contains("0dc382f40121a5fbd34db10a2bb04d826c2be6a8"));
    assert!(MANIFEST.contains("082c5ae26168796d3ebac6adcf54bb8b5354daa1"));
    assert!(MANIFEST.contains("e838520693f269e7984f55bc8eb3c2d60ccf246bf4b896d4be9bcabe3e4b0fe3"));
}

#[test]
fn public_artifact_matches_nemo() {
    let Some(path) = std::env::var_os("VOKRA_TITANET_GGUF") else {
        eprintln!(
            "[parity_titanet_real] SKIP: set VOKRA_TITANET_GGUF to a public canonical TitaNet-L GGUF"
        );
        return;
    };
    let model = TitaNet::from_path(path).expect("strict public TitaNet-L bind");
    let pcm = f32s(PCM);
    let expected_features = f32s(FEATURES);
    let expected_embedding = f32s(EMBEDDING);
    let (features, frames) = model
        .frontend_features(&pcm, 16_000)
        .expect("CPU NeMo filterbank");
    assert_eq!(frames, 52);
    let feature_metrics = measure("CPU frontend vs NeMo", &features, &expected_features);
    let embedding = model
        .embed_features(&features, frames)
        .expect("CPU TitaNet-L network");
    let embedding_metrics = measure("CPU embedding vs NeMo", &embedding, &expected_embedding);
    let end_to_end = model.embed_pcm(&pcm, 16_000).expect("CPU end-to-end");
    let end_to_end_metrics = measure("CPU end-to-end vs NeMo", &end_to_end, &expected_embedding);

    // VAST 48569784 measured frontend max_abs 1.907e-5 / relative-L1
    // 3.125e-6 and embedding max_abs 7.339e-7 / relative-L1 8.252e-6.
    // These gates leave less than 2x headroom and keep aggregate,
    // worst-element, and direction checks together.
    assert!(feature_metrics.max_abs <= 4.0e-5);
    assert!(feature_metrics.relative_l1 <= 6.0e-6);
    assert!(feature_metrics.cosine >= 0.999_999_5);
    assert!(embedding_metrics.max_abs <= 2.0e-6);
    assert!(embedding_metrics.relative_l1 <= 1.6e-5);
    assert!(embedding_metrics.cosine >= 0.999_999_5);
    assert!(end_to_end_metrics.max_abs <= 2.0e-6);
    assert!(end_to_end_metrics.relative_l1 <= 1.6e-5);
    assert!(end_to_end_metrics.cosine >= 0.999_999_5);

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use vokra_core::BackendKind;

        let metal = TitaNet::from_path(
            std::env::var_os("VOKRA_TITANET_GGUF").expect("path remains set for Metal"),
        )
        .expect("strict TitaNet-L bind for Metal")
        .with_backend(BackendKind::Metal);
        let metal_embedding = metal
            .embed_features(&features, frames)
            .expect("Metal TitaNet-L network");
        let metal_metrics = measure("Metal embedding vs CPU", &metal_embedding, &embedding);
        assert!(metal_metrics.relative_l1 <= 1.0e-3);
        assert!(metal_metrics.cosine >= 0.999);
    }
}
