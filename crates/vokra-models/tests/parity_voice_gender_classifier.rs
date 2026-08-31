//! Independent parity against JaesungHuh's official `model.ECAPA_gender`.

use std::path::Path;

use vokra_models::voice_gender_classifier::{CLASS_COUNT, VoiceGenderClassifier};

const FP32_PARITY_BOUND: f32 = 0.01;
// The PCM is the fixed synthetic tone emitted by the pinned independent
// upstream dumper, not caller voice data; its aggregate errors are safe to
// record as deterministic parity metrics.
const FIXTURE_KIND: &str = "official_canned_synthetic_tone";

fn f32s(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read parity fixture");
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn assert_finite_nonzero(values: &[f32], name: &str) {
    assert!(!values.is_empty(), "{name} reference is empty");
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{name} reference contains non-finite values"
    );
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!(
        norm.is_finite() && norm > 0.0,
        "{name} reference norm is invalid: {norm}"
    );
}

fn max_abs(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|value| value.is_finite()));
    assert!(expected.iter().all(|value| value.is_finite()));
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f32::max)
}

#[test]
fn real_voice_gender_classifier_matches_official_reference() {
    let Some(gguf) = std::env::var_os("VOKRA_VOICE_GENDER_GGUF") else {
        eprintln!("SKIP: VOKRA_VOICE_GENDER_GGUF is not set");
        return;
    };
    let Some(pcm) = std::env::var_os("VOKRA_VOICE_GENDER_PCM") else {
        eprintln!("SKIP: VOKRA_VOICE_GENDER_PCM is not set");
        return;
    };
    let Some(features) = std::env::var_os("VOKRA_VOICE_GENDER_FEATURES") else {
        eprintln!("SKIP: VOKRA_VOICE_GENDER_FEATURES is not set");
        return;
    };
    let Some(logits) = std::env::var_os("VOKRA_VOICE_GENDER_LOGITS") else {
        eprintln!("SKIP: VOKRA_VOICE_GENDER_LOGITS is not set");
        return;
    };
    let Some(embedding) = std::env::var_os("VOKRA_VOICE_GENDER_EMBEDDING") else {
        eprintln!("SKIP: VOKRA_VOICE_GENDER_EMBEDDING is not set");
        return;
    };
    let Some(probabilities) = std::env::var_os("VOKRA_VOICE_GENDER_PROBABILITIES") else {
        eprintln!("SKIP: VOKRA_VOICE_GENDER_PROBABILITIES is not set");
        return;
    };
    let Some(fixture_kind) = std::env::var_os("VOKRA_VOICE_GENDER_FIXTURE_KIND") else {
        eprintln!("SKIP: VOKRA_VOICE_GENDER_FIXTURE_KIND is not set");
        return;
    };
    assert_eq!(fixture_kind.to_string_lossy(), FIXTURE_KIND);

    let model = VoiceGenderClassifier::from_path(gguf).expect("strict dedicated bind");
    assert_eq!(model.weight_license(), vokra_core::LicenseClass::Permissive);
    let pcm = f32s(Path::new(&pcm));
    let expected_features = f32s(Path::new(&features));
    let expected_logits = f32s(Path::new(&logits));
    let expected_embedding = f32s(Path::new(&embedding));
    let expected_probabilities = f32s(Path::new(&probabilities));
    assert_finite_nonzero(&expected_features, "features");
    assert_finite_nonzero(&expected_embedding, "embedding");
    assert_finite_nonzero(&expected_logits, "logits");
    assert_finite_nonzero(&expected_probabilities, "probabilities");
    assert_eq!(expected_logits.len(), CLASS_COUNT);
    assert_eq!(
        expected_embedding.len(),
        vokra_models::voice_gender_classifier::EMBED_DIM
    );
    assert_eq!(expected_probabilities.len(), CLASS_COUNT);

    let (actual_features, frames) = model
        .frontend_features(&pcm, 16_000)
        .expect("official frontend");
    assert_eq!(actual_features.len(), expected_features.len());
    assert!(frames > 1);
    let feature_error = max_abs(&actual_features, &expected_features);
    assert!(feature_error.is_finite());

    let actual_logits = model
        .logits_features(&actual_features, frames)
        .expect("CPU classifier");
    let actual_embedding = model
        .embedding_features(&actual_features, frames)
        .expect("CPU embedding");
    let embedding_error = max_abs(&actual_embedding, &expected_embedding);
    assert!(embedding_error.is_finite());
    let logit_error = max_abs(&actual_logits, &expected_logits);
    assert!(logit_error.is_finite());
    let prediction = model.classify_pcm(&pcm, 16_000).expect("CPU prediction");
    let actual_probabilities = prediction.probabilities;
    let probability_error = max_abs(&actual_probabilities, &expected_probabilities);
    assert!(probability_error.is_finite());
    assert!(actual_probabilities.iter().all(|value| value.is_finite()));
    eprintln!(
        "VOICE_GENDER_OFFICIAL_PARITY_METRICS feature_max_abs={feature_error:.9} embedding_max_abs={embedding_error:.9} logits_max_abs={logit_error:.9} probability_max_abs={probability_error:.9} bound={FP32_PARITY_BOUND:.9} fixture={FIXTURE_KIND}"
    );
    assert!(
        feature_error <= FP32_PARITY_BOUND,
        "frontend feature max_abs {feature_error:.9} exceeds FP32 bound {FP32_PARITY_BOUND:.9}"
    );
    assert!(
        embedding_error <= FP32_PARITY_BOUND,
        "embedding max_abs {embedding_error:.9} exceeds FP32 bound {FP32_PARITY_BOUND:.9}"
    );
    assert!(
        logit_error <= FP32_PARITY_BOUND,
        "logit max_abs {logit_error:.9} exceeds FP32 bound {FP32_PARITY_BOUND:.9}"
    );
    assert!(
        probability_error <= FP32_PARITY_BOUND,
        "probability max_abs {probability_error:.9} exceeds FP32 bound {FP32_PARITY_BOUND:.9}"
    );
    let expected_label = if expected_probabilities[1] > expected_probabilities[0] {
        "female"
    } else {
        "male"
    };
    assert_eq!(prediction.label, expected_label);
    eprintln!(
        "VOICE_GENDER_OFFICIAL_PARITY PASS bound={FP32_PARITY_BOUND:.9} fixture={FIXTURE_KIND} oracle=official_upstream"
    );

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let metal = VoiceGenderClassifier::from_path(
            std::env::var_os("VOKRA_VOICE_GENDER_GGUF").expect("GGUF remains set"),
        )
        .expect("strict Metal bind")
        .with_backend(vokra_core::BackendKind::Metal);
        let metal_logits = metal.logits_pcm(&pcm, 16_000).expect("Metal classifier");
        let metal_error = max_abs(&metal_logits, &actual_logits);
        assert!(metal_error.is_finite());
        eprintln!(
            "VOICE_GENDER_METAL_VS_CPU_METRICS logits_max_abs={metal_error:.9} bound={FP32_PARITY_BOUND:.9} fixture={FIXTURE_KIND}"
        );
        assert!(
            metal_error <= FP32_PARITY_BOUND,
            "Metal-vs-CPU logit max_abs {metal_error:.9} exceeds FP32 bound {FP32_PARITY_BOUND:.9}"
        );
        let metal_prediction = metal.classify_pcm(&pcm, 16_000).expect("Metal prediction");
        assert_eq!(metal_prediction.label, expected_label);
        eprintln!(
            "VOICE_GENDER_METAL_VS_CPU PASS bound={FP32_PARITY_BOUND:.9} fixture={FIXTURE_KIND} label={expected_label}"
        );
    }
}
