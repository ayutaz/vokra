//! Independent parity against JaesungHuh's official `model.ECAPA_gender`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(all(feature = "metal", target_os = "macos"))]
use vokra_core::BackendKind;
use vokra_models::voice_gender_classifier::{CLASS_COUNT, VoiceGenderClassifier};

const FP32_PARITY_BOUND: f32 = 0.01;
const GGUF_SHA256: &str = "afb03696d8a640d5d701ea0c136bb065cac648cbfe905a5dcc4eae04e0769b1a";
const GGUF_SHA_ENV: &str = "VOKRA_VOICE_GENDER_GGUF_SHA256";
const REFERENCE_SHA_ENV: &str = "VOKRA_VOICE_GENDER_REFERENCE_MANIFEST_SHA256";
const REFERENCE_ENV: &str = "VOKRA_VOICE_GENDER_REFERENCE_DIR";
const GGUF_ENV: &str = "VOKRA_VOICE_GENDER_GGUF";
const EVIDENCE_ENV: &str = "VOKRA_VOICE_GENDER_EVIDENCE_DIR";
const ARGMAX_ENV: &str = "VOKRA_VOICE_GENDER_ARGMAX";
const REMOTE_ENV: &str = "VOKRA_REMOTE_APPLE_SILICON";
const VAST_ENV: &str = "VOKRA_REMOTE_VAST";
// The PCM is the fixed synthetic tone emitted by the pinned independent
// upstream dumper, not caller voice data; its aggregate errors are safe to
// record as deterministic parity metrics.
const FIXTURE_KIND: &str = "official_canned_synthetic_tone";

fn env_path(name: &str) -> PathBuf {
    let raw = std::env::var_os(name).unwrap_or_else(|| panic!("{name} is required"));
    let raw = raw
        .to_str()
        .unwrap_or_else(|| panic!("{name} must be UTF-8"));
    assert!(
        !raw.is_empty() && raw.starts_with('/'),
        "{name} must be absolute"
    );
    assert!(
        !raw.split('/').any(|part| matches!(part, "." | "..")),
        "{name} contains a lexical dot component"
    );
    let path = PathBuf::from(raw);
    for ancestor in path.ancestors() {
        if let Ok(metadata) = fs::symlink_metadata(ancestor) {
            assert!(
                !metadata.file_type().is_symlink(),
                "{name} has symlink ancestry"
            );
        }
    }
    path
}

fn required_file(path: &Path, label: &str) {
    let metadata = fs::symlink_metadata(path).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert!(metadata.file_type().is_file() && !metadata.file_type().is_symlink());
}

fn file_sha256(path: &Path) -> String {
    let output = Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .expect("shasum");
    assert!(output.status.success(), "shasum failed");
    String::from_utf8(output.stdout)
        .expect("shasum output")
        .split_whitespace()
        .next()
        .expect("shasum digest")
        .to_owned()
}

fn require_disjoint(paths: &[(&Path, &str)]) {
    let canonical = paths
        .iter()
        .map(|(path, label)| {
            let scope = if path.exists() {
                path.canonicalize()
                    .unwrap_or_else(|e| panic!("{label}: {e}"))
            } else {
                let parent = path
                    .parent()
                    .unwrap_or_else(|| panic!("{label} has no parent"));
                assert!(
                    parent.is_dir() && !parent.is_symlink(),
                    "{label} parent must exist"
                );
                parent
                    .canonicalize()
                    .unwrap_or_else(|e| panic!("{label} parent: {e}"))
                    .join(path.file_name().unwrap())
            };
            (scope, *label)
        })
        .collect::<Vec<_>>();
    for (index, (left, left_label)) in canonical.iter().enumerate() {
        for (right, right_label) in canonical.iter().skip(index + 1) {
            assert!(
                !(left == right || left.starts_with(right) || right.starts_with(left)),
                "{left_label} and {right_label} overlap"
            );
        }
    }
}

fn f32s(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read parity fixture");
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn argmax(path: &Path) -> usize {
    let bytes = fs::read(path).expect("read reference argmax");
    assert_eq!(bytes.len(), 4, "reference argmax must be one u32");
    let value = u32::from_le_bytes(bytes.try_into().unwrap());
    assert!(
        value < CLASS_COUNT as u32,
        "reference argmax is out of range"
    );
    value as usize
}

fn argmax_values(values: &[f32]) -> usize {
    assert!(!values.is_empty(), "argmax requires non-empty values");
    let mut best = 0;
    for (index, value) in values.iter().enumerate().skip(1) {
        if value > &values[best] {
            best = index;
        }
    }
    best
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
#[ignore]
fn real_voice_gender_classifier_matches_official_reference() {
    if cfg!(target_os = "macos") {
        assert_eq!(
            std::env::consts::ARCH,
            "aarch64",
            "Apple worker requires arm64"
        );
        assert_eq!(std::env::var(REMOTE_ENV).as_deref(), Ok("1"));
    } else {
        assert_eq!(std::env::consts::OS, "linux", "VAST worker requires Linux");
        assert_eq!(
            std::env::consts::ARCH,
            "x86_64",
            "VAST worker requires x86_64"
        );
        assert_eq!(std::env::var(VAST_ENV).as_deref(), Ok("1"));
        assert_eq!(
            std::env::var("VOKRA_PUBLISH_ON_VAST").as_deref(),
            Ok("1"),
            "VAST worker requires the explicit publish-on-VAST marker"
        );
    }
    let gguf = env_path(GGUF_ENV);
    required_file(&gguf, "GGUF");
    assert_eq!(std::env::var(GGUF_SHA_ENV).as_deref(), Ok(GGUF_SHA256));
    assert_eq!(file_sha256(&gguf), GGUF_SHA256, "corrected GGUF digest");
    let reference = env_path(REFERENCE_ENV);
    assert!(reference.is_dir() && !reference.is_symlink());
    let reference_sha =
        std::env::var(REFERENCE_SHA_ENV).expect("reference manifest SHA is required");
    assert!(
        reference_sha.len() == 64
            && reference_sha
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    );
    assert_eq!(file_sha256(&reference.join("meta.json")), reference_sha);
    let expected_files = [
        "pcm.f32",
        "features.f32",
        "embedding.f32",
        "logits.f32",
        "probabilities.f32",
        "argmax.u32",
        "meta.json",
    ];
    let actual_files = fs::read_dir(&reference)
        .expect("read reference directory")
        .map(|entry| entry.expect("reference directory entry").file_name())
        .collect::<BTreeSet<_>>();
    let expected_files = expected_files
        .into_iter()
        .map(std::ffi::OsString::from)
        .collect::<BTreeSet<_>>();
    assert_eq!(actual_files, expected_files, "reference file set drifted");
    for name in [
        "pcm.f32",
        "features.f32",
        "embedding.f32",
        "logits.f32",
        "probabilities.f32",
        "argmax.u32",
        "meta.json",
    ] {
        required_file(&reference.join(name), "reference artifact");
    }
    let evidence = env_path(EVIDENCE_ENV);
    assert!(
        evidence.is_dir() && !evidence.is_symlink(),
        "worker evidence directory must be claimed"
    );
    require_disjoint(&[
        (&gguf, "GGUF"),
        (&reference, "reference"),
        (&evidence, "evidence"),
        (
            &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
            "checkout",
        ),
    ]);
    let pcm = env_path("VOKRA_VOICE_GENDER_PCM");
    let features = env_path("VOKRA_VOICE_GENDER_FEATURES");
    let logits = env_path("VOKRA_VOICE_GENDER_LOGITS");
    let embedding = env_path("VOKRA_VOICE_GENDER_EMBEDDING");
    let probabilities = env_path("VOKRA_VOICE_GENDER_PROBABILITIES");
    let argmax_path = env_path(ARGMAX_ENV);
    let fixture_kind =
        std::env::var_os("VOKRA_VOICE_GENDER_FIXTURE_KIND").expect("fixture kind is required");
    assert_eq!(fixture_kind.to_string_lossy(), FIXTURE_KIND);
    for (path, label) in [
        (&pcm, "PCM"),
        (&features, "features"),
        (&logits, "logits"),
        (&embedding, "embedding"),
        (&probabilities, "probabilities"),
        (&argmax_path, "argmax"),
    ] {
        required_file(path, label);
    }
    assert_eq!(
        pcm,
        reference.join("pcm.f32"),
        "PCM path must be packet member"
    );
    assert_eq!(
        features,
        reference.join("features.f32"),
        "features path must be packet member"
    );
    assert_eq!(
        logits,
        reference.join("logits.f32"),
        "logits path must be packet member"
    );
    assert_eq!(
        embedding,
        reference.join("embedding.f32"),
        "embedding path must be packet member"
    );
    assert_eq!(
        probabilities,
        reference.join("probabilities.f32"),
        "probabilities path must be packet member"
    );
    assert_eq!(
        argmax_path,
        reference.join("argmax.u32"),
        "argmax path must be packet member"
    );

    let model = VoiceGenderClassifier::from_path(&gguf).expect("strict dedicated bind");
    assert_eq!(model.weight_license(), vokra_core::LicenseClass::Permissive);
    let pcm = f32s(&pcm);
    assert_eq!(pcm.len(), 32_000, "reference PCM shape drifted");
    let expected_features = f32s(&features);
    let expected_logits = f32s(&logits);
    let expected_embedding = f32s(&embedding);
    let expected_probabilities = f32s(&probabilities);
    let expected_argmax = argmax(&argmax_path);
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
    assert_eq!(expected_features.len() % 80, 0);
    assert_eq!(expected_argmax, argmax_values(&expected_probabilities));

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
    // libtest may print its `test ...` preamble without a trailing newline
    // before replaying captured stderr; keep the canonical marker line-based.
    eprintln!();
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
    let expected_label = vokra_models::voice_gender_classifier::CLASS_LABELS[expected_argmax];
    assert_eq!(argmax_values(&actual_probabilities), expected_argmax);
    assert_eq!(prediction.label, expected_label);
    eprintln!(
        "VOICE_GENDER_OFFICIAL_PARITY PASS bound={FP32_PARITY_BOUND:.9} fixture={FIXTURE_KIND} oracle=official_upstream"
    );

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let metal = VoiceGenderClassifier::from_path(&gguf)
            .expect("strict Metal bind")
            .with_backend(BackendKind::Metal);
        assert_eq!(metal.backend(), BackendKind::Metal);
        let metal_logits = metal.logits_pcm(&pcm, 16_000).expect("Metal classifier");
        let metal_cpu_logit_error = max_abs(&metal_logits, &actual_logits);
        let metal_prediction = metal.classify_pcm(&pcm, 16_000).expect("Metal prediction");
        let metal_probabilities = metal_prediction.probabilities;
        let metal_cpu_probability_error = max_abs(&metal_probabilities, &actual_probabilities);
        let metal_reference_logit_error = max_abs(&metal_logits, &expected_logits);
        let metal_reference_probability_error =
            max_abs(&metal_probabilities, &expected_probabilities);
        assert!(
            metal_cpu_logit_error <= FP32_PARITY_BOUND
                && metal_cpu_probability_error <= FP32_PARITY_BOUND,
            "Metal-vs-CPU metric exceeds FP32 bound {FP32_PARITY_BOUND:.9}"
        );
        eprintln!(
            "VOICE_GENDER_METAL_VS_CPU_METRICS logits_cpu_max_abs={metal_cpu_logit_error:.9} probabilities_cpu_max_abs={metal_cpu_probability_error:.9} bound={FP32_PARITY_BOUND:.9} fixture={FIXTURE_KIND}"
        );
        assert_eq!(argmax_values(&metal_probabilities), expected_argmax);
        assert_eq!(metal_prediction.label, expected_label);
        eprintln!(
            "VOICE_GENDER_METAL_VS_CPU PASS bound={FP32_PARITY_BOUND:.9} fixture={FIXTURE_KIND} label={expected_label}"
        );
        eprintln!(
            "VOICE_GENDER_METAL_VS_REFERENCE_METRICS logits_reference_max_abs={metal_reference_logit_error:.9} probabilities_reference_max_abs={metal_reference_probability_error:.9} bound={FP32_PARITY_BOUND:.9} fixture={FIXTURE_KIND}"
        );
        assert!(
            metal_reference_logit_error <= FP32_PARITY_BOUND
                && metal_reference_probability_error <= FP32_PARITY_BOUND,
            "Metal-vs-reference metric exceeds FP32 bound {FP32_PARITY_BOUND:.9}"
        );
        eprintln!(
            "VOICE_GENDER_METAL_VS_REFERENCE PASS bound={FP32_PARITY_BOUND:.9} fixture={FIXTURE_KIND} label={expected_label}"
        );
        eprintln!(
            "VOICE_GENDER_ARGMAX_LABEL_AGREEMENT PASS argmax={expected_argmax} label={expected_label} official={expected_label} cpu={} metal={}",
            prediction.label, metal_prediction.label
        );
    }
}
