//! Real-checkpoint measurement against the independent SpeechBrain 1.0.3
//! Lang-ID implementation.
//!
//! Numeric bounds are intentionally absent until both official variants have
//! been measured on VAST (CPU) and an Apple-silicon runner (Metal). The tests
//! are ignored shell tests: explicitly invoking one without both required
//! inputs is a hard failure, never a skip or a synthetic pass.

use std::path::{Path, PathBuf};

use vokra_core::json::{self, JsonValue};
use vokra_models::lang_id::{LangIdEcapa, LangIdVariant};

const GGUF_ENV: &str = "VOKRA_LANG_ID_GGUF";
const REFERENCE_DIR_ENV: &str = "VOKRA_LANG_ID_REFERENCE_DIR";

#[derive(Debug, Clone, Copy)]
struct Axes {
    n_mels: usize,
    embedding_dim: usize,
    class_count: usize,
    tensor_count: usize,
}

impl Axes {
    fn for_variant(variant: LangIdVariant) -> Self {
        match variant {
            LangIdVariant::VoxLingua107 => Self {
                n_mels: 60,
                embedding_dim: 256,
                class_count: 107,
                tensor_count: 212,
            },
            LangIdVariant::CommonLanguage => Self {
                n_mels: 80,
                embedding_dim: 192,
                class_count: 45,
                tensor_count: 201,
            },
        }
    }
}

#[derive(Debug)]
struct Reference {
    pcm: Vec<f32>,
    features: Vec<f32>,
    frames: usize,
    embedding: Vec<f32>,
    scores: Vec<f32>,
    labels: Vec<String>,
    best_index: usize,
    best_label: String,
}

impl Reference {
    fn load(directory: &Path, model: &LangIdEcapa, axes: Axes) -> Self {
        let manifest = read_json(&directory.join("manifest.json"));
        assert_eq!(
            required_str(&manifest, "format"),
            "vokra-speechbrain-lang-id-reference-v1",
            "reference format"
        );
        assert_eq!(
            required_str(&manifest, "source"),
            model.upstream_hf().expect("strict Lang-ID source"),
            "reference/GGUF upstream source"
        );
        assert_eq!(
            required_str(&manifest, "revision"),
            model.upstream_revision(),
            "reference/GGUF immutable revision"
        );
        assert_eq!(required_usize(&manifest, "sample_rate"), 16_000);

        for key in ["python", "numpy", "torch", "torchaudio", "speechbrain"] {
            assert!(
                !required_str(&manifest, key).is_empty(),
                "manifest `{key}` version is empty"
            );
        }
        validate_checkpoint_hashes(&manifest);

        let feature_shape = required_shape(&manifest, "feature_shape");
        let raw_feature_shape = required_shape(&manifest, "raw_feature_shape");
        assert_eq!(feature_shape.len(), 3, "feature rank");
        assert_eq!(
            raw_feature_shape, feature_shape,
            "raw/normalized feature shape"
        );
        assert_eq!(feature_shape[0], 1, "feature batch");
        assert_eq!(feature_shape[2], axes.n_mels, "feature mel width");
        let frames = feature_shape[1];
        assert!(frames > 0, "reference has no feature frames");
        assert_eq!(
            required_shape(&manifest, "embedding_shape"),
            [1, 1, axes.embedding_dim],
            "embedding shape"
        );
        assert_eq!(
            required_shape(&manifest, "score_shape"),
            [1, axes.class_count],
            "score shape"
        );

        let pcm = read_f32(&directory.join("pcm.f32.bin"));
        let features = read_f32(&directory.join("features.f32.bin"));
        let embedding = read_f32(&directory.join("embedding.f32.bin"));
        let scores = read_f32(&directory.join("scores.f32.bin"));
        let labels = read_labels(&directory.join("labels.json"));
        let best_index = required_usize(&manifest, "best_index");
        let best_label = required_str(&manifest, "best_label").to_owned();

        assert_eq!(
            pcm.len(),
            required_usize(&manifest, "pcm_samples"),
            "PCM samples"
        );
        assert_eq!(features.len(), frames * axes.n_mels, "feature elements");
        assert_eq!(embedding.len(), axes.embedding_dim, "embedding elements");
        assert_eq!(scores.len(), axes.class_count, "score elements");
        assert_eq!(labels.len(), axes.class_count, "label elements");
        assert_eq!(labels, model.labels(), "ordered official labels in GGUF");
        assert_eq!(argmax(&scores), best_index, "reference score argmax");
        assert_eq!(labels[best_index], best_label, "reference decoded label");

        Self {
            pcm,
            features,
            frames,
            embedding,
            scores,
            labels,
            best_index,
            best_label,
        }
    }
}

#[derive(Debug)]
struct Metrics {
    max_abs: f32,
    max_abs_index: usize,
    mean_abs: f64,
    relative_l1: f64,
    cosine: f64,
}

fn measure(label: &str, actual: &[f32], expected: &[f32]) -> Metrics {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    assert!(!actual.is_empty(), "{label} is empty");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{label} contains non-finite actual values"
    );
    assert!(
        expected.iter().all(|value| value.is_finite()),
        "{label} contains non-finite reference values"
    );

    let (max_abs_index, max_abs) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty measurement");
    let sum_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| f64::from((actual - expected).abs()))
        .sum::<f64>();
    let expected_l1 = expected
        .iter()
        .map(|value| f64::from(value.abs()))
        .sum::<f64>();
    let dot = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| f64::from(*actual) * f64::from(*expected))
        .sum::<f64>();
    let actual_norm = actual
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    let expected_norm = expected
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    let metrics = Metrics {
        max_abs,
        max_abs_index,
        mean_abs: sum_abs / actual.len() as f64,
        relative_l1: sum_abs / expected_l1.max(1.0e-30),
        cosine: dot / (actual_norm * expected_norm).max(1.0e-30),
    };
    eprintln!(
        "LANG_ID_MEASURE stage={label:?} elements={} max_abs={:.9e} max_index={} actual_at_max={:.9e} reference_at_max={:.9e} mean_abs={:.9e} relative_l1={:.9e} cosine={:.9e}",
        actual.len(),
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

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    assert!(!bytes.is_empty(), "{} is empty", path.display());
    assert_eq!(bytes.len() % 4, 0, "{} is truncated", path.display());
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{} contains non-finite values",
        path.display()
    );
    values
}

fn read_json(path: &Path) -> JsonValue {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    json::parse(&bytes).unwrap_or_else(|error| panic!("invalid {}: {error}", path.display()))
}

fn read_labels(path: &Path) -> Vec<String> {
    read_json(path)
        .as_array()
        .unwrap_or_else(|| panic!("{} must be a JSON array", path.display()))
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .filter(|label| !label.is_empty())
                .unwrap_or_else(|| panic!("{} label {index} is empty/non-string", path.display()))
                .to_owned()
        })
        .collect()
}

fn required_str<'a>(value: &'a JsonValue, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .unwrap_or_else(|| panic!("manifest `{key}` must be a string"))
}

fn required_usize(value: &JsonValue, key: &str) -> usize {
    usize::try_from(
        value
            .get(key)
            .and_then(JsonValue::as_u64)
            .unwrap_or_else(|| panic!("manifest `{key}` must be a non-negative integer")),
    )
    .unwrap_or_else(|_| panic!("manifest `{key}` does not fit usize"))
}

fn required_shape(value: &JsonValue, key: &str) -> Vec<usize> {
    value
        .get(key)
        .and_then(JsonValue::as_array)
        .unwrap_or_else(|| panic!("manifest `{key}` must be an array"))
        .iter()
        .enumerate()
        .map(|(index, dimension)| {
            usize::try_from(
                dimension
                    .as_u64()
                    .unwrap_or_else(|| panic!("manifest `{key}` dimension {index} is invalid")),
            )
            .unwrap_or_else(|_| panic!("manifest `{key}` dimension {index} does not fit usize"))
        })
        .collect()
}

fn validate_checkpoint_hashes(manifest: &JsonValue) {
    let hashes = manifest
        .get("checkpoint_sha256")
        .and_then(JsonValue::as_object)
        .expect("manifest `checkpoint_sha256` must be an object");
    for filename in [
        "embedding_model.ckpt",
        "classifier.ckpt",
        "label_encoder.txt",
    ] {
        let hash = hashes
            .iter()
            .find(|(name, _)| name == filename)
            .and_then(|(_, value)| value.as_str())
            .unwrap_or_else(|| panic!("manifest checkpoint hash missing `{filename}`"));
        assert_eq!(hash.len(), 64, "{filename} SHA-256 width");
        assert!(
            hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "{filename} SHA-256 is not hexadecimal"
        );
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .expect("argmax requires non-empty values")
}

fn load_case() -> (PathBuf, LangIdEcapa, Reference) {
    let gguf = std::env::var_os(GGUF_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {GGUF_ENV} when explicitly running this ignored test"));
    let reference_dir = std::env::var_os(REFERENCE_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!("set {REFERENCE_DIR_ENV} when explicitly running this ignored test")
        });
    let model = LangIdEcapa::from_path(&gguf).expect("strict prepared-v2 Lang-ID GGUF bind");
    let variant = model.variant().expect("strict Lang-ID variant");
    let axes = Axes::for_variant(variant);
    assert_eq!(
        model.tensor_count(),
        axes.tensor_count,
        "complete tensor count"
    );
    assert_eq!(
        model.language_count(),
        Some(axes.class_count),
        "official class count"
    );
    let reference = Reference::load(&reference_dir, &model, axes);
    (gguf, model, reference)
}

fn print_environment(backend: &str) {
    let parallelism = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(0);
    let cpu_isa = std::env::var("VOKRA_CPU_ISA").unwrap_or_else(|_| "auto".to_owned());
    eprintln!(
        "LANG_ID_ENV backend={backend} os={} arch={} parallelism={} vokra_cpu_isa={cpu_isa}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        parallelism,
    );
}

fn assert_official_winner(model: &LangIdEcapa, scores: &[f32], reference: &Reference) {
    let actual_index = argmax(scores);
    let (best_index, best_label) = model.best_label(scores).expect("decode Lang-ID score");
    assert_eq!(best_index, actual_index, "runtime argmax helper");
    assert_eq!(actual_index, reference.best_index, "official winning index");
    assert_eq!(
        best_label,
        reference.best_label.as_str(),
        "official winning label"
    );
    assert_eq!(
        reference.labels[actual_index], reference.best_label,
        "ordered label inventory"
    );
}

#[test]
#[ignore = "requires a VAST-generated prepared GGUF and independent SpeechBrain fixture"]
fn measure_cpu_against_independent_speechbrain() {
    let (_gguf, model, reference) = load_case();
    print_environment("cpu");

    let (features, frames) = model
        .frontend_features(&reference.pcm, 16_000)
        .expect("CPU SpeechBrain frontend");
    assert_eq!(frames, reference.frames, "frontend frame count");
    measure(
        "cpu_frontend_vs_speechbrain",
        &features,
        &reference.features,
    );

    let embedding = model
        .embed_features(&reference.features, reference.frames)
        .expect("CPU ECAPA on official features");
    measure(
        "cpu_embedding_vs_speechbrain",
        &embedding,
        &reference.embedding,
    );

    let head_from_official_embedding = model
        .classify_embedding(&reference.embedding)
        .expect("CPU classifier on official embedding");
    measure(
        "cpu_classifier_vs_speechbrain",
        &head_from_official_embedding,
        &reference.scores,
    );

    let scores_from_official_features = model
        .identify_features(&reference.features, reference.frames)
        .expect("CPU ECAPA + classifier on official features");
    measure(
        "cpu_network_vs_speechbrain",
        &scores_from_official_features,
        &reference.scores,
    );

    let end_to_end = model
        .identify_pcm(&reference.pcm, 16_000)
        .expect("CPU Lang-ID end-to-end");
    measure(
        "cpu_end_to_end_vs_speechbrain",
        &end_to_end,
        &reference.scores,
    );
    assert_official_winner(&model, &end_to_end, &reference);

    eprintln!(
        "LANG_ID_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED"
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires an Apple-silicon runner, prepared GGUF and independent SpeechBrain fixture"]
fn measure_metal_against_cpu_and_independent_speechbrain() {
    use vokra_core::BackendKind;

    let (gguf, cpu, reference) = load_case();
    let metal = LangIdEcapa::from_path(gguf)
        .expect("strict Lang-ID bind for Metal")
        .with_backend(BackendKind::Metal);
    print_environment("metal");

    let cpu_embedding = cpu
        .embed_features(&reference.features, reference.frames)
        .expect("CPU ECAPA on official features");
    let metal_embedding = metal
        .embed_features(&reference.features, reference.frames)
        .expect("Metal ECAPA on official features");
    measure(
        "metal_embedding_vs_speechbrain",
        &metal_embedding,
        &reference.embedding,
    );
    measure("metal_embedding_vs_cpu", &metal_embedding, &cpu_embedding);

    let cpu_head = cpu
        .classify_embedding(&reference.embedding)
        .expect("CPU classifier on official embedding");
    let metal_head = metal
        .classify_embedding(&reference.embedding)
        .expect("Metal classifier on official embedding");
    measure(
        "metal_classifier_vs_speechbrain",
        &metal_head,
        &reference.scores,
    );
    measure("metal_classifier_vs_cpu", &metal_head, &cpu_head);

    let cpu_scores = cpu
        .identify_features(&reference.features, reference.frames)
        .expect("CPU network on official features");
    let metal_scores = metal
        .identify_features(&reference.features, reference.frames)
        .expect("Metal network on official features");
    measure(
        "metal_network_vs_speechbrain",
        &metal_scores,
        &reference.scores,
    );
    measure("metal_network_vs_cpu", &metal_scores, &cpu_scores);

    let cpu_end_to_end = cpu
        .identify_pcm(&reference.pcm, 16_000)
        .expect("CPU end-to-end");
    let metal_end_to_end = metal
        .identify_pcm(&reference.pcm, 16_000)
        .expect("Metal end-to-end");
    measure(
        "metal_end_to_end_vs_speechbrain",
        &metal_end_to_end,
        &reference.scores,
    );
    measure(
        "metal_end_to_end_vs_cpu",
        &metal_end_to_end,
        &cpu_end_to_end,
    );
    assert_official_winner(&cpu, &cpu_end_to_end, &reference);
    assert_official_winner(&metal, &metal_end_to_end, &reference);

    eprintln!(
        "LANG_ID_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED"
    );
}
