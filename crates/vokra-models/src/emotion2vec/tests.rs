use std::path::Path;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{
    GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType, chunks,
};

use super::bound::{CONTRACT_KEYS, validate_metadata};
use super::forward::{alibi_bias, alibi_slopes};
use super::*;

fn base_builder() -> GgufBuilder {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string("vokra.model.category", CATEGORY);
    builder.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "permissive");
    builder.add_string(chunks::KEY_PROVENANCE_LICENSE, "mit");
    builder.add_string(chunks::KEY_PROVENANCE_MODEL_ID, NAME);
    builder.add_string(chunks::KEY_PROVENANCE_SOURCE, UPSTREAM_HF);
    builder
}

fn add_u32_array(builder: &mut GgufBuilder, key: &str, values: &[usize]) {
    builder.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: values
                .iter()
                .map(|value| GgufMetadataValue::U32(*value as u32))
                .collect(),
        }),
    );
}

fn add_string_array(builder: &mut GgufBuilder, key: &str, values: &[&str]) {
    builder.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: values
                .iter()
                .map(|value| GgufMetadataValue::String((*value).to_owned()))
                .collect(),
        }),
    );
}

fn stamp_contract(builder: &mut GgufBuilder, labels: &[&str]) {
    builder.add_u32("vokra.emotion2vec.sample_rate", SAMPLE_RATE);
    builder.add_u32("vokra.emotion2vec.embed_dim", HIDDEN as u32);
    builder.add_u32("vokra.emotion2vec.depth", GLOBAL_LAYERS as u32);
    builder.add_u32("vokra.emotion2vec.prenet_depth", CONTEXT_LAYERS as u32);
    builder.add_u32("vokra.emotion2vec.num_heads", HEADS as u32);
    builder.add_u32("vokra.emotion2vec.mlp_dim", FFN as u32);
    builder.add_u32("vokra.emotion2vec.num_extra_tokens", EXTRA_TOKENS as u32);
    builder.add_u32("vokra.emotion2vec.num_classes", NUM_CLASSES as u32);
    builder.add_u32("vokra.emotion2vec.conv_pos_depth", POSITION_LAYERS as u32);
    builder.add_u32("vokra.emotion2vec.conv_pos_kernel", POSITION_KERNEL as u32);
    builder.add_u32("vokra.emotion2vec.conv_pos_groups", POSITION_GROUPS as u32);
    builder.add_f32("vokra.emotion2vec.layer_norm_eps", LAYER_NORM_EPS);
    builder.add_bool("vokra.emotion2vec.normalize", true);
    add_u32_array(builder, "vokra.emotion2vec.conv_dim", &CONV_DIM);
    add_u32_array(builder, "vokra.emotion2vec.conv_kernel", &CONV_KERNEL);
    add_u32_array(builder, "vokra.emotion2vec.conv_stride", &CONV_STRIDE);
    add_string_array(builder, "vokra.emotion2vec.class_labels", labels);
}

fn parse(builder: GgufBuilder) -> GgufFile {
    GgufFile::parse(builder.to_bytes().expect("serialize GGUF")).expect("parse GGUF")
}

#[test]
fn topology_and_labels_are_pinned() {
    assert_eq!(TENSOR_COUNT, 185);
    assert_eq!(HIDDEN, 1_024);
    assert_eq!(GLOBAL_LAYERS, 8);
    assert_eq!(CONTEXT_LAYERS, 4);
    assert_eq!(HEADS, 16);
    assert_eq!(Emotion2Vec::class_labels(), &EMOTION_CLASS_LABELS);
    assert_eq!(EMOTION_CLASS_LABELS[0], "生气/angry");
    assert_eq!(EMOTION_CLASS_LABELS[8], "<unk>");
}

#[test]
fn exact_legacy_metadata_is_repaired() {
    let file = parse(base_builder());
    assert!(validate_metadata(&file).expect("legacy public metadata"));
}

#[test]
fn complete_contract_is_accepted_without_repair() {
    let mut builder = base_builder();
    stamp_contract(&mut builder, &EMOTION_CLASS_LABELS);
    let file = parse(builder);
    assert!(!validate_metadata(&file).expect("complete canonical group"));
}

#[test]
fn partial_contract_fails_closed() {
    let mut builder = base_builder();
    builder.add_u32(CONTRACT_KEYS[0], SAMPLE_RATE);
    let error = validate_metadata(&parse(builder)).unwrap_err().to_string();
    assert!(error.contains("partial `vokra.emotion2vec.*` metadata"));
    assert!(error.contains("1/17"));
}

#[test]
fn label_reorder_fails_closed() {
    let mut labels = EMOTION_CLASS_LABELS;
    labels.swap(0, 1);
    let mut builder = base_builder();
    stamp_contract(&mut builder, &labels);
    let error = validate_metadata(&parse(builder)).unwrap_err().to_string();
    assert!(error.contains("class_labels"));
    assert!(error.contains("element 0"));
}

#[test]
fn alibi_is_symmetric_and_zero_around_extra_tokens() {
    let slopes = alibi_slopes(HEADS);
    assert_eq!(slopes.len(), HEADS);
    assert!((slopes[0] - 2.0f32.powf(-0.5)).abs() < 1.0e-7);
    assert!((slopes[15] - 1.0 / 256.0).abs() < 1.0e-7);
    let scales = vec![1.0f32; HEADS];
    assert_eq!(alibi_bias(0, 0, 20, &slopes, &scales), 0.0);
    assert_eq!(alibi_bias(0, 20, 9, &slopes, &scales), 0.0);
    let left = alibi_bias(0, 12, 20, &slopes, &scales);
    let right = alibi_bias(0, 20, 12, &slopes, &scales);
    assert_eq!(left, right);
    assert!(left < 0.0);
    let negative_scale = vec![-1.0f32; HEADS];
    assert_eq!(alibi_bias(0, 12, 20, &slopes, &negative_scale), 0.0);
}

#[test]
fn invalid_pcm_and_rate_are_explicit() {
    assert!(validate_pcm(&[0.0; MIN_PCM_SAMPLES - 1], SAMPLE_RATE).is_err());
    assert!(validate_pcm(&[0.0; MIN_PCM_SAMPLES], 44_100).is_err());
    let mut non_finite = [0.0f32; MIN_PCM_SAMPLES];
    non_finite[13] = f32::NAN;
    assert!(validate_pcm(&non_finite, SAMPLE_RATE).is_err());
}

#[test]
fn manifest_identity_is_pinned() {
    assert_eq!(SPEC.tensor_count, 185);
    assert_eq!(
        SPEC.manifest_sha256,
        [
            0xf5, 0xf8, 0xf6, 0x84, 0x30, 0x2c, 0xf5, 0x5f, 0xb3, 0x99, 0x27, 0x7a, 0x74, 0x46,
            0x97, 0x6a, 0x77, 0xf5, 0x70, 0x81, 0x6e, 0x7e, 0x33, 0x45, 0xa0, 0x08, 0xe4, 0xd0,
            0xb6, 0x77, 0x44, 0x01,
        ]
    );
}

#[test]
fn learned_hot_ops_are_cpu_and_metal_complete() {
    Compute::for_backend(BackendKind::Cpu, EMOTION2VEC_HOT_OPS)
        .expect("CPU covers every emotion2vec learned operation");
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    match Compute::for_backend(BackendKind::Metal, EMOTION2VEC_HOT_OPS) {
        Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
        Err(VokraError::BackendUnavailable(_)) => {}
        Err(error) => panic!("emotion2vec has a Metal coverage gap: {error}"),
    }
}

#[test]
#[ignore = "requires VAST-prepared public GGUF and official FunASR fixture"]
fn measure_official_cpu_against_funasr() {
    let (model, wave, reference) = real_case(BackendKind::Cpu);
    let (logits, scores, taps) = model
        .classify_with_taps(&wave.samples, wave.sample_rate)
        .expect("native emotion2vec CPU forward");
    measure_taps("cpu_vs_funasr", &reference, &taps, &logits, &scores);
    eprintln!(
        "EMOTION2VEC_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED"
    );
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires Apple Silicon, public GGUF and official FunASR fixture"]
fn measure_official_metal_against_cpu_and_funasr() {
    if vokra_backend_metal::vokra_metal_probe().is_err() {
        eprintln!("skipping emotion2vec Metal measurement: no system Metal device");
        return;
    }
    let (cpu, wave, reference) = real_case(BackendKind::Cpu);
    let (cpu_logits, cpu_scores, cpu_taps) = cpu
        .classify_with_taps(&wave.samples, wave.sample_rate)
        .expect("native emotion2vec CPU forward");
    let (metal, _, _) = real_case(BackendKind::Metal);
    let (metal_logits, metal_scores, metal_taps) = metal
        .classify_with_taps(&wave.samples, wave.sample_rate)
        .expect("native emotion2vec Metal forward");
    measure_taps(
        "metal_vs_funasr",
        &reference,
        &metal_taps,
        &metal_logits,
        &metal_scores,
    );
    measure_pair(
        "metal_vs_cpu/normalized_pcm",
        &metal_taps.normalized_pcm,
        &cpu_taps.normalized_pcm,
    );
    measure_pair(
        "metal_vs_cpu/conv_features",
        &metal_taps.conv_features,
        &cpu_taps.conv_features,
    );
    measure_pair(
        "metal_vs_cpu/projected_features",
        &metal_taps.projected_features,
        &cpu_taps.projected_features,
    );
    measure_pair(
        "metal_vs_cpu/context_features",
        &metal_taps.context_features,
        &cpu_taps.context_features,
    );
    measure_pair(
        "metal_vs_cpu/final_features",
        &metal_taps.final_features,
        &cpu_taps.final_features,
    );
    measure_pair(
        "metal_vs_cpu/pooled_embedding",
        &metal_taps.pooled_embedding,
        &cpu_taps.pooled_embedding,
    );
    measure_pair("metal_vs_cpu/logits", &metal_logits, &cpu_logits);
    measure_pair("metal_vs_cpu/scores", &metal_scores, &cpu_scores);
    eprintln!(
        "EMOTION2VEC_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED"
    );
}

fn real_case(
    backend: BackendKind,
) -> (
    Emotion2Vec,
    crate::silero_vad::wav::WavData,
    std::path::PathBuf,
) {
    let gguf = std::env::var("VOKRA_EMOTION2VEC_GGUF")
        .expect("VOKRA_EMOTION2VEC_GGUF must point at the strict public GGUF");
    let reference = std::env::var("VOKRA_EMOTION2VEC_REFERENCE_DIR")
        .expect("VOKRA_EMOTION2VEC_REFERENCE_DIR must point at the official dump");
    let wav = std::env::var("VOKRA_EMOTION2VEC_WAV")
        .expect("VOKRA_EMOTION2VEC_WAV must point at the official example WAV");
    let wave = crate::silero_vad::wav::read_wav_f32(wav).expect("read official WAV");
    assert_eq!(wave.sample_rate, SAMPLE_RATE);
    let model = Emotion2Vec::open(gguf)
        .expect("bind public GGUF")
        .with_backend(backend);
    (model, wave, Path::new(&reference).to_path_buf())
}

fn measure_taps(
    prefix: &str,
    reference: &Path,
    taps: &ForwardTaps,
    logits: &[f32],
    scores: &[f32],
) {
    for (name, actual) in [
        ("normalized_pcm", taps.normalized_pcm.as_slice()),
        ("conv_features", taps.conv_features.as_slice()),
        ("projected_features", taps.projected_features.as_slice()),
        ("context_features", taps.context_features.as_slice()),
        ("final_features", taps.final_features.as_slice()),
        ("pooled_embedding", taps.pooled_embedding.as_slice()),
        ("logits", logits),
        ("scores", scores),
    ] {
        measure_pair(
            &format!("{prefix}/{name}"),
            actual,
            &read_f32(&reference.join(format!("{name}.f32"))),
        );
    }
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(bytes.len() % 4, 0, "unaligned f32 fixture {path:?}");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn measure_pair(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    assert!(!actual.is_empty(), "{label} must not be empty");
    assert!(
        actual.iter().chain(expected).all(|value| value.is_finite()),
        "{label} must be finite"
    );
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut actual_sq = 0.0f64;
    let mut expected_sq = 0.0f64;
    let mut max_index = 0usize;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let error = (actual - expected).abs();
        if error > max_abs {
            max_abs = error;
            max_index = index;
        }
        sum_abs += error;
        sum_sq += error * error;
        dot += actual * expected;
        actual_sq += actual * actual;
        expected_sq += expected * expected;
    }
    let count = actual.len() as f64;
    let norm_product = actual_sq.sqrt() * expected_sq.sqrt();
    let cosine = if norm_product == 0.0 {
        f64::NAN
    } else {
        dot / norm_product
    };
    eprintln!(
        "EMOTION2VEC_MEASUREMENT label={label} elements={} max_abs={max_abs:.9e} \
         worst_index={max_index} mean_abs={:.9e} rms={:.9e} cosine={cosine:.12}",
        actual.len(),
        sum_abs / count,
        (sum_sq / count).sqrt(),
    );
}
