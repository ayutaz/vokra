use super::*;
use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufBuilder};

fn add_zero(builder: &mut GgufBuilder, name: &str, dimensions: &[u64]) {
    let elements: u64 = dimensions.iter().product();
    builder
        .add_tensor(
            name,
            GgmlType::F32,
            dimensions.to_vec(),
            vec![0u8; (elements * 4) as usize],
        )
        .expect("add tensor");
}

fn add_manifest(builder: &mut GgufBuilder) {
    let channels = [1u64, 16, 32, 64, 64, 64, 64];
    for layer in 1..=6 {
        let output = channels[layer];
        let prefix = format!("cnn.model.bn{layer}");
        for suffix in ["bias", "running_mean", "running_var", "weight"] {
            add_zero(builder, &format!("{prefix}.{suffix}"), &[output]);
        }
        add_zero(builder, &format!("cnn.model.conv{layer}.bias"), &[output]);
        add_zero(
            builder,
            &format!("cnn.model.conv{layer}.weight"),
            &[output, channels[layer - 1], 3, 3],
        );
    }
    for head in 0..5 {
        let prefix = format!("pool_layers.{head}.model");
        for (name, dimensions) in [
            ("linear1.bias", vec![128]),
            ("linear1.weight", vec![128, 64]),
            ("linear2.bias", vec![1]),
            ("linear2.weight", vec![1, 128]),
            ("linear3.bias", vec![1]),
            ("linear3.weight", vec![1, 64]),
        ] {
            add_zero(builder, &format!("{prefix}.{name}"), &dimensions);
        }
    }
    for layer in 0..2 {
        let prefix = format!("time_dependency.model.layers.{layer}");
        for (name, dimensions) in [
            ("linear1.bias", vec![64]),
            ("linear1.weight", vec![64, 64]),
            ("linear2.bias", vec![64]),
            ("linear2.weight", vec![64, 64]),
            ("norm1.bias", vec![64]),
            ("norm1.weight", vec![64]),
            ("norm2.bias", vec![64]),
            ("norm2.weight", vec![64]),
            ("self_attn.in_proj_bias", vec![192]),
            ("self_attn.in_proj_weight", vec![192, 64]),
            ("self_attn.out_proj.bias", vec![64]),
            ("self_attn.out_proj.weight", vec![64, 64]),
        ] {
            add_zero(builder, &format!("{prefix}.{name}"), &dimensions);
        }
    }
    for (name, dimensions) in [
        ("time_dependency.model.linear.bias", vec![64]),
        ("time_dependency.model.linear.weight", vec![64, 384]),
        ("time_dependency.model.norm1.bias", vec![64]),
        ("time_dependency.model.norm1.weight", vec![64]),
    ] {
        add_zero(builder, name, &dimensions);
    }
}

fn fixture() -> GgufFile {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(chunks::KEY_PROVENANCE_MODEL_ID, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
    builder.add_string(chunks::KEY_PROVENANCE_LICENSE, DEFAULT_LICENSE_SPDX);
    builder.add_string(
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        LicenseClass::NonCommercialShareAlike.as_str(),
    );
    add_manifest(&mut builder);
    GgufFile::parse(builder.to_bytes().expect("serialize")).expect("parse")
}

#[test]
fn public_contract_is_pinned() {
    assert_eq!(ARCH, "nisqa_v2_weight");
    assert_eq!(NAME, "nisqa_v2_weight");
    assert_eq!(TENSOR_COUNT, 94);
    assert_eq!(HEAD_ORDER, ["mos", "noi", "dis", "col", "loud"]);
    assert_eq!(DEFAULT_LICENSE_SPDX, "cc-by-nc-sa-4.0");
    assert_eq!(CANONICAL_FRONT_END.n_fft, 4096);
    assert_eq!(CANONICAL_FRONT_END.seg_hop_length, 4);
    assert_eq!(CANONICAL_TOPOLOGY.cnn_pool, [[24, 7], [12, 5], [6, 3]]);
    assert_eq!(
        NISQA_HOT_OPS,
        [HotOp::Gemm, HotOp::Softmax, HotOp::LayerNorm]
    );
}

#[test]
fn complete_public_manifest_binds_and_fills_historical_metadata_gap() {
    let model = Nisqa::from_gguf(&fixture()).expect("bind exact public manifest");
    assert_eq!(model.tensor_count(), TENSOR_COUNT);
    assert_eq!(model.variant(), NisqaVariant::MultiDim);
    assert_eq!(model.config().front_end, Some(CANONICAL_FRONT_END));
    assert_eq!(model.config().topology, Some(CANONICAL_TOPOLOGY));
    assert_eq!(
        model.weight_license(),
        LicenseClass::NonCommercialShareAlike
    );
    assert!(model.is_research_only());
    assert_eq!(model.backend(), BackendKind::Cpu);
}

#[test]
fn strict_manifest_rejects_a_truncated_checkpoint() {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    add_zero(&mut builder, "cnn.model.conv1.weight", &[16, 1, 3, 3]);
    let file = GgufFile::parse(builder.to_bytes().expect("serialize")).expect("parse");
    let error = Nisqa::from_gguf(&file).expect_err("truncated manifest must fail");
    assert!(matches!(error, VokraError::ModelLoad(message) if message.contains("tensor count")));
}

#[test]
fn rate_less_api_fails_instead_of_guessing_48khz() {
    let model = Nisqa::from_gguf(&fixture()).expect("bind");
    let error = model
        .score(&[0.0; 128])
        .expect_err("sample rate is required");
    assert!(
        matches!(error, VokraError::InvalidArgument(message) if message.contains("score_at_sample_rate"))
    );
}

#[test]
fn score_head_order_round_trips_without_swapping_dimensions() {
    let score = NisqaScore::from_heads(&[1.0, 2.0, 3.0, 4.0, 5.0]).expect("five heads");
    assert_eq!(score.discontinuity, 3.0);
    assert_eq!(score.coloration, 4.0);
    assert_eq!(score.to_heads(), [1.0, 2.0, 3.0, 4.0, 5.0]);
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("could not read {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{} is not F32-aligned", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect()
}

fn validate_reference_manifest(reference: &std::path::Path) {
    let path = reference.join("manifest.json");
    let manifest = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("could not read {}: {error}", path.display()));
    for needle in [
        "official gabrielmittag/NISQA NISQA_lib.NISQA_DIM direct import",
        SOURCE_REVISION,
        SOURCE_CHECKPOINT_SHA256,
        "\"sample_rate\": 48000",
        "\"mos\"",
        "\"noi\"",
        "\"dis\"",
        "\"col\"",
        "\"loud\"",
    ] {
        assert!(
            manifest.contains(needle),
            "{} does not contain pinned contract {needle:?}",
            path.display()
        );
    }
}

fn measure(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    assert!(!actual.is_empty(), "{label} must not be empty");
    assert!(
        actual.iter().chain(expected).all(|value| value.is_finite()),
        "{label} must be finite"
    );
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut worst = 0usize;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let delta = (f64::from(actual) - f64::from(expected)).abs();
        if delta > max_abs {
            max_abs = delta;
            worst = index;
        }
        sum_abs += delta;
        sum_sq += delta * delta;
        eprintln!(
            "NISQA_MEASUREMENT label={label} head={} actual={actual:.9e} \
             expected={expected:.9e} abs={delta:.9e}",
            HEAD_ORDER[index]
        );
    }
    let count = actual.len() as f64;
    eprintln!(
        "NISQA_MEASUREMENT_SUMMARY label={label} heads={} max_abs={max_abs:.9e} \
         worst_head={} mean_abs={:.9e} rms={:.9e}",
        actual.len(),
        HEAD_ORDER[worst],
        sum_abs / count,
        (sum_sq / count).sqrt(),
    );
}

fn real_case() -> (GgufFile, Vec<f32>, Vec<f32>) {
    let gguf = std::env::var_os("VOKRA_NISQA_GGUF")
        .expect("VOKRA_NISQA_GGUF must point at the strict public or regenerated GGUF");
    let reference = std::env::var_os("VOKRA_NISQA_REFERENCE_DIR")
        .expect("VOKRA_NISQA_REFERENCE_DIR must point at the independent official dump");
    let reference = std::path::PathBuf::from(reference);
    validate_reference_manifest(&reference);
    let file = GgufFile::open(gguf).expect("open real NISQA GGUF");
    let pcm = read_f32(&reference.join("pcm.f32le"));
    let expected = read_f32(&reference.join("score.f32le"));
    assert_eq!(expected.len(), N_HEADS, "official score width");
    (file, pcm, expected)
}

#[test]
#[ignore = "requires VAST-prepared public GGUF and independent pinned NISQA fixture"]
fn measure_real_cpu_against_official_nisqa() {
    let (file, pcm, expected) = real_case();
    let model =
        Nisqa::from_gguf_with_backend(&file, BackendKind::Cpu).expect("strict real NISQA CPU bind");
    let actual = model
        .score_at_sample_rate(&pcm, 48_000)
        .expect("real NISQA CPU forward")
        .to_heads();
    measure("cpu_vs_official", &actual, &expected);
    eprintln!("NISQA_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon, public GGUF and independent pinned NISQA fixture"]
fn measure_real_metal_against_cpu_and_official_nisqa() {
    if vokra_backend_metal::vokra_metal_probe().is_err() {
        eprintln!("skipping NISQA Metal measurement: no system Metal device");
        return;
    }
    let (file, pcm, expected) = real_case();
    let cpu = Nisqa::from_gguf_with_backend(&file, BackendKind::Cpu)
        .expect("strict real NISQA CPU bind")
        .score_at_sample_rate(&pcm, 48_000)
        .expect("real NISQA CPU forward")
        .to_heads();
    let metal = Nisqa::from_gguf_with_backend(&file, BackendKind::Metal)
        .expect("strict real NISQA Metal bind")
        .score_at_sample_rate(&pcm, 48_000)
        .expect("real NISQA Metal forward")
        .to_heads();
    measure("metal_vs_cpu", &metal, &cpu);
    measure("metal_vs_official", &metal, &expected);
    eprintln!(
        "NISQA_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED"
    );
}
