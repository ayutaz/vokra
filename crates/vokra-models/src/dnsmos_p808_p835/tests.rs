use super::*;
use vokra_core::gguf::{GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType};

#[test]
fn audited_contract_constants_are_stable() {
    assert_eq!(TENSOR_COUNT, 38);
    assert_eq!(SAMPLE_RATE, 16_000);
    assert_eq!(INPUT_LENGTH_SAMPLES, 144_160);
    assert_eq!(
        (P808_FRAMES, P808_N_FFT, P808_HOP, P808_N_MELS),
        (900, 321, 160, 120)
    );
    assert_eq!(
        (P835_FRAMES, P835_WINDOW, P835_HOP, P835_BINS),
        (900, 320, 160, 161)
    );
    assert_eq!(MANIFEST_SHA256.len(), 64);
    assert_eq!(DnsmosSubmodel::P808.tensor_prefix(), "p808.");
    assert_eq!(DnsmosSubmodel::P835.tensor_prefix(), "p835.");
    assert_eq!(KEY_DNSMOS_P808_TOPOLOGY, "vokra.dnsmos.p808.topology");
    assert_eq!(KEY_DNSMOS_P835_TOPOLOGY, "vokra.dnsmos.p835.topology");
}

#[test]
fn public_config_parser_retains_partial_inventory_compatibility() {
    let mut builder = GgufBuilder::new();
    builder.add_u32(KEY_DNSMOS_SAMPLE_RATE, SAMPLE_RATE);
    builder.add_metadata(
        KEY_DNSMOS_BUNDLE,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: vec![GgufMetadataValue::String("p808".to_owned())],
        }),
    );
    let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
    let config = DnsmosConfig::from_gguf(&file).unwrap();
    assert_eq!(config.bundle, ["p808"]);
    assert!(config.has_p808);
    assert!(!config.has_p835);
    config.validate().unwrap();
}

#[test]
fn synthesized_model_preserves_public_surface_without_partial_claims() {
    let model = Dnsmos::synthesized();
    assert_eq!(model.backend(), BackendKind::Cpu);
    assert_eq!(model.sample_rate(), SAMPLE_RATE);
    assert_eq!(model.tensor_count(), TENSOR_COUNT);
    assert_eq!(model.config().bundle, ["p808", "p835"]);
    assert_eq!(model.bundles().len(), 2);
    assert_eq!(model.weight_license(), LicenseClass::Unknown);
}

#[test]
fn invalid_pcm_fails_before_any_cnn_work() {
    let model = Dnsmos::synthesized();
    let error = model.score_p808(&[]).unwrap_err();
    assert!(matches!(error, VokraError::InvalidArgument(_)));
    assert!(error.to_string().contains("at least one"));

    let error = model.score_p835(&[f32::NAN]).unwrap_err();
    assert!(matches!(error, VokraError::InvalidArgument(_)));
    assert!(error.to_string().contains("not finite"));
}

#[test]
fn foreign_arch_is_rejected_before_tensor_decode() {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, "utmos");
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
    let error = Dnsmos::from_gguf(&file).unwrap_err();
    assert!(matches!(error, VokraError::ModelLoad(_)));
    assert!(error.to_string().contains("unsupported"));
}

#[test]
fn cpu_and_metal_backend_contract_is_gemm_only() {
    assert_eq!(DNSMOS_HOT_OPS, &[HotOp::Gemm]);
    Compute::for_backend(BackendKind::Cpu, DNSMOS_HOT_OPS).unwrap();
    match Compute::for_backend(BackendKind::Metal, DNSMOS_HOT_OPS) {
        Ok(_) | Err(VokraError::BackendUnavailable(_)) => {}
        Err(other) => panic!("unexpected Metal DNSMOS preflight result: {other}"),
    }
}
