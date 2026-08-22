//! External contract tests for the pinned aiola Whisper-Medusa-v1 converter.
//!
//! The real 6.25 GB checkpoint is covered by the VAST-only parity test. These
//! small fixtures pin the public side-car API, source-name canonicalisation,
//! metadata group, MIT default, and the fail-fast legacy-dispatch error.

use std::path::PathBuf;

use vokra_convert::{ModelKind, convert_file, convert_whisper_medusa_v1_with_config};
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufFile};

const OFFICIAL_CONFIG: &str = r#"{
    "whisper_model_name":"openai/whisper-large-v2",
    "medusa_num_heads":10,
    "medusa_num_layers":1,
    "medusa_hidden_size":1280,
    "medusa_heads_type":"base_head",
    "medusa_choices":[1,1,1,1,1,1,1,1,1,1,1],
    "init_from_proj":true
}"#;

fn tmp_path(tag: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "vokra-whisper-medusa-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.subsec_nanos())
            .unwrap_or(0)
    ));
    path
}

fn one_bf16_tensor(name: &str, shape: &[u64], bytes: &[u8]) -> Vec<u8> {
    let elements: u64 = shape.iter().product();
    assert_eq!(bytes.len(), elements as usize * 2);
    let shape = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let header = format!(
        r#"{{"{name}":{{"dtype":"BF16","shape":[{shape}],"data_offsets":[0,{}]}}}}"#,
        bytes.len()
    );
    let mut output = Vec::new();
    output.extend_from_slice(&(header.len() as u64).to_le_bytes());
    output.extend_from_slice(header.as_bytes());
    output.extend_from_slice(bytes);
    output
}

#[test]
fn config_aware_api_stamps_exact_contract_and_canonicalises_base_names() {
    let input = tmp_path("input");
    let config = tmp_path("config");
    let output = tmp_path("output");
    let payload = [0x80, 0x3f, 0x20, 0xc0, 0x20, 0x3e, 0x28, 0x42];
    let source_name = "whisper_model.model.encoder.layers.0.self_attn.q_proj.weight";
    std::fs::write(&input, one_bf16_tensor(source_name, &[2, 2], &payload)).unwrap();
    std::fs::write(&config, OFFICIAL_CONFIG).unwrap();

    let summary = convert_whisper_medusa_v1_with_config(&input, &config, &output, None)
        .expect("config-aware conversion");
    assert_eq!(summary.model, ModelKind::WhisperMedusaV1);
    assert_eq!(summary.tensor_count, 1);

    let file = GgufFile::open(&output).unwrap();
    let tensor = file
        .tensor_info("model.encoder.layers.0.self_attn.q_proj.weight")
        .expect("outer whisper_model wrapper removed");
    assert_eq!(tensor.dtype, GgmlType::BF16);
    assert_eq!(file.tensor_bytes(tensor), payload);
    assert!(file.tensor_info(source_name).is_none());

    assert_eq!(
        file.get("vokra.model.arch")
            .and_then(|value| value.as_str()),
        Some("whisper-medusa-v1")
    );
    assert_eq!(
        file.get("vokra.provenance.license")
            .and_then(|value| value.as_str()),
        Some("MIT")
    );
    assert_eq!(
        file.get("vokra.provenance.weight_license")
            .and_then(|value| value.as_str()),
        Some(LicenseClass::Permissive.as_str())
    );
    assert_eq!(
        file.get("vokra.medusa.module_count")
            .and_then(|value| value.as_u64()),
        Some(11)
    );
    assert_eq!(
        file.get("vokra.medusa.hidden_size")
            .and_then(|value| value.as_u64()),
        Some(1280)
    );
    assert_eq!(
        file.get("vokra.medusa.checkpoint_sha256")
            .and_then(|value| value.as_str()),
        Some("ec634d5ece33a8d634ed2e188c7bfbde7adab4410932b8fa6c20440836a423f3")
    );

    std::fs::remove_file(input).ok();
    std::fs::remove_file(config).ok();
    std::fs::remove_file(output).ok();
}

#[test]
fn legacy_dispatch_rejects_before_reading_the_large_input() {
    let error = convert_file(
        ModelKind::WhisperMedusaV1,
        &PathBuf::from("/definitely/missing/whisper-medusa.safetensors"),
        &tmp_path("unused"),
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("requires the exact upstream config.json"));
    assert!(message.contains("convert_whisper_medusa_v1_with_config"));
}
