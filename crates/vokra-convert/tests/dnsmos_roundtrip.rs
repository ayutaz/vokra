//! DNSMOS converter external contract fences.
//!
//! Full-manifest success is covered next to the strict converter, where the
//! audited 38-tensor manifest is authoritative. These integration tests pin
//! the crate-root API and verify that the public dispatch path preserves the
//! same fail-closed behavior for a historical two-tensor skeleton.

use std::path::PathBuf;

use vokra_convert::{DnsmosReport, ModelKind, convert_dnsmos_file, convert_file};

fn scratch(tag: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "vokra-dnsmos-rt-{}-{tag}-{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        extension
    ))
}

fn historical_skeleton() -> Vec<u8> {
    let header = r#"{"p808.model_v8.conv1.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"p835.sig_bak_ovr.conv1.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[16,40]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(bytes.len() + 40, 0);
    bytes
}

#[test]
fn crate_root_reexports_keep_the_file_converter_signature() {
    let _: fn(
        &std::path::Path,
        &std::path::Path,
        Option<&str>,
    ) -> Result<DnsmosReport, vokra_convert::ConvertError> = convert_dnsmos_file;
}

#[test]
fn direct_and_model_kind_dispatch_both_reject_partial_bundles() {
    let input = scratch("partial", "safetensors");
    let direct = scratch("partial-direct", "gguf");
    let dispatch = scratch("partial-dispatch", "gguf");
    std::fs::write(&input, historical_skeleton()).unwrap();

    let direct_error = convert_dnsmos_file(&input, &direct, None).unwrap_err();
    assert!(direct_error.to_string().contains("expected exactly 38"));
    let dispatch_error = convert_file(ModelKind::Dnsmos, &input, &dispatch).unwrap_err();
    assert!(dispatch_error.to_string().contains("expected exactly 38"));
    assert!(!direct.exists());
    assert!(!dispatch.exists());

    std::fs::remove_file(input).ok();
}

#[test]
fn canonical_aliases_resolve_to_dnsmos() {
    for alias in ["dnsmos", "dnsmos-p808-p835", "microsoft/dnsmos"] {
        assert_eq!(ModelKind::from_arg(alias), Some(ModelKind::Dnsmos));
    }
}
