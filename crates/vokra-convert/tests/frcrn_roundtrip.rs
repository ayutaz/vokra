//! Public FRCRN converter surface tests.
//!
//! The real 812-tensor success path is a VAST-only gate. These lightweight
//! tests prove that the CLI dispatch remains wired and that the public entry
//! point no longer stamps arbitrary float checkpoints as FRCRN.

use std::path::PathBuf;

use vokra_convert::{ModelKind, convert_file, convert_frcrn_file};

fn scratch(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "vokra-convert-frcrn-it-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn incomplete_safetensors() -> Vec<u8> {
    let payload_bytes = 642 * 640 * 4;
    let header = format!(
        "{{\"stft.weight\":{{\"dtype\":\"F32\",\"shape\":[642,1,640],\"data_offsets\":[0,{payload_bytes}]}}}}"
    );
    let mut bytes = Vec::with_capacity(8 + header.len() + payload_bytes);
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(8 + header.len() + payload_bytes, 0);
    bytes
}

#[test]
fn public_entry_points_refuse_incomplete_checkpoint() {
    let input = scratch("in.safetensors");
    let direct_output = scratch("direct.gguf");
    let dispatch_output = scratch("dispatch.gguf");
    std::fs::write(&input, incomplete_safetensors()).unwrap();

    let direct = convert_frcrn_file(&input, &direct_output, None).unwrap_err();
    assert!(direct.to_string().contains("expected exactly 812"));
    assert!(!direct_output.exists());

    let dispatched = convert_file(ModelKind::Frcrn, &input, &dispatch_output).unwrap_err();
    assert!(dispatched.to_string().contains("expected exactly 812"));
    assert!(!dispatch_output.exists());

    std::fs::remove_file(input).ok();
}

#[test]
fn frcrn_alias_dispatch_round_trips() {
    let kind = ModelKind::from_arg("frcrn").expect("`--model frcrn` must resolve");
    assert_eq!(kind, ModelKind::Frcrn);
    assert_eq!(kind.as_arg(), "frcrn");

    for alias in [
        "frcrn-se-16k",
        "frcrn_se_16k",
        "alibabasglab/frcrn_se_16k",
        "alibabasglab/frcrn",
        "clearervoice-studio/frcrn",
        "modelscope/clearervoice-studio-frcrn",
    ] {
        assert_eq!(ModelKind::from_arg(alias), Some(ModelKind::Frcrn));
    }
    for miss in ["frcrn-v2", "frcrn/large", "frcrn-huge"] {
        assert!(ModelKind::from_arg(miss).is_none());
    }
}
