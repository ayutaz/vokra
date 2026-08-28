//! Canary-1B-v2 public converter contract.
//!
//! Real conversion is VAST-only because the pinned prepared checkpoint is
//! larger than 2 GB. These integration tests exercise fail-closed boundaries
//! without reading or synthesizing model payloads.

use std::path::PathBuf;

use vokra_convert::{
    ModelKind, convert_canary_file, convert_canary_file_with_tokenizer, convert_file,
};

fn tmp_path(tag: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "vokra-convert-canary-1b-v2-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default()
    ));
    path
}

#[test]
fn generic_dispatch_rejects_tokenizer_less_conversion_before_io() {
    let output = tmp_path("generic-output");
    let error = convert_file(
        ModelKind::Canary,
        std::path::Path::new("does-not-exist.safetensors"),
        &output,
    )
    .expect_err("generic path must not emit an unusable tokenizer-less GGUF");
    let message = error.to_string();
    assert!(message.contains("--tokenizer"), "message: {message}");
    assert!(message.contains("tokenizer.vocab"), "message: {message}");
    assert!(
        !output.exists(),
        "rejected conversion must not create output"
    );
}

#[test]
fn direct_legacy_entry_rejects_tokenizer_less_conversion_before_io() {
    let output = tmp_path("direct-output");
    let error = convert_canary_file(std::path::Path::new("does-not-exist.safetensors"), &output)
        .expect_err("legacy direct path must fail closed");
    assert!(error.to_string().contains("--tokenizer"));
    assert!(
        !output.exists(),
        "rejected conversion must not create output"
    );
}

#[test]
fn dedicated_path_authenticates_tokenizer_before_large_checkpoint_io() {
    let tokenizer = tmp_path("wrong-tokenizer");
    let output = tmp_path("wrong-tokenizer-output");
    std::fs::write(&tokenizer, b"<unk>\t0\n").expect("write tiny wrong tokenizer");

    let error = convert_canary_file_with_tokenizer(
        std::path::Path::new("does-not-exist.safetensors"),
        &output,
        None,
        &tokenizer,
    )
    .expect_err("wrong tokenizer must fail before opening checkpoint");
    let message = error.to_string();
    assert!(message.contains("SHA-256"), "message: {message}");
    assert!(
        !message.contains("does-not-exist.safetensors"),
        "checkpoint I/O happened before tokenizer authentication: {message}"
    );
    assert!(
        !output.exists(),
        "rejected conversion must not create output"
    );

    let _ = std::fs::remove_file(tokenizer);
}

#[test]
fn canary_1b_v2_alias_dispatch_round_trips() {
    let kind = ModelKind::from_arg("canary-1b-v2").expect("canonical v2 alias resolves");
    assert_eq!(kind, ModelKind::Canary);
    assert_eq!(kind.as_arg(), "canary");

    for alias in ["canary", "canary-1b-v2", "canary-1b-v2-en", "canary-1b_v2"] {
        assert_eq!(ModelKind::from_arg(alias), Some(ModelKind::Canary));
    }
    assert_eq!(
        ModelKind::from_arg("nvidia/canary-1b-flash"),
        Some(ModelKind::Canary1bFlash),
        "Flash must not be routed through the v2 manifest"
    );
}
