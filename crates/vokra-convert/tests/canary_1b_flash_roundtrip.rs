//! Canary-1B-Flash public converter contract.
//!
//! Real conversion is VAST-only because the pinned source is 3.54 GB. These
//! integration tests exercise the cheap fail-closed boundaries without
//! reading or synthesizing model payloads.

use std::path::PathBuf;

use vokra_convert::{
    ModelKind, convert_canary_1b_flash_file, convert_canary_1b_flash_file_with_tokenizer,
    convert_file,
};

fn tmp_path(tag: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "vokra-convert-canary-1b-flash-{tag}-{}-{}",
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
        ModelKind::Canary1bFlash,
        std::path::Path::new("does-not-exist.safetensors"),
        &output,
    )
    .expect_err("generic path must not emit an unusable tokenizer-less GGUF");
    let message = error.to_string();
    assert!(message.contains("--tokenizer"), "message: {message}");
    assert!(message.contains("aggregate.vocab"), "message: {message}");
    assert!(
        !output.exists(),
        "rejected conversion must not create output"
    );
}

#[test]
fn direct_legacy_entry_rejects_tokenizer_less_conversion_before_io() {
    let output = tmp_path("direct-output");
    let error = convert_canary_1b_flash_file(
        std::path::Path::new("does-not-exist.safetensors"),
        &output,
        None,
    )
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

    let error = convert_canary_1b_flash_file_with_tokenizer(
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
fn canary_1b_flash_alias_dispatch_round_trips() {
    let kind =
        ModelKind::from_arg("canary-1b-flash").expect("canonical Canary-1B-Flash alias resolves");
    assert_eq!(kind, ModelKind::Canary1bFlash);
    assert_eq!(kind.as_arg(), "canary-1b-flash");

    for alias in [
        "canary-1b-flash",
        "canary_1b_flash",
        "canary-flash",
        "canary-1b-flash-en",
        "nvidia/canary-1b-flash",
    ] {
        assert_eq!(
            ModelKind::from_arg(alias),
            Some(ModelKind::Canary1bFlash),
            "alias {alias}"
        );
    }
    for alias in ["canary", "canary-1b-v2", "canary-1b-v2-en", "canary-1b_v2"] {
        assert_eq!(ModelKind::from_arg(alias), Some(ModelKind::Canary));
    }
    for invalid in [
        "canary-1b-flash-v2",
        "canary-2b-flash",
        "canary/flash",
        "canary-flash-large",
    ] {
        assert!(ModelKind::from_arg(invalid).is_none(), "invalid {invalid}");
    }
}
