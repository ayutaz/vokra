//! VAST-only Kyutai STT three-artifact identity/schema gate.
//!
//! This ignored test consumes the decoder GGUF, dedicated tokenizer GGUF, and
//! raw Mimi sidecar produced by the VAST runner. It authenticates each whole
//! file before invoking KyutaiSttCompositeBinding. The test proves only
//! component identity, metadata/tensor manifest, tokenizer schema/table
//! digest, and Mimi bytes; it is not PCM, streaming, or numerical parity.

use std::path::{Path, PathBuf};
use std::process::Command;

use vokra_models::kyutai_stt::{
    KYUTAI_STT_MIMI_FILE, KYUTAI_STT_MIMI_SHA256, KyutaiSttCompositeBinding,
};

fn required_paths() -> Option<(PathBuf, PathBuf, PathBuf, String, String, String)> {
    let values = [
        "VOKRA_KYUTAI_STT_DECODER_GGUF",
        "VOKRA_KYUTAI_STT_TOKENIZER_GGUF",
        "VOKRA_KYUTAI_STT_MIMI_FILE",
        "VOKRA_KYUTAI_STT_DECODER_GGUF_SHA256",
        "VOKRA_KYUTAI_STT_TOKENIZER_GGUF_SHA256",
        "VOKRA_KYUTAI_STT_MIMI_SHA256",
    ]
    .map(|name| std::env::var_os(name));
    if values.iter().all(Option::is_none) {
        eprintln!(
            "skipping Kyutai composite bind: set decoder/tokenizer/Mimi paths and all expected SHA-256 values"
        );
        return None;
    }
    let [
        decoder,
        tokenizer,
        mimi,
        decoder_sha,
        tokenizer_sha,
        mimi_sha,
    ] = values;
    Some((
        PathBuf::from(decoder.expect("Kyutai decoder path is required")),
        PathBuf::from(tokenizer.expect("Kyutai tokenizer path is required")),
        PathBuf::from(mimi.expect("Kyutai Mimi path is required")),
        decoder_sha
            .and_then(|value| value.into_string().ok())
            .expect("Kyutai decoder SHA-256 is required"),
        tokenizer_sha
            .and_then(|value| value.into_string().ok())
            .expect("Kyutai tokenizer SHA-256 is required"),
        mimi_sha
            .and_then(|value| value.into_string().ok())
            .expect("Kyutai Mimi SHA-256 is required"),
    ))
}

fn regular_file(path: &Path, label: &str) {
    let metadata = std::fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("{label} cannot be inspected: {error}"));
    assert!(
        metadata.file_type().is_file(),
        "{label} is not a regular file"
    );
    assert!(!metadata.file_type().is_symlink(), "{label} is a symlink");
}

fn sha256_file(path: &Path) -> String {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .unwrap_or_else(|error| panic!("sha256sum failed for {}: {error}", path.display()));
    assert!(
        output.status.success(),
        "sha256sum failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("sha256sum output must be UTF-8")
        .split_whitespace()
        .next()
        .expect("sha256sum output must contain a digest")
        .to_owned()
}

#[test]
#[ignore = "VAST-only: requires the authenticated decoder, tokenizer, and Mimi artifacts"]
fn parity_kyutai_stt_composite_bind_real() {
    let Some((decoder_path, tokenizer_path, mimi_path, decoder_sha, tokenizer_sha, mimi_sha)) =
        required_paths()
    else {
        return;
    };
    regular_file(&decoder_path, "Kyutai decoder GGUF");
    regular_file(&tokenizer_path, "Kyutai tokenizer GGUF");
    regular_file(&mimi_path, "Kyutai Mimi sidecar");
    assert_eq!(
        sha256_file(&decoder_path),
        decoder_sha,
        "decoder GGUF SHA-256 mismatch"
    );
    assert_eq!(
        sha256_file(&tokenizer_path),
        tokenizer_sha,
        "tokenizer GGUF SHA-256 mismatch"
    );
    assert_eq!(
        sha256_file(&mimi_path),
        mimi_sha,
        "Mimi sidecar SHA-256 mismatch"
    );
    assert_eq!(
        mimi_path.file_name().and_then(|name| name.to_str()),
        Some(KYUTAI_STT_MIMI_FILE)
    );
    assert_eq!(
        mimi_sha, KYUTAI_STT_MIMI_SHA256,
        "Mimi SHA-256 is not the pinned identity"
    );

    let decoder = vokra_mmap::open_gguf(&decoder_path).expect("open decoder GGUF through mmap");
    let tokenizer =
        vokra_mmap::open_gguf(&tokenizer_path).expect("open tokenizer GGUF through mmap");
    let mimi_bytes = std::fs::read(&mimi_path).expect("read authenticated Mimi sidecar");
    KyutaiSttCompositeBinding::bind(&decoder, &tokenizer, KYUTAI_STT_MIMI_FILE, &mimi_bytes)
        .expect("strict Kyutai decoder/tokenizer/Mimi identity-schema gate");
    println!(
        "KYUTAI_STT_COMPOSITE_BIND backend=metadata mmap verdict=IDENTITY_SCHEMA_GATE pcm_streaming_parity=BLOCKED"
    );
}
