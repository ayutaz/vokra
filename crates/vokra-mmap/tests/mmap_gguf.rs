//! Integration tests for the true-`mmap` GGUF loader (M1-11b).
//!
//! The central guarantee is **byte-identity**: a GGUF opened through the owned
//! `std::io` path ([`GgufFile::open`]) and through the `mmap` path
//! ([`vokra_mmap::open_gguf`]) must decode to the same version, alignment,
//! metadata and — crucially — the same raw bytes for every tensor. Both paths
//! run the *same* parser in `vokra-core`; this pins that the external
//! (`AsBytes`) provenance changes nothing observable.

use std::path::PathBuf;

use vokra_core::gguf::{GgmlType, GgufBuilder, GgufError, GgufFile};

/// A unique temp path for one test (tests share a pid, so the tag disambiguates).
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("vokra-mmap-it-{tag}-{}.gguf", std::process::id()));
    p
}

/// Builds a small but non-trivial GGUF: several metadata value types plus an
/// F32 and an F16 tensor whose payloads span more than one alignment block.
fn build_sample_gguf() -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "vokra-mmap-test");
    b.add_u32("vokra.test.n_fft", 400);
    b.add_f32("vokra.test.scale", 1.5);
    b.add_bool("vokra.test.flag", true);

    // 6-element F32 tensor (24 bytes).
    let f32_bytes: Vec<u8> = [1.0f32, -2.0, 3.5, 4.25, -5.0, 6.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    b.add_tensor("t.f32", GgmlType::F32, vec![2, 3], f32_bytes)
        .expect("valid f32 tensor");

    // 2-element F16 tensor: 0x3C00 = 1.0, 0x4000 = 2.0 (little-endian bytes).
    let f16_bytes: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40];
    b.add_tensor("t.f16", GgmlType::F16, vec![2], f16_bytes)
        .expect("valid f16 tensor");

    b.to_bytes().expect("serialize gguf")
}

#[test]
fn mmap_and_owned_paths_are_byte_identical() {
    let path = tmp_path("identity");
    std::fs::write(&path, build_sample_gguf()).expect("write temp gguf");

    let owned = GgufFile::open(&path).expect("owned (std::io) path opens");
    let mapped = vokra_mmap::open_gguf(&path).expect("mmap path opens");

    // Header.
    assert_eq!(owned.version(), mapped.version());
    assert_eq!(owned.alignment(), mapped.alignment());

    // Metadata: same keys, same values, same order.
    assert_eq!(owned.metadata(), mapped.metadata());

    // Tensor descriptors and — the load-bearing check — raw payload bytes.
    assert_eq!(owned.tensors().len(), mapped.tensors().len());
    assert!(!owned.tensors().is_empty());
    for info in owned.tensors() {
        let a = owned
            .tensor_data(&info.name)
            .expect("owned tensor bytes present");
        let b = mapped
            .tensor_data(&info.name)
            .expect("mapped tensor bytes present");
        assert_eq!(a, b, "tensor `{}` bytes differ between paths", info.name);
    }

    // Decoded values also match (exercises the dequant path over mapped bytes).
    assert_eq!(
        owned.tensor_f32("t.f32").unwrap(),
        mapped.tensor_f32("t.f32").unwrap()
    );
    assert_eq!(
        owned.tensor_f32("t.f16").unwrap(),
        mapped.tensor_f32("t.f16").unwrap()
    );
    assert_eq!(mapped.tensor_f32("t.f16").unwrap(), vec![1.0, 2.0]);

    std::fs::remove_file(&path).ok();
}

#[test]
fn mapped_bytes_survive_dropping_the_mmap_handle_indirection() {
    // The `GgufFile` owns the boxed `Mmap`; the payload slice must stay valid
    // for as long as the `GgufFile` lives (i.e. the mapping is not released
    // early). Read the bytes into an owned copy and compare to the raw file.
    let path = tmp_path("lifetime");
    let raw = build_sample_gguf();
    std::fs::write(&path, &raw).expect("write temp gguf");

    let mapped = vokra_mmap::open_gguf(&path).expect("mmap path opens");
    let copied: Vec<u8> = mapped.tensor_data("t.f32").unwrap().to_vec();
    // Deleting the file must not affect the already-established mapping.
    std::fs::remove_file(&path).ok();
    let again: Vec<u8> = mapped.tensor_data("t.f32").unwrap().to_vec();
    assert_eq!(copied, again);
    assert_eq!(copied.len(), 24);
}

#[test]
fn empty_file_is_rejected() {
    let path = tmp_path("empty");
    std::fs::write(&path, b"").expect("write empty file");

    // High-level entry point: surfaces as GgufError::Io.
    let err = vokra_mmap::open_gguf(&path).unwrap_err();
    assert!(matches!(err, GgufError::Io(_)), "got {err:?}");

    // Low-level entry point: InvalidInput (mmap rejects a zero length).
    let err = vokra_mmap::Mmap::open(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

    std::fs::remove_file(&path).ok();
}

#[test]
fn nonexistent_path_is_io_error() {
    let missing = "/no/such/vokra/mmap/file.gguf";

    let err = vokra_mmap::open_gguf(missing).unwrap_err();
    assert!(matches!(err, GgufError::Io(_)), "got {err:?}");

    let err = vokra_mmap::Mmap::open(missing).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn malformed_gguf_surfaces_a_parse_error_not_a_panic() {
    // A non-empty, non-GGUF file must map fine but fail parsing with BadMagic
    // (proves the mmap path reaches the shared parser and reports cleanly).
    let path = tmp_path("badmagic");
    std::fs::write(&path, b"NOTGGUF-more-bytes-to-be-non-empty").expect("write");

    let err = vokra_mmap::open_gguf(&path).unwrap_err();
    assert!(matches!(err, GgufError::BadMagic(_)), "got {err:?}");

    std::fs::remove_file(&path).ok();
}

/// Compile-time proof that a mapping-backed handle stays thread-shareable, so a
/// `GgufFile` built from it can live in a `Send + Sync` `Session`.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<vokra_mmap::Mmap>();
    assert_send_sync::<GgufFile>();
};
