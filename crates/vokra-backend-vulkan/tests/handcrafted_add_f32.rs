//! M3-02-T14 partial / T24 — pin the `add_f32` hand-crafted SPIR-V module's
//! **structural invariants** at integration-test level, from the *outside*
//! (through the public `spirv` module surface). Complements the unit tests
//! inside `handcrafted_add_f32.rs` by asserting the manifest side of the
//! same invariants — a rename / mis-registration would silently break the
//! T24 eval-op arm without this test.

use vokra_backend_vulkan::spirv;

/// The public re-export path is stable — downstream code (integration tests,
/// benchmarks) reaches `handcrafted_add_f32::{SPIRV_MODULE, LOCAL_SIZE_X,
/// BINDING_COUNT, bytes}` through `vokra_backend_vulkan::spirv`. Failing this
/// test means the path changed.
#[test]
fn public_module_surface_is_stable() {
    // Basic module accessibility.
    assert!(!spirv::handcrafted_add_f32::SPIRV_MODULE.is_empty());
    // Consumer-visible constants.
    assert_eq!(spirv::handcrafted_add_f32::LOCAL_SIZE_X, 64);
    assert_eq!(spirv::handcrafted_add_f32::BINDING_COUNT, 3);
}

/// The SPIR-V module header decodes to the little-endian magic word and
/// SPIR-V 1.3 version literal that any conforming driver expects (spec §2.3).
/// A malformed header is the fastest way for a driver to reject the module,
/// so pin the first two words as a mirror of the unit test inside the module.
#[test]
fn bytecode_starts_with_valid_spirv_header() {
    let blob = spirv::handcrafted_add_f32::bytes();
    // Length is 4-byte aligned (spec §2.3).
    assert_eq!(blob.len() % 4, 0);
    // Reasonable size: 172 words × 4 bytes = 688 bytes. Allow a wide window
    // so hand-tweaks that add a decoration or a debug op don't force this
    // test to move too, but a wildly different size is a bug.
    assert!(
        (400..=1200).contains(&blob.len()),
        "add_f32 SPIR-V size {} outside expected 400..=1200 byte window",
        blob.len()
    );
    // Magic + version literals from the SPIR-V spec §2.3.
    let magic = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
    assert_eq!(magic, 0x0723_0203, "SPIR-V magic mismatch");
    let version = u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]);
    assert_eq!(
        version, 0x0001_0300,
        "SPIR-V version must be 1.3 (Vulkan 1.1 minimum)"
    );
}

/// `bytes()` is deterministic — repeated calls return the same buffer with
/// the same content. If the underlying re-encoder ever grew a stateful bug
/// (e.g. cached a stale buffer between calls), this test surfaces it.
#[test]
fn bytes_is_deterministic() {
    let a = spirv::handcrafted_add_f32::bytes();
    let b = spirv::handcrafted_add_f32::bytes();
    let c = spirv::handcrafted_add_f32::bytes();
    assert_eq!(a, b);
    assert_eq!(b, c);
}

/// `bytes()` re-encodes the `const [u32]` module into little-endian bytes,
/// regardless of host endianness (SPIR-V spec §2.3 defines the on-disk
/// format as little-endian). Verify by walking chunks of 4 bytes and
/// comparing against the `SPIRV_MODULE` slice.
#[test]
fn bytes_are_little_endian_encoding_of_spirv_module() {
    let bytes = spirv::handcrafted_add_f32::bytes();
    let module = spirv::handcrafted_add_f32::SPIRV_MODULE;
    assert_eq!(bytes.len(), module.len() * 4);
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        assert_eq!(
            word, module[i],
            "word {i} did not round-trip through little-endian bytes",
        );
    }
}

/// The manifest side of `add_f32`: an entry exists, is `Handcrafted`, and
/// carries a pinned SHA-256 that `load_spv_owned` reproduces. A rename that
/// only touches `SHADERS` (without matching `load_spv_owned`) would leave a
/// dead entry — this test catches it.
#[test]
fn manifest_registers_add_f32_as_handcrafted_and_pinned() {
    let entry = spirv::SHADERS
        .iter()
        .find(|s| s.name == "add_f32")
        .expect("SHADERS manifest must include `add_f32`");
    assert!(
        matches!(entry.variant, spirv::ShaderVariant::Handcrafted),
        "add_f32 must be Handcrafted, was {}",
        entry.variant
    );
    assert!(
        entry.expected_sha256_hex.is_some(),
        "add_f32 must have a pinned SHA-256 (test verifies at build time)"
    );
    // `load_spv_owned("add_f32")` must return the runtime bytes.
    let blob = spirv::load_spv_owned("add_f32").expect("load_spv_owned should reach add_f32");
    assert!(!blob.is_empty());
}

/// The manifest ships **exactly two** Handcrafted entries (ADR §5): the T13
/// `copy_f32` and the T24 `add_f32`. Adding a third silently would defeat
/// the ADR cap — this test blocks it. If a real reason emerges to add
/// another hand-crafted kernel, update the ADR + this test in the same PR.
#[test]
fn manifest_caps_handcrafted_entries_at_two() {
    let hc: Vec<&str> = spirv::SHADERS
        .iter()
        .filter(|s| matches!(s.variant, spirv::ShaderVariant::Handcrafted))
        .map(|s| s.name)
        .collect();
    assert_eq!(
        hc.len(),
        2,
        "ADR M3-02-spirv-generation §5 caps hand-authored kernels at 2. Found {hc:?}."
    );
    // Both names present.
    assert!(hc.contains(&"copy_f32"));
    assert!(hc.contains(&"add_f32"));
}
