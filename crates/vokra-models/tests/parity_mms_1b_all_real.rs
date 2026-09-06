//! MMS-1B-All real-artifact gate.
//!
//! The current public Vokra artifact is adapter-only. Keep this test loud
//! until a VAST-produced full backbone + selected language adapter artifact
//! and official oracle fixture have been audited.

use std::path::Path;
use std::{
    fs,
    io::{Read, Write},
    time::{SystemTime, UNIX_EPOCH},
};
use vokra_models::wav2vec2_ctc::Wav2Vec2Ctc;

// Zero-dependency streaming FIPS 180-4 SHA-256. The ignored MMS test may
// authenticate multi-gigabyte artifacts on macOS, where sha256sum is absent.
struct Sha256 {
    h: [u32; 8],
    len: u64,
    block: [u8; 64],
    used: usize,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            len: 0,
            block: [0; 64],
            used: 0,
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let a = w[i - 15];
            let b = w[i - 2];
            w[i] = w[i - 16]
                .wrapping_add(a.rotate_right(7) ^ a.rotate_right(18) ^ (a >> 3))
                .wrapping_add(w[i - 7])
                .wrapping_add(b.rotate_right(17) ^ b.rotate_right(19) ^ (b >> 10));
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) = (
            self.h[0], self.h[1], self.h[2], self.h[3], self.h[4], self.h[5], self.h[6], self.h[7],
        );
        for i in 0..64 {
            let t1 = hh
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ ((!e) & g))
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
                .wrapping_add((a & b) ^ (a & c) ^ (b & c));
            (hh, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
        }
        for (dst, add) in self.h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *dst = dst.wrapping_add(add);
        }
    }

    fn update(&mut self, mut bytes: &[u8]) {
        self.len = self.len.wrapping_add(bytes.len() as u64);
        while !bytes.is_empty() {
            let take = (64 - self.used).min(bytes.len());
            self.block[self.used..self.used + take].copy_from_slice(&bytes[..take]);
            self.used += take;
            bytes = &bytes[take..];
            if self.used == 64 {
                let block = self.block;
                self.compress(&block);
                self.used = 0;
            }
        }
    }

    fn finish(mut self) -> String {
        let bits = self.len * 8;
        self.update(&[0x80]);
        let zero = [0u8; 64];
        if self.used > 56 {
            self.update(&zero[..64 - self.used]);
        }
        if self.used < 56 {
            self.update(&zero[..56 - self.used]);
        }
        self.update(&bits.to_be_bytes());
        self.h.iter().map(|value| format!("{value:08x}")).collect()
    }
}

fn sha256_file(path: &Path) -> String {
    let mut file =
        fs::File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 1 << 20];
    loop {
        let count = file
            .read(&mut buffer)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        if count == 0 {
            return hash.finish();
        }
        hash.update(&buffer[..count]);
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(bytes);
    hash.finish()
}

struct TempPath(std::path::PathBuf);

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn valid_language(value: &str) -> bool {
    !value.is_empty()
        && value
            .split(|byte| byte == '-' || byte == '_')
            .all(|segment| {
                !segment.is_empty()
                    && segment
                        .bytes()
                        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
            })
}

#[test]
fn mms_sha256_known_vectors_and_chunk_boundary() {
    assert!(valid_language("eng"));
    assert!(valid_language("azj-script_cyrillic"));
    for invalid in ["", "-eng", "eng-", "eng--foo", "eng__foo", "ENG", "eng."] {
        assert!(
            !valid_language(invalid),
            "accepted invalid language: {invalid}"
        );
    }
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let bytes = vec![0x5au8; (1 << 20) + 17];
    let path = std::env::temp_dir().join(format!(
        "vokra-mms-sha256-{}-{}-{}",
        std::process::id(),
        bytes.len(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_nanos()
    ));
    let _guard = TempPath(path.clone());
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("create SHA self-test fixture exclusively");
    file.write_all(&bytes).expect("write SHA self-test fixture");
    assert_eq!(sha256_file(&path), sha256_hex(&bytes));
}

fn require_regular_input(path: &str, label: &str) {
    let candidate = Path::new(path);
    assert!(candidate.is_absolute(), "{label} must be an absolute path");
    assert!(
        !candidate.components().any(|component| {
            matches!(
                component,
                std::path::Component::CurDir | std::path::Component::ParentDir
            )
        }),
        "{label} must not contain dot path components"
    );
    let metadata = std::fs::symlink_metadata(candidate)
        .unwrap_or_else(|error| panic!("{label} is unreadable: {error}"));
    assert!(
        metadata.file_type().is_file(),
        "{label} must be a regular file"
    );
    let mut parent = candidate.parent();
    while let Some(directory) = parent {
        let metadata = std::fs::symlink_metadata(directory)
            .unwrap_or_else(|error| panic!("{label} ancestry is unreadable: {error}"));
        assert!(
            !metadata.file_type().is_symlink(),
            "{label} has symlinked ancestry"
        );
        parent = directory.parent();
    }
}

#[test]
#[ignore = "requires authenticated VAST-produced full MMS backbone+adapter artifact"]
fn mms_1b_all_stays_blocked_until_full_manifest() {
    let path = std::env::var_os("VOKRA_MMS_1B_ALL_GGUF")
        .expect("VOKRA_MMS_1B_ALL_GGUF is required; refusing an implicit inspection skip");
    let language =
        std::env::var("VOKRA_MMS_1B_ALL_LANGUAGE").expect("VOKRA_MMS_1B_ALL_LANGUAGE is required");
    assert!(
        valid_language(&language),
        "MMS language adapter code must match [a-z0-9]+([_-][a-z0-9]+)*"
    );
    let manifest = std::env::var("VOKRA_MMS_1B_ALL_REFERENCE_MANIFEST")
        .expect("VOKRA_MMS_1B_ALL_REFERENCE_MANIFEST is required");
    let approval = std::env::var("VOKRA_MMS_1B_ALL_APPROVAL_EVIDENCE")
        .expect("VOKRA_MMS_1B_ALL_APPROVAL_EVIDENCE is required");
    let expected_head =
        std::env::var("VOKRA_EXPECTED_COMMIT").expect("VOKRA_EXPECTED_COMMIT is required");
    let reference_digest = std::env::var("VOKRA_MMS_1B_ALL_REFERENCE_MANIFEST_SHA256")
        .expect("VOKRA_MMS_1B_ALL_REFERENCE_MANIFEST_SHA256 is required");
    let approval_digest = std::env::var("VOKRA_MMS_1B_ALL_APPROVAL_EVIDENCE_SHA256")
        .expect("VOKRA_MMS_1B_ALL_APPROVAL_EVIDENCE_SHA256 is required");
    assert!(!manifest.is_empty() && !approval.is_empty());
    require_regular_input(
        Path::new(&path)
            .to_str()
            .expect("GGUF path must be valid UTF-8"),
        "MMS GGUF",
    );
    require_regular_input(&manifest, "MMS reference manifest");
    require_regular_input(&approval, "MMS approval evidence");
    assert_eq!(
        sha256_file(Path::new(&manifest)),
        reference_digest,
        "MMS reference manifest digest mismatch"
    );
    assert_eq!(
        sha256_file(Path::new(&approval)),
        approval_digest,
        "MMS approval evidence digest mismatch"
    );
    if let Ok(gguf_digest) = std::env::var("VOKRA_MMS_1B_ALL_GGUF_SHA256") {
        assert!(
            gguf_digest.len() == 64
                && gguf_digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "MMS GGUF digest must be exactly 64 lowercase hexadecimal characters"
        );
        assert_eq!(
            sha256_file(Path::new(&path)),
            gguf_digest,
            "MMS GGUF digest mismatch"
        );
    }
    for (label, value) in [
        ("reference manifest digest", &reference_digest),
        ("approval evidence digest", &approval_digest),
    ] {
        assert!(
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "{label} must be exactly 64 lowercase hexadecimal characters"
        );
    }
    assert!(
        expected_head.len() == 40
            && expected_head
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "expected Vokra HEAD must be exactly 40 lowercase hexadecimal characters"
    );
    let error = Wav2Vec2Ctc::from_gguf(Path::new(&path)).expect_err(
        "adapter-only or unaudited MMS artifact must not bind as a complete checkpoint",
    );
    let message = error.to_string();
    assert!(
        message.contains("inspection-only"),
        "unexpected MMS result: {message}"
    );
    assert!(
        message.contains("CC-BY-NC-4.0"),
        "license gate missing: {message}"
    );
    assert!(
        message.contains("load_adapter"),
        "adapter contract missing: {message}"
    );
    eprintln!(
        "MMS_1B_ALL BLOCKED_PENDING_AUTHENTICATED_MANIFEST: {language}, reference={manifest}, approval={approval}; no CPU/Metal parity measured"
    );
}
