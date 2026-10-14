//! Gated real-public-GGUF parity against the pinned official X-Codec2 decoder.
//!
//! The model-free fixture contract test authenticates the historical fixture.
//! The real decoder leg additionally requires the audited public GGUF artifact;
//! an unset artifact is a documented skip, never a fabricated pass.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::json::{self, JsonValue};
use vokra_models::xcodec2::XCodec2;

const FP32_MAX_ABS: f32 = 2.0e-4;
const FP32_RMSE: f64 = 2.0e-5;
const COSINE_MIN: f64 = 0.999_999;
const GGUF_BYTES: u64 = 3_291_064_672;
const GGUF_SHA256: &str = "7ab4b94006068226b0741930081f7e149316e045511c1cddb94769e7f598698e";
const MANIFEST_BYTES: u64 = 1_212;
const MANIFEST_SHA256: &str = "cfe60f214df9229b0f519d0548f0ce14cabe810df680d8eddb5d5e80420eb2ba";
const CODES_SHA256: &str = "3bab5738a2c63b11352ad1211664346c1f1ff19de67bd750e1b8a61408c353ea";
const DECODED_SHA256: &str = "fac4ac18b9781df6934fbe99f3b38a8eb32522b3883935c0950463f7af6a8ceb";
const FEATURES_SHA256: &str = "acf8f5ce29a035eebb121517f85b252ddd80f86c6b9d6d1794ae6b21a5c38c4d";
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/xcodec2")
}

// Zero-dependency streaming SHA-256 (FIPS 180-4).
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
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa81a664b, 0xc24b8b70,
            0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
            0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee,
            0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];
        let mut w = [0u32; 64];
        for (i, c) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
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
            let n = (64 - self.used).min(bytes.len());
            self.block[self.used..self.used + n].copy_from_slice(&bytes[..n]);
            self.used += n;
            bytes = &bytes[n..];
            if self.used == 64 {
                let b = self.block;
                self.compress(&b);
                self.used = 0;
            }
        }
    }
    fn finish(mut self) -> String {
        let bits = self.len * 8;
        self.update(&[0x80]);
        let z = [0u8; 64];
        if self.used > 56 {
            self.update(&z[..64 - self.used]);
        }
        if self.used < 56 {
            self.update(&z[..56 - self.used]);
        }
        self.update(&bits.to_be_bytes());
        self.h.iter().map(|v| format!("{v:08x}")).collect()
    }
}
fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finish()
}

fn safe_regular(path: &Path) {
    assert!(
        path.is_absolute(),
        "path must be absolute: {}",
        path.display()
    );
    let mut prefix = PathBuf::new();
    for component in path.components() {
        match component {
            Component::RootDir => prefix.push(std::path::MAIN_SEPARATOR_STR),
            Component::Normal(v) => prefix.push(v),
            Component::CurDir | Component::ParentDir => panic!("unsafe path"),
            // Do not stat a drive-relative `C:` prefix.  The following
            // RootDir component completes it as `C:\\` first.
            Component::Prefix(v) => {
                prefix.push(v.as_os_str());
                continue;
            }
        }
        let m = fs::symlink_metadata(&prefix)
            .unwrap_or_else(|e| panic!("stat {}: {e}", prefix.display()));
        assert!(
            !m.file_type().is_symlink(),
            "symlink path component: {}",
            prefix.display()
        );
    }
    let m = fs::symlink_metadata(path).unwrap_or_else(|e| panic!("stat {}: {e}", path.display()));
    assert!(m.is_file(), "not a regular file: {}", path.display());
}
fn safe_directory(path: &Path) {
    assert!(
        path.is_absolute(),
        "path must be absolute: {}",
        path.display()
    );
    let mut prefix = PathBuf::new();
    for component in path.components() {
        match component {
            Component::RootDir => prefix.push(std::path::MAIN_SEPARATOR_STR),
            Component::Normal(v) => prefix.push(v),
            Component::CurDir | Component::ParentDir => panic!("unsafe path"),
            Component::Prefix(v) => {
                prefix.push(v.as_os_str());
                continue;
            }
        }
        let m = fs::symlink_metadata(&prefix)
            .unwrap_or_else(|e| panic!("stat {}: {e}", prefix.display()));
        assert!(
            !m.file_type().is_symlink(),
            "symlink path component: {}",
            prefix.display()
        );
    }
    let m = fs::symlink_metadata(path).unwrap_or_else(|e| panic!("stat {}: {e}", path.display()));
    assert!(m.is_dir(), "not a directory: {}", path.display());
}
fn authenticated(path: &Path, size: u64, max: u64, want: &str) -> Vec<u8> {
    safe_regular(path);
    let m = fs::metadata(path).unwrap();
    assert_eq!(m.len(), size);
    assert!(size <= max);
    assert!(
        size <= usize::MAX as u64,
        "authenticated file is too large for this platform"
    );
    let mut f = File::open(path).unwrap();
    // The bytes hashed below are read from this one opened handle and the same
    // owned buffer is passed to GgufFile::parse. This authenticates the parsed
    // bytes; portable pathname replacement atomicity is not claimed.
    let opened = f.metadata().unwrap();
    assert!(
        opened.is_file(),
        "opened artifact is not regular: {}",
        path.display()
    );
    assert_eq!(
        opened.len(),
        size,
        "opened artifact size changed: {}",
        path.display()
    );
    let mut out = Vec::with_capacity(size as usize);
    let mut h = Sha256::new();
    let mut buf = [0u8; 64 << 10];
    loop {
        let n = f.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        assert!((out.len() as u64).saturating_add(n as u64) <= max);
        h.update(&buf[..n]);
        out.extend_from_slice(&buf[..n]);
    }
    assert_eq!(out.len() as u64, size);
    assert_eq!(h.finish(), want);
    out
}

fn object<'a>(value: &'a JsonValue, want: &[&str], label: &str) -> &'a [(String, JsonValue)] {
    let entries = value
        .as_object()
        .unwrap_or_else(|| panic!("{label} object"));
    assert_eq!(entries.len(), want.len(), "{label} key count");
    let expected: BTreeSet<&str> = want.iter().copied().collect();
    let actual: BTreeSet<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(actual, expected, "{label} keys");
    entries
}
fn field<'a>(entries: &'a [(String, JsonValue)], key: &str, label: &str) -> &'a JsonValue {
    entries
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
        .unwrap_or_else(|| panic!("{label} missing {key}"))
}
fn string_field(e: &[(String, JsonValue)], k: &str, w: &str, l: &str) {
    assert_eq!(field(e, k, l).as_str(), Some(w), "{l}.{k}");
}
fn int_field(e: &[(String, JsonValue)], k: &str, w: i64, l: &str) {
    assert_eq!(field(e, k, l), &JsonValue::Int(w), "{l}.{k}");
}
fn shape_field(e: &[(String, JsonValue)], k: &str, w: &[i64], l: &str) {
    let a = field(e, k, l).as_array().unwrap();
    assert_eq!(a.len(), w.len());
    for (v, n) in a.iter().zip(w) {
        assert_eq!(v, &JsonValue::Int(*n));
    }
}
fn validate_manifest(bytes: &[u8]) {
    let v = json::parse(bytes).unwrap();
    let e = object(
        &v,
        &[
            "code_count",
            "decoded_shape",
            "decoder_source_sha256",
            "feature_shape",
            "files",
            "format",
            "gguf_sha256",
            "official_defaulted_deterministic_buffers",
            "official_state_tensors_loaded",
            "oracle",
            "source_distribution",
            "source_distribution_sha256",
            "torch",
            "torchtune",
            "torchtune_rope_sha256",
            "transformer_source_sha256",
            "vector_quantize_pytorch",
        ],
        "manifest",
    );
    int_field(e, "code_count", 4, "manifest");
    shape_field(e, "decoded_shape", &[1, 1, 1280], "manifest");
    shape_field(e, "feature_shape", &[1, 1024, 4], "manifest");
    int_field(e, "official_state_tensors_loaded", 117, "manifest");
    assert_eq!(
        field(e, "official_defaulted_deterministic_buffers", "manifest"),
        &JsonValue::Array(Vec::new())
    );
    string_field(e, "format", "vokra-xcodec2-reference-v1", "manifest");
    string_field(
        e,
        "oracle",
        "official xcodec2==0.1.5 CodecDecoderVocos FSQ + forward",
        "manifest",
    );
    string_field(e, "source_distribution", "xcodec2==0.1.5", "manifest");
    string_field(
        e,
        "source_distribution_sha256",
        "dc1a73b32090706e65fb73b2469411bc27bb72048677a23b430ab21ad325e45b",
        "manifest",
    );
    string_field(
        e,
        "decoder_source_sha256",
        "8a770d35c4d90a3a82b38869b7b39bd6fab6ab7b2079a44915c7740549f19282",
        "manifest",
    );
    string_field(
        e,
        "transformer_source_sha256",
        "54786751f363ed6ea510c7a4a13d5c093cd392f79c545ce31dedcd745d6662d0",
        "manifest",
    );
    string_field(
        e,
        "torchtune_rope_sha256",
        "8d79a03e1334fe6ecaff14b1e6a2d554e7e6209c95db058846973de013f92b80",
        "manifest",
    );
    string_field(e, "torch", "2.5.0+cu124", "manifest");
    string_field(e, "torchtune", "0.3.1", "manifest");
    string_field(e, "vector_quantize_pytorch", "1.17.8", "manifest");
    string_field(e, "gguf_sha256", GGUF_SHA256, "manifest");
    let f = object(
        field(e, "files", "manifest"),
        &["codes.u32le", "decoded_pcm.f32", "features.f32"],
        "manifest.files",
    );
    string_field(f, "codes.u32le", CODES_SHA256, "files");
    string_field(f, "decoded_pcm.f32", DECODED_SHA256, "files");
    string_field(f, "features.f32", FEATURES_SHA256, "files");
}

fn f32s(bytes: &[u8], n: usize, label: &str) -> Vec<f32> {
    assert_eq!(bytes.len(), n * 4, "{label} length");
    let out: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(out.iter().all(|v| v.is_finite()), "{label} non-finite");
    out
}
fn u32s(bytes: &[u8], n: usize) -> Vec<u32> {
    assert_eq!(bytes.len(), n * 4);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
struct Packet {
    codes: Vec<u32>,
    expected: Vec<f32>,
}
fn validate_fixture(dir: &Path) -> Packet {
    safe_directory(dir);
    let names: BTreeSet<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    let expected: [&str; 5] = [
        "README.md",
        "manifest.json",
        "codes.u32le",
        "decoded_pcm.f32",
        "features.f32",
    ];
    assert_eq!(names, expected.into_iter().map(str::to_owned).collect());
    safe_regular(&dir.join("README.md"));
    let manifest = authenticated(
        &dir.join("manifest.json"),
        MANIFEST_BYTES,
        MANIFEST_BYTES,
        MANIFEST_SHA256,
    );
    validate_manifest(&manifest);
    let codes = authenticated(&dir.join("codes.u32le"), 16, 1 << 20, CODES_SHA256);
    let decoded = authenticated(&dir.join("decoded_pcm.f32"), 5120, 1 << 20, DECODED_SHA256);
    let features = authenticated(&dir.join("features.f32"), 16384, 1 << 20, FEATURES_SHA256);
    let codes = u32s(&codes, 4);
    assert!(codes.iter().all(|v| *v < 65_536));
    let expected = f32s(&decoded, 1280, "decoded_pcm.f32");
    let _ = f32s(&features, 4096, "features.f32");
    Packet { codes, expected }
}
fn parse_backend(v: Option<&str>) -> BackendKind {
    match v {
        Some("metal") => BackendKind::Metal,
        Some("cpu") | None => BackendKind::Cpu,
        Some(x) => panic!("invalid backend {x:?}"),
    }
}
fn parse_backend_os(v: Option<std::ffi::OsString>) -> BackendKind {
    match v {
        None => parse_backend(None),
        Some(v) => match v.into_string() {
            Ok(v) => parse_backend(Some(&v)),
            Err(_) => panic!("VOKRA_XCODEC2_BACKEND must be valid UTF-8"),
        },
    }
}
fn selected_backend() -> BackendKind {
    parse_backend_os(std::env::var_os("VOKRA_XCODEC2_BACKEND"))
}

#[test]
fn historical_fixture_contract_is_authenticated() {
    let _ = validate_fixture(&fixture_root());
}
#[test]
fn sha256_known_vectors_and_chunk_boundary() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let nist = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    assert_eq!(nist.len(), 56);
    for split in [63, 64, 65] {
        let mut hash = Sha256::new();
        for chunk in nist.chunks(split) {
            hash.update(chunk);
        }
        assert_eq!(
            hash.finish(),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }
    let d = temp("sha");
    let p = d.join("payload");
    let b = vec![0x5a; (1 << 20) + 17];
    fs::write(&p, &b).unwrap();
    for split in [63, 64, 65] {
        let mut hash = Sha256::new();
        for chunk in b.chunks(split) {
            hash.update(chunk);
        }
        assert_eq!(
            hash.finish(),
            "ee00e469f2bdf447766b815707f434911e270ae9f5e77f3a8688a03b3026216f"
        );
    }
    assert_eq!(
        authenticated(
            &p,
            b.len() as u64,
            b.len() as u64,
            "ee00e469f2bdf447766b815707f434911e270ae9f5e77f3a8688a03b3026216f"
        ),
        b
    );
    fs::remove_dir_all(d).unwrap();
}
#[test]
fn fixture_negative_contracts_fail_closed() {
    let original = fs::read(fixture_root().join("manifest.json")).unwrap();
    assert_panics(|| validate_manifest(b"{}"));
    let mut dup = b"{\"code_count\":4,".to_vec();
    dup.extend_from_slice(&original[1..]);
    assert_panics(|| validate_manifest(&dup));
    for (old, new) in [
        (
            b"\"code_count\": 4".as_slice(),
            b"\"code_count\": \"4\"".as_slice(),
        ),
        (
            b"\"source_distribution\": \"xcodec2==0.1.5\"",
            b"\"source_distribution\": \"xcodec2==0.1.6\"",
        ),
        (b"\"torch\": \"2.5.0+cu124\"", b"\"torch\": \"2.6.0\""),
    ] {
        assert_panics(|| validate_manifest(&replace_once(&original, old, new)));
    }
    let shape = replace_once(
        &original,
        b"\"decoded_shape\": [\n    1,\n    1,\n    1280\n  ]",
        b"\"decoded_shape\": [1, 2]",
    );
    assert_panics(|| validate_manifest(&shape));
    let files_duplicate = replace_once(
        &original,
        b"\"files\": {",
        b"\"files\": {\"codes.u32le\": \"duplicate\",",
    );
    assert_panics(|| validate_manifest(&files_duplicate));
    let crlf = original
        .split(|byte| *byte == b'\n')
        .enumerate()
        .flat_map(|(index, line)| {
            let mut out = line.to_vec();
            if index + 1 < original.split(|byte| *byte == b'\n').count() {
                out.push(b'\r');
            }
            out.push(b'\n');
            out
        })
        .collect::<Vec<_>>();
    let d = copy_fixture();
    fs::write(d.join("manifest.json"), crlf).unwrap();
    assert_panics(|| validate_fixture(&d));
    fs::remove_dir_all(d).unwrap();

    let d = copy_fixture();
    let mut b = fs::read(d.join("codes.u32le")).unwrap();
    b[0] ^= 1;
    fs::write(d.join("codes.u32le"), b).unwrap();
    assert_panics(|| validate_fixture(&d));
    fs::remove_dir_all(d).unwrap();

    let d = copy_fixture();
    fs::write(d.join("codes.u32le"), [0u8; 15]).unwrap();
    assert_panics(|| validate_fixture(&d));
    fs::remove_dir_all(d).unwrap();

    let d = copy_fixture();
    fs::remove_file(d.join("features.f32")).unwrap();
    assert_panics(|| validate_fixture(&d));
    fs::remove_dir_all(d).unwrap();

    let d = copy_fixture();
    let mut b = fs::read(d.join("decoded_pcm.f32")).unwrap();
    b[0] ^= 1;
    fs::write(d.join("decoded_pcm.f32"), b).unwrap();
    assert_panics(|| validate_fixture(&d));
    fs::remove_dir_all(d).unwrap();

    let d = copy_fixture();
    let mut b = fs::read(d.join("features.f32")).unwrap();
    b[0] ^= 1;
    fs::write(d.join("features.f32"), b).unwrap();
    assert_panics(|| validate_fixture(&d));
    fs::remove_dir_all(d).unwrap();

    let d = copy_fixture();
    let mut b = fs::read(d.join("manifest.json")).unwrap();
    b.push(b'\n');
    fs::write(d.join("manifest.json"), b).unwrap();
    assert_panics(|| validate_fixture(&d));
    fs::remove_dir_all(d).unwrap();

    let d = copy_fixture();
    fs::write(d.join("extra.bin"), b"unexpected").unwrap();
    assert_panics(|| validate_fixture(&d));
    fs::remove_dir_all(d).unwrap();

    let d = copy_fixture();
    fs::remove_file(d.join("manifest.json")).unwrap();
    assert_panics(|| validate_fixture(&d));
    fs::remove_dir_all(d).unwrap();
}

fn replace_once(bytes: &[u8], old: &[u8], new: &[u8]) -> Vec<u8> {
    let start = bytes
        .windows(old.len())
        .position(|window| window == old)
        .expect("mutation marker present");
    let mut result = Vec::with_capacity(bytes.len() + new.len() - old.len());
    result.extend_from_slice(&bytes[..start]);
    result.extend_from_slice(new);
    result.extend_from_slice(&bytes[start + old.len()..]);
    result
}
#[test]
fn fixture_payload_semantics_and_paths_fail_closed() {
    let nan = f32::NAN.to_le_bytes();
    let infinity = f32::INFINITY.to_le_bytes();
    assert_eq!(f32s(&1.0f32.to_le_bytes(), 1, "finite"), vec![1.0]);
    assert_panics(|| f32s(&nan, 1, "nan"));
    assert_panics(|| f32s(&infinity, 1, "infinity"));
    assert_panics(|| f32s(&[0; 4], 2, "short"));
    assert_panics(|| u32s(&[0; 1], 1));
    assert_panics(|| authenticated(Path::new("relative"), 0, 0, ""));
    let d = temp("auth");
    let p = d.join("payload");
    fs::write(&p, b"abc").unwrap();
    assert_panics(|| authenticated(&d.join("missing"), 0, 0, ""));
    assert_panics(|| {
        authenticated(
            &p,
            4,
            4,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
    });
    assert_panics(|| {
        authenticated(
            &p,
            3,
            3,
            "0000000000000000000000000000000000000000000000000000000000000000",
        )
    });
    let nonregular = d.join("directory");
    fs::create_dir(&nonregular).unwrap();
    assert_panics(|| authenticated(&nonregular, 0, 0, ""));
    fs::remove_dir_all(d).unwrap();
    assert_eq!(parse_backend(Some("cpu")), BackendKind::Cpu);
    assert_eq!(parse_backend(Some("metal")), BackendKind::Metal);
    assert_panics(|| parse_backend(Some("cuda")));
}
#[test]
#[cfg(unix)]
fn backend_rejects_non_utf8_environment_value() {
    use std::os::unix::ffi::OsStringExt;
    assert_panics(|| parse_backend_os(Some(std::ffi::OsString::from_vec(vec![0xff]))));
}
#[test]
#[cfg(unix)]
fn fixture_symlink_and_nonregular_paths_fail_closed() {
    use std::os::unix::fs::symlink;
    let parent = temp("symlink");
    let d = parent.join("fixture");
    fs::create_dir(&d).unwrap();
    populate_fixture(&d);
    let _ = validate_fixture(&d);
    let p = d.join("codes.u32le");
    let real = parent.join("codes.real");
    fs::copy(&p, &real).unwrap();
    fs::remove_file(&p).unwrap();
    symlink(&real, &p).unwrap();
    assert_panics(|| validate_fixture(&d));
    fs::remove_dir_all(parent).unwrap();
}

#[test]
fn fixture_nonregular_path_fails_closed() {
    let d = copy_fixture();
    let p = d.join("codes.u32le");
    fs::remove_file(&p).unwrap();
    fs::create_dir(&p).unwrap();
    assert_panics(|| validate_fixture(&d));
    fs::remove_dir_all(d).unwrap();
}

#[test]
#[cfg(unix)]
fn fixture_ancestor_symlink_fails_closed() {
    use std::os::unix::fs::symlink;
    let base = temp("ancestor");
    let real_parent = base.join("real-parent");
    let real_fixture = real_parent.join("fixture");
    fs::create_dir(&real_parent).unwrap();
    fs::create_dir(&real_fixture).unwrap();
    populate_fixture(&real_fixture);
    let alias = base.join("alias-parent");
    symlink(&real_parent, &alias).unwrap();
    assert_panics(|| validate_fixture(&alias.join("fixture")));
    fs::remove_dir_all(base).unwrap();
}

#[test]
#[cfg(unix)]
fn gguf_authentication_rejects_symlink() {
    use std::os::unix::fs::symlink;
    let d = temp("gguf");
    let real = d.join("real.gguf");
    let link = d.join("link.gguf");
    fs::write(&real, b"GGUF").unwrap();
    symlink(&real, &link).unwrap();
    assert_panics(|| authenticated(&link, 4, 4, &sha256_hex(b"GGUF")));
    fs::remove_dir_all(d).unwrap();
}
fn assert_panics(f: impl FnOnce() + std::panic::UnwindSafe) {
    assert!(std::panic::catch_unwind(f).is_err());
}
fn temp(label: &str) -> PathBuf {
    let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let base = fs::canonicalize(std::env::temp_dir()).expect("canonicalize test temp root");
    for attempt in 0..128 {
        let p = base.join(format!(
            "vokra-xcodec2-{label}-{}-{serial}-{attempt}",
            std::process::id()
        ));
        match fs::create_dir(&p) {
            Ok(()) => return p,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!("create owned test directory {}: {error}", p.display()),
        }
    }
    panic!("unable to allocate unique owned test directory");
}
fn copy_fixture() -> PathBuf {
    let d = temp("fixture");
    populate_fixture(&d);
    d
}
fn populate_fixture(d: &Path) {
    for n in [
        "README.md",
        "manifest.json",
        "codes.u32le",
        "decoded_pcm.f32",
        "features.f32",
    ] {
        fs::copy(fixture_root().join(n), d.join(n)).unwrap();
    }
}

#[test]
fn real_xcodec2_decode_matches_official() {
    let packet = validate_fixture(&fixture_root());
    let codes = packet.codes;
    let expected = packet.expected;
    let Some(path) = std::env::var_os("VOKRA_XCODEC2_GGUF") else {
        eprintln!("skipping X-Codec2 real parity: set VOKRA_XCODEC2_GGUF");
        return;
    };
    let backend = selected_backend();
    let bytes = authenticated(&PathBuf::from(path), GGUF_BYTES, GGUF_BYTES, GGUF_SHA256);
    let file = GgufFile::parse(bytes).expect("parse authenticated public X-Codec2 GGUF");
    let model = XCodec2::from_gguf(&file)
        .expect("strict public X-Codec2 bind")
        .with_backend(backend);
    let actual = model.decode_codes(&codes).expect("X-Codec2 decode");
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    for (&left, &right) in actual.iter().zip(&expected) {
        let left = f64::from(left);
        let right = f64::from(right);
        let delta = left - right;
        max_abs = max_abs.max(delta.abs() as f32);
        squared_error += delta * delta;
        dot += left * right;
        actual_norm += left * left;
        expected_norm += right * right;
    }
    let rmse = (squared_error / actual.len() as f64).sqrt();
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
    eprintln!(
        "X-Codec2 {backend:?}: codes={}, samples={}, max_abs={max_abs:.9e}, rmse={rmse:.9e}, cosine={cosine:.12}",
        codes.len(),
        actual.len()
    );
    assert!(
        max_abs <= FP32_MAX_ABS,
        "X-Codec2 {backend:?} max_abs {max_abs} exceeds {FP32_MAX_ABS}"
    );
    assert!(
        rmse <= FP32_RMSE,
        "X-Codec2 {backend:?} RMSE {rmse} exceeds {FP32_RMSE}"
    );
    assert!(
        cosine >= COSINE_MIN,
        "X-Codec2 {backend:?} cosine {cosine} is below {COSINE_MIN}"
    );
}
