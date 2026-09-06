//! Real-weight CosyVoice2 HiFT CPU/Metal parity worker for Apple Silicon.
//!
//! This ignored test is intentionally fail-closed. It consumes only an
//! authenticated packet produced by the independent upstream PyTorch dumper;
//! the shell worker validates the target and invokes this exact test in
//! release mode. It never downloads, uploads, or executes the reference model.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use vokra_core::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::json::{self, JsonValue};
use vokra_models::cosyvoice2::HiFTChain;

const ATOL: f32 = 0.01;
const SOURCE_REPOSITORY: &str = "https://github.com/FunAudioLLM/CosyVoice.git";
const MODEL_REVISION: &str = "eec1ae6c79877dbd9379285cf8789c9e0879293d";
const MODEL_REPOSITORY: &str = "FunAudioLLM/CosyVoice2-0.5B";
const MODEL_PATH: &str = "hift.pt";
const MODEL_BYTES: u64 = 83_390_254;
const MODEL_SHA256: &str = "3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879";
const SOURCE_REVISION: &str = "8555549e882236e6541748b1042d95693caa82ba";
const CONFIG_PATH: &str = "cosyvoice2.yaml";
const CONFIG_BYTES: u64 = 7_330;
const CONFIG_SHA256: &str = "0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959";
const CONFIG_BLOB_SHA1: &str = "bc19267bbfd373c9a760b7667a74349ddd487db1";
const TENSOR_MANIFEST_SHA256: &str =
    "cecbb2d68f91337f263db0f0333c75573516e7087b6e75d6ea647b3f86afec7c";
const GGUF_ENV: &str = "VOKRA_COSYVOICE2_HIFT_GGUF";
const GGUF_SHA_ENV: &str = "VOKRA_COSYVOICE2_HIFT_GGUF_SHA256";
const REFERENCE_ENV: &str = "VOKRA_COSYVOICE2_HIFT_REFERENCE_DIR";
const REFERENCE_SHA_ENV: &str = "VOKRA_COSYVOICE2_HIFT_REFERENCE_MANIFEST_SHA256";
const LICENSE_ENV: &str = "VOKRA_COSYVOICE2_HIFT_LICENSE_MANIFEST";
const LICENSE_SHA_ENV: &str = "VOKRA_COSYVOICE2_HIFT_LICENSE_MANIFEST_SHA256";
const EVIDENCE_ENV: &str = "VOKRA_COSYVOICE2_HIFT_APPLE_EVIDENCE_DIR";
const SOURCE_ROLES: &[(&str, &str)] = &[
    (
        "cosyvoice/hifigan/generator.py",
        "326a1a70ae7707662939c20493b3a8e4b0906216",
    ),
    (
        "cosyvoice/hifigan/f0_predictor.py",
        "5797c31aada757ac7ef65a70ff8ee21867a25df8",
    ),
    (
        "cosyvoice/transformer/activation.py",
        "8cea54816385d3b6585ccc2417bc71630d578177",
    ),
    (
        "cosyvoice/utils/common.py",
        "6f5a3dd8b7ae99601783c3a4ed91b3b64270fab3",
    ),
];

fn has_lexical_dot_component(raw: &str) -> bool {
    raw.split('/')
        .any(|component| matches!(component, "." | ".."))
}

fn env_path(name: &str) -> PathBuf {
    let raw = std::env::var_os(name).unwrap_or_else(|| panic!("{name} is required"));
    let raw = raw
        .to_str()
        .unwrap_or_else(|| panic!("{name} must be UTF-8"));
    assert!(
        !raw.is_empty() && raw.starts_with('/'),
        "{name} must be absolute"
    );
    assert!(
        !has_lexical_dot_component(raw),
        "{name} must not contain lexical dot path components"
    );
    let path = PathBuf::from(raw);
    assert!(
        path.is_absolute(),
        "{name} must remain absolute after lexical validation"
    );
    let mut current = path.as_path();
    loop {
        assert!(
            !current.is_symlink(),
            "{name} has symlink ancestry: {}",
            current.display()
        );
        if current == Path::new("/") {
            break;
        }
        current = current
            .parent()
            .unwrap_or_else(|| panic!("{name} has no parent"));
    }
    path
}

fn file(name: &str) -> PathBuf {
    let path = env_path(name);
    assert!(
        path.is_file() && !path.is_symlink(),
        "{name} must be a regular non-symlink file: {}",
        path.display()
    );
    path
}

fn directory(name: &str) -> PathBuf {
    let path = env_path(name);
    assert!(
        path.is_dir() && !path.is_symlink(),
        "{name} must be a regular non-symlink directory: {}",
        path.display()
    );
    path
}

fn canonical_for_scope(path: &Path) -> PathBuf {
    if path.exists() {
        path.canonicalize()
            .unwrap_or_else(|e| panic!("canonicalize {}: {e}", path.display()))
    } else {
        let parent = path
            .parent()
            .unwrap_or_else(|| panic!("{} has no parent", path.display()));
        assert!(
            parent.is_dir() && !parent.is_symlink(),
            "scope parent must be an existing non-symlink directory: {}",
            parent.display()
        );
        parent
            .canonicalize()
            .unwrap_or_else(|e| panic!("canonicalize {}: {e}", parent.display()))
            .join(path.file_name().unwrap())
    }
}

fn require_disjoint(paths: &[(&Path, &str)]) {
    for (index, (left, left_label)) in paths.iter().enumerate() {
        let left = canonical_for_scope(left);
        for (right, right_label) in paths.iter().skip(index + 1) {
            let right = canonical_for_scope(right);
            assert!(
                !(left == right || left.starts_with(&right) || right.starts_with(&left)),
                "{left_label} and {right_label} paths overlap"
            );
        }
    }
}

// Exact FIPS-180-4 SHA-256 implementation copied from the established real
// HiFT CPU parity contract. No digest dependency is added to the runtime.
fn sha256(data: &[u8]) -> String {
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
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut p = data.to_vec();
    let bits = (p.len() as u64) * 8;
    p.push(0x80);
    while p.len() % 64 != 56 {
        p.push(0);
    }
    p.extend_from_slice(&bits.to_be_bytes());
    for block in p.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).take(16).enumerate() {
            w[i] = u32::from_be_bytes(chunk.try_into().unwrap());
        }
        for i in 16..64 {
            let a = w[i - 15];
            let b = w[i - 2];
            w[i] = w[i - 16]
                .wrapping_add(a.rotate_right(7) ^ a.rotate_right(18) ^ (a >> 3))
                .wrapping_add(w[i - 7])
                .wrapping_add(b.rotate_right(17) ^ b.rotate_right(19) ^ (b >> 10));
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut x) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let t1 = x
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ (!e & g))
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
                .wrapping_add((a & b) ^ (a & c) ^ (b & c));
            (x, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
        }
        for (dst, add) in h.iter_mut().zip([a, b, c, d, e, f, g, x]) {
            *dst = dst.wrapping_add(add);
        }
    }
    h.iter().map(|value| format!("{value:08x}")).collect()
}

struct StreamSha256 {
    h: [u32; 8],
    len: u64,
    block: [u8; 64],
    used: usize,
}

impl StreamSha256 {
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
        for (i, c) in block.chunks_exact(4).take(16).enumerate() {
            w[i] = u32::from_be_bytes(c.try_into().unwrap());
        }
        for i in 16..64 {
            let a = w[i - 15];
            let b = w[i - 2];
            w[i] = w[i - 16]
                .wrapping_add(a.rotate_right(7) ^ a.rotate_right(18) ^ (a >> 3))
                .wrapping_add(w[i - 7])
                .wrapping_add(b.rotate_right(17) ^ b.rotate_right(19) ^ (b >> 10));
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut x) = (
            self.h[0], self.h[1], self.h[2], self.h[3], self.h[4], self.h[5], self.h[6], self.h[7],
        );
        for i in 0..64 {
            let t1 = x
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ (!e & g))
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
                .wrapping_add((a & b) ^ (a & c) ^ (b & c));
            (x, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
        }
        for (dst, add) in self.h.iter_mut().zip([a, b, c, d, e, f, g, x]) {
            *dst = dst.wrapping_add(add);
        }
    }
    fn update(&mut self, mut input: &[u8]) {
        self.len = self.len.wrapping_add(input.len() as u64);
        while !input.is_empty() {
            let take = (64 - self.used).min(input.len());
            self.block[self.used..self.used + take].copy_from_slice(&input[..take]);
            self.used += take;
            input = &input[take..];
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
        let zeros = [0u8; 64];
        if self.used > 56 {
            self.update(&zeros[..64 - self.used]);
        }
        if self.used < 56 {
            self.update(&zeros[..56 - self.used]);
        }
        self.update(&bits.to_be_bytes());
        self.h.iter().map(|v| format!("{v:08x}")).collect()
    }
}

fn sha256_file(path: &Path) -> String {
    let mut file = fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut digest = StreamSha256::new();
    let mut buffer = [0u8; 1 << 20];
    loop {
        let count = file
            .read(&mut buffer)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    digest.finish()
}
fn field<'a>(root: &'a JsonValue, path: &[&str]) -> &'a JsonValue {
    path.iter().fold(root, |value, key| {
        value
            .get(key)
            .unwrap_or_else(|| panic!("manifest missing {}", path.join(".")))
    })
}
fn text(root: &JsonValue, path: &[&str]) -> String {
    field(root, path)
        .as_str()
        .unwrap_or_else(|| panic!("manifest {} is not a string", path.join(".")))
        .to_owned()
}
fn integer(root: &JsonValue, path: &[&str]) -> u64 {
    field(root, path)
        .as_u64()
        .unwrap_or_else(|| panic!("manifest {} is not an integer", path.join(".")))
}
fn reject_duplicate_keys(value: &JsonValue) {
    if let JsonValue::Object(entries) = value {
        let mut keys = std::collections::BTreeSet::new();
        for (key, child) in entries {
            assert!(keys.insert(key), "duplicate JSON key: {key}");
            reject_duplicate_keys(child);
        }
    } else if let JsonValue::Array(items) = value {
        for item in items {
            reject_duplicate_keys(item);
        }
    }
}
fn exact_keys(value: &JsonValue, expected: &[&str], label: &str) {
    let entries = value
        .as_object()
        .unwrap_or_else(|| panic!("{label} must be an object"));
    assert_eq!(entries.len(), expected.len(), "{label} schema drift");
    for key in expected {
        assert!(
            entries.iter().any(|(actual, _)| actual == key),
            "{label} missing key {key}"
        );
    }
}
fn artifact_record<'a>(record: &'a JsonValue, label: &str) -> &'a JsonValue {
    field(record, &[label])
}
fn hash(value: &str, label: &str) {
    assert_eq!(value.len(), 64, "{label} must be SHA-256 hex");
    assert!(
        value
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f')),
        "{label} must be lowercase hex"
    );
}

fn authenticated_license_manifest(path: &Path) -> (String, JsonValue) {
    let expected =
        std::env::var(LICENSE_SHA_ENV).unwrap_or_else(|_| panic!("{LICENSE_SHA_ENV} is required"));
    hash(&expected, LICENSE_SHA_ENV);
    let bytes = fs::read(path).expect("external owner-signed license manifest bytes");
    assert_eq!(sha256(&bytes), expected, "external license manifest digest");
    let manifest = json::parse(&bytes).expect("external owner-signed license manifest JSON");
    reject_duplicate_keys(&manifest);
    (expected, manifest)
}

fn f32_artifact(
    path: &Path,
    expected_bytes: usize,
    expected: &[usize],
    record: &JsonValue,
    label: &str,
) -> Vec<f32> {
    assert!(
        !path.is_symlink() && path.is_file(),
        "{label} must be a regular file"
    );
    let raw = fs::read(path).expect("read reference artifact");
    assert_eq!(raw.len(), expected_bytes, "{label} byte count");
    let artifact_record = artifact_record(record, label);
    if label == "input" {
        exact_keys(
            artifact_record,
            &[
                "file", "dtype", "shape", "bytes", "sha256", "seed", "formula",
            ],
            "reference input record",
        );
    } else {
        exact_keys(
            artifact_record,
            &["file", "dtype", "shape", "bytes", "sha256"],
            "reference output record",
        );
    }
    assert_eq!(
        text(artifact_record, &["file"]),
        path.file_name().unwrap().to_string_lossy()
    );
    assert_eq!(text(artifact_record, &["dtype"]), "F32");
    assert_eq!(integer(artifact_record, &["bytes"]), expected_bytes as u64);
    let shape = field(artifact_record, &["shape"])
        .as_array()
        .expect("artifact shape");
    assert_eq!(
        shape
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>(),
        expected
    );
    let digest = text(artifact_record, &["sha256"]);
    hash(&digest, label);
    assert_eq!(sha256(&raw), digest, "{label} SHA-256");
    let values = raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(
        values.iter().all(|v| v.is_finite()) && values.iter().any(|v| *v != 0.0),
        "{label} is non-finite or vacuous"
    );
    values
}
fn compare(label: &str, actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let (index, diff) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(i, (a, b))| (i, (a - b).abs()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    eprintln!("CosyVoice2 HiFT Apple {label}: max_abs={diff:.9e} at {index}");
    assert!(diff <= ATOL, "{label} exceeds fixed atol {ATOL}: {diff}");
    diff
}
#[test]
#[ignore]
fn cosyvoice2_hift_apple_cpu_metal_parity() {
    assert_eq!(
        std::env::consts::OS,
        "macos",
        "Apple worker refuses non-macOS hosts"
    );
    assert_eq!(
        std::env::consts::ARCH,
        "aarch64",
        "Apple worker refuses non-arm64 hosts"
    );
    assert_eq!(
        std::env::var("VOKRA_REMOTE_APPLE_SILICON").as_deref(),
        Ok("1"),
        "Apple parity requires the explicit remote Apple worker gate"
    );
    let gguf = file(GGUF_ENV);
    let license = file(LICENSE_ENV);
    let (license_sha, license_json) = authenticated_license_manifest(&license);
    let reference = directory(REFERENCE_ENV);
    let evidence = env_path(EVIDENCE_ENV);
    assert!(
        !evidence.exists() && !evidence.is_symlink(),
        "evidence directory must be absent"
    );
    let evidence_parent = evidence
        .parent()
        .unwrap_or_else(|| panic!("{} has no parent", evidence.display()));
    assert!(
        evidence_parent.is_dir() && !evidence_parent.is_symlink(),
        "evidence parent must be an existing non-symlink directory: {}",
        evidence_parent.display()
    );
    require_disjoint(&[
        (&gguf, "GGUF"),
        (&reference, "reference"),
        (&license, "license"),
        (&evidence, "evidence"),
        (
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."),
            "checkout",
        ),
    ]);
    let mut entries = fs::read_dir(&reference)
        .expect("read reference packet")
        .map(|entry| entry.expect("reference entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    assert_eq!(
        entries
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect::<Vec<_>>(),
        vec![
            "f0.f32".to_owned(),
            "manifest.json".to_owned(),
            "mel.f32".to_owned(),
            "pcm.f32".to_owned(),
        ]
    );
    for entry in &entries {
        assert!(
            entry.is_file() && !entry.is_symlink(),
            "reference entries must be regular files"
        );
    }
    let manifest_path = reference.join("manifest.json");
    let manifest = json::parse(&fs::read(&manifest_path).expect("read reference manifest"))
        .expect("parse reference manifest");
    reject_duplicate_keys(&manifest);
    exact_keys(
        &manifest,
        &[
            "format",
            "status",
            "publication",
            "source",
            "model",
            "config",
            "license_manifest_sha256",
            "approval_scope_sha256",
            "project_sha256",
            "uv_lock_sha256",
            "checkpoint_tensor_manifest_sha256",
            "input",
            "outputs",
            "execution",
        ],
        "reference manifest",
    );
    exact_keys(
        field(&manifest, &["source"]),
        &["repository", "revision", "clean", "roles"],
        "reference source",
    );
    exact_keys(
        field(&manifest, &["model"]),
        &["repository", "revision", "path", "bytes", "sha256"],
        "reference model",
    );
    exact_keys(
        field(&manifest, &["config"]),
        &["path", "bytes", "sha256", "git_blob_sha1"],
        "reference config",
    );
    exact_keys(
        field(&manifest, &["outputs"]),
        &["f0", "pcm"],
        "reference outputs",
    );
    exact_keys(
        field(&manifest, &["execution"]),
        &[
            "device",
            "torch_deterministic_algorithms",
            "threads",
            "entropy_override",
            "python",
            "torch",
            "numpy",
            "scipy",
            "uv_lock_sha256",
        ],
        "reference execution",
    );
    assert_eq!(
        text(&manifest, &["source", "repository"]),
        SOURCE_REPOSITORY
    );
    assert_eq!(text(&manifest, &["source", "revision"]), SOURCE_REVISION);
    assert!(matches!(
        field(&manifest, &["source", "clean"]),
        JsonValue::Bool(true)
    ));
    let source_roles = field(&manifest, &["source", "roles"])
        .as_object()
        .expect("reference source roles");
    assert_eq!(source_roles.len(), SOURCE_ROLES.len());
    for (role, blob) in SOURCE_ROLES {
        let role_record = field(&manifest, &["source", "roles", role]);
        exact_keys(
            role_record,
            &["git_blob_sha1", "sha256"],
            "reference source role",
        );
        assert_eq!(text(role_record, &["git_blob_sha1"]), *blob);
        hash(&text(role_record, &["sha256"]), role);
    }
    assert_eq!(text(&manifest, &["model", "repository"]), MODEL_REPOSITORY);
    assert_eq!(text(&manifest, &["model", "path"]), MODEL_PATH);
    assert_eq!(integer(&manifest, &["model", "bytes"]), MODEL_BYTES);
    assert_eq!(text(&manifest, &["config", "path"]), CONFIG_PATH);
    assert_eq!(integer(&manifest, &["config", "bytes"]), CONFIG_BYTES);
    assert_eq!(text(&manifest, &["execution", "device"]), "cpu");
    assert!(matches!(
        field(&manifest, &["execution", "torch_deterministic_algorithms"]),
        JsonValue::Bool(true)
    ));
    assert_eq!(integer(&manifest, &["execution", "threads"]), 1);
    assert_eq!(
        text(&manifest, &["execution", "entropy_override"]),
        "torch.rand and torch.randn_like -> zeros only during official forward"
    );
    assert!(text(&manifest, &["execution", "python"]).starts_with("3.12."));
    assert_eq!(text(&manifest, &["execution", "torch"]), "2.7.1+cpu");
    assert_eq!(text(&manifest, &["execution", "numpy"]), "2.3.5");
    assert_eq!(text(&manifest, &["execution", "scipy"]), "1.16.3");
    assert_eq!(
        text(&manifest, &["execution", "uv_lock_sha256"]),
        text(&manifest, &["uv_lock_sha256"])
    );
    assert_eq!(
        text(&manifest, &["format"]),
        "vokra-cosyvoice2-hift-reference-v1"
    );
    assert_eq!(text(&manifest, &["status"]), "AUTHENTICATED_REFERENCE");
    assert_eq!(text(&manifest, &["publication"]), "NO_UPLOAD");
    assert_eq!(text(&manifest, &["model", "revision"]), MODEL_REVISION);
    assert_eq!(text(&manifest, &["model", "sha256"]), MODEL_SHA256);
    assert_eq!(text(&manifest, &["config", "sha256"]), CONFIG_SHA256);
    assert_eq!(
        text(&manifest, &["config", "git_blob_sha1"]),
        CONFIG_BLOB_SHA1
    );
    assert_eq!(text(&manifest, &["source", "revision"]), SOURCE_REVISION);
    assert_eq!(
        text(&manifest, &["checkpoint_tensor_manifest_sha256"]),
        TENSOR_MANIFEST_SHA256
    );
    let reference_sha = std::env::var(REFERENCE_SHA_ENV)
        .unwrap_or_else(|_| panic!("{REFERENCE_SHA_ENV} is required"));
    hash(&reference_sha, REFERENCE_SHA_ENV);
    assert_eq!(
        sha256_file(&manifest_path),
        reference_sha,
        "reference manifest digest"
    );
    assert_eq!(license_sha, text(&manifest, &["license_manifest_sha256"]));
    exact_keys(
        &license_json,
        &[
            "gate_version",
            "status",
            "owner_signoff",
            "publication",
            "source",
            "model",
            "config",
            "python_closure",
            "evidence",
            "decision",
            "approval",
        ],
        "license manifest",
    );
    assert_eq!(text(&license_json, &["status"]), "APPROVED");
    assert_eq!(text(&license_json, &["owner_signoff"]), "OWNER_SIGNED_OFF");
    assert_eq!(text(&license_json, &["publication"]), "NO_UPLOAD");
    assert_eq!(
        text(&license_json, &["decision"]),
        "DO_NOT_DISTRIBUTE_UNTIL_OWNER_SIGNOFF"
    );
    assert_eq!(
        text(&license_json, &["approval", "scope_sha256"]),
        text(&manifest, &["approval_scope_sha256"])
    );
    let gguf_sha =
        std::env::var(GGUF_SHA_ENV).unwrap_or_else(|_| panic!("{GGUF_SHA_ENV} is required"));
    hash(&gguf_sha, GGUF_SHA_ENV);
    assert_eq!(sha256_file(&gguf), gguf_sha, "GGUF digest");
    let record = field(&manifest, &["outputs"]);
    let mel = f32_artifact(
        &reference.join("mel.f32"),
        2560,
        &[1, 80, 8],
        &manifest,
        "input",
    );
    let f0 = f32_artifact(&reference.join("f0.f32"), 32, &[1, 8], record, "f0");
    let pcm = f32_artifact(&reference.join("pcm.f32"), 15360, &[1, 3840], record, "pcm");
    assert_eq!(integer(&manifest, &["input", "seed"]), 20260906);
    let file = GgufFile::open(&gguf).expect("open authenticated GGUF");
    let cpu = HiFTChain::from_gguf(&file).expect("bind CPU HiFT");
    let f0_cpu = cpu.f0_predictor_forward(&mel, 8).expect("CPU F0");
    let f0_diff = compare("F0 CPU/reference", &f0_cpu, &f0);
    let pcm_cpu = cpu
        .forward_with_backend(&mel, 8, BackendKind::Cpu)
        .expect("CPU PCM");
    let cpu_diff = compare("PCM CPU/reference", &pcm_cpu, &pcm);
    let metal = HiFTChain::from_gguf(&file).expect("bind Metal HiFT");
    let pcm_metal = metal
        .forward_with_backend(&mel, 8, BackendKind::Metal)
        .expect("actual Metal HiFT; no fallback or skip");
    let metal_ref_diff = compare("PCM Metal/reference", &pcm_metal, &pcm);
    let metal_cpu_diff = compare("PCM Metal/CPU", &pcm_metal, &pcm_cpu);
    // HiFTChain's public Metal seam returns PCM and computes F0 internally;
    // F0 is exposed only through the CPU predictor seam. The full resident
    // Metal forward above nevertheless exercises the same F0 stage.
    let device = Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-detailLevel", "mini"])
        .output()
        .expect("query Metal device");
    assert!(device.status.success());
    assert!(!device.stdout.is_empty(), "Metal device evidence is empty");
    let device_evidence = "system_profiler:success";
    fs::create_dir(&evidence).expect("claim absent evidence directory");
    let out = evidence.join("evidence.json");
    let payload = format!(
        "{{\n  \"format\": \"vokra-cosyvoice2-hift-apple-evidence-v1\",\n  \"status\": \"PASS\",\n  \"publication\": \"NO_UPLOAD\",\n  \"backend\": \"CPU+Metal\",\n  \"device_evidence\": \"{}\",\n  \"gguf_sha256\": \"{}\",\n  \"reference_manifest_sha256\": \"{}\",\n  \"license_manifest_sha256\": \"{}\",\n  \"atol\": {},\n  \"f0_cpu_reference_max_abs\": {},\n  \"pcm_cpu_reference_max_abs\": {},\n  \"pcm_metal_reference_max_abs\": {},\n  \"pcm_metal_cpu_max_abs\": {},\n  \"scope\": {{\"component\":\"standalone_cosyvoice2_hift\",\"f0\":\"CPU/reference only; Metal F0 not separately exposed\",\"pcm\":\"CPU/reference, Metal/reference, Metal/CPU\",\"full_cosyvoice2_e2e\":\"NOT_CLAIMED\"}}\n}}\n",
        device_evidence,
        gguf_sha,
        reference_sha,
        license_sha,
        ATOL,
        f0_diff,
        cpu_diff,
        metal_ref_diff,
        metal_cpu_diff
    );
    let mut stream = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)
        .expect("create evidence atomically");
    stream
        .write_all(payload.as_bytes())
        .expect("write evidence");
    stream.sync_all().expect("sync evidence");
}

#[cfg(test)]
mod hash_vectors {
    #[test]
    fn artifact_schema_is_read_from_named_child_record() {
        let manifest = super::json::parse(
            r#"{"outputs":{"f0":{"file":"f0.f32","dtype":"F32","shape":[1,8],"bytes":32,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"},"pcm":{"file":"pcm.f32","dtype":"F32","shape":[1,3840],"bytes":15360,"sha256":"1111111111111111111111111111111111111111111111111111111111111111"}}}"#,
        )
        .unwrap();
        let outputs = super::field(&manifest, &["outputs"]);
        let f0 = super::artifact_record(outputs, "f0");
        super::exact_keys(f0, &["file", "dtype", "shape", "bytes", "sha256"], "f0");
        assert_eq!(super::text(f0, &["file"]), "f0.f32");
        assert!(outputs.get("file").is_none());
    }

    #[test]
    fn lexical_path_components_are_detected_before_path_normalization() {
        assert!(!super::has_lexical_dot_component("/tmp/reference"));
        assert!(super::has_lexical_dot_component("/tmp/./reference"));
        assert!(super::has_lexical_dot_component("/tmp/../reference"));
    }

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            super::sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            super::sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut streaming = super::StreamSha256::new();
        streaming.update(b"abc");
        assert_eq!(
            streaming.finish(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn streaming_file_hash_matches_chunked_hash() {
        use std::io::Write;
        let root = std::env::temp_dir().join(format!(
            "vokra-hift-apple-hash-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("fixture.bin");
        let mut writer = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let data: Vec<u8> = (0..(1 << 20) + 17)
            .map(|i| (i as u8).wrapping_mul(31))
            .collect();
        writer.write_all(&data).unwrap();
        writer.sync_all().unwrap();
        assert_eq!(super::sha256_file(&path), super::sha256(&data));
        std::fs::remove_dir_all(root).unwrap();
    }
}
