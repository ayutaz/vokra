//! VAST-only CosyVoice2 LLM parity against the official Transformers fixture.
//!
//! This integration test is ignored by default because opening the standalone
//! GGUF reads its multi-gigabyte payload. When opted in, every input is
//! authenticated: the reference manifest, fixed token IDs, artifact digests,
//! and the exact Qwen2 axes are all required before comparing logits.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::{
    fs,
    io::{Read, Write},
    time::{SystemTime, UNIX_EPOCH},
};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, chunks};
use vokra_models::cosyvoice2::llm::{LlmBackboneConfig, parity};
use vokra_models::cosyvoice2::{CosyVoice2Config, LlmBackbone};

const GGUF_ENV: &str = "VOKRA_COSYVOICE2_LLM_COMPONENT_GGUF";
const REFERENCE_ENV: &str = "VOKRA_COSYVOICE2_LLM_REFERENCE";
const GGUF_SHA_ENV: &str = "VOKRA_COSYVOICE2_LLM_COMPONENT_GGUF_SHA256";
const REFERENCE_MANIFEST_SHA_ENV: &str = "VOKRA_COSYVOICE2_LLM_REFERENCE_MANIFEST_SHA256";
const LICENSE_MANIFEST_ENV: &str = "VOKRA_COSYVOICE2_LLM_LICENSE_MANIFEST";
const LICENSE_MANIFEST_SHA_ENV: &str = "VOKRA_COSYVOICE2_LLM_LICENSE_MANIFEST_SHA256";
const MODEL_REPOSITORY: &str = "FunAudioLLM/CosyVoice2-0.5B";
const MODEL_REVISION: &str = "eec1ae6c79877dbd9379285cf8789c9e0879293d";
const MODEL_PATH: &str = "llm.pt";
const MODEL_BYTES: u64 = 2_023_316_821;
const MODEL_SHA256: &str = "b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608";
const MANIFEST_SHA256: &str = "07cf10ae088c27a7c88e1c08fb231d00b01bba0c13f312a74d2fd4b35403bda2";
const SOURCE_REPOSITORY: &str = "https://github.com/FunAudioLLM/CosyVoice.git";
const TENSOR_COUNT: usize = 295;
const ATOL: f32 = 3e-4;
const TOKEN_IDS: &[u32] = &[
    151_643, 785, 3_974, 13_876, 38_835, 34_208, 916, 279, 15_678, 5_562, 13,
];

fn required_file(path: &Path, label: &str) {
    assert!(path.is_absolute(), "{label} must be an absolute path");
    for component in path.components() {
        assert!(
            !matches!(component, Component::CurDir | Component::ParentDir),
            "{label} contains a lexical dot component: {path:?}"
        );
    }
    for ancestor in path.ancestors().skip(1) {
        if let Ok(metadata) = std::fs::symlink_metadata(ancestor) {
            assert!(
                !metadata.file_type().is_symlink(),
                "{label} has a symlinked ancestor: {ancestor:?}"
            );
        }
    }
    let metadata = std::fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("{label} is unreadable: {error}"));
    assert!(
        metadata.file_type().is_file(),
        "{label} must be a regular file"
    );
    assert!(
        !metadata.file_type().is_symlink(),
        "{label} must not be a symlink"
    );
}

fn required_directory(path: &Path, label: &str) {
    assert!(path.is_absolute(), "{label} must be an absolute path");
    for component in path.components() {
        assert!(
            !matches!(component, Component::CurDir | Component::ParentDir),
            "{label} contains a lexical dot component: {path:?}"
        );
    }
    assert!(
        path.is_dir() && !path.is_symlink(),
        "{label} must be a real directory"
    );
    for ancestor in path.ancestors() {
        let metadata = fs::symlink_metadata(ancestor)
            .unwrap_or_else(|error| panic!("{label} is unreadable: {error}"));
        assert!(
            !metadata.file_type().is_symlink() && metadata.file_type().is_dir(),
            "{label} has symlinked/non-directory ancestry: {ancestor:?}"
        );
    }
}

fn required_path(name: &str) -> PathBuf {
    let value = std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"));
    reject_raw_dot_components(&value, name);
    PathBuf::from(value)
}

fn reject_raw_dot_components(value: &str, label: &str) {
    assert!(value.starts_with('/'), "{label} must be an absolute path");
    for component in value.split('/') {
        assert!(
            component != "." && component != "..",
            "{label} contains a raw lexical dot component"
        );
    }
}

#[test]
fn required_path_rejects_raw_dot_components() {
    assert!(
        std::panic::catch_unwind(|| reject_raw_dot_components("/tmp/./fixture", "fixture"))
            .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| reject_raw_dot_components("/tmp/../fixture", "fixture"))
            .is_err()
    );
    reject_raw_dot_components("/tmp//fixture", "fixture");
}

fn required_sha256(name: &str) -> String {
    let value = std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"));
    assert!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "{name} must be lowercase SHA-256 hex"
    );
    value
}

fn json_string<'a>(value: &'a vokra_core::json::JsonValue, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(vokra_core::json::JsonValue::as_str)
        .unwrap_or_else(|| panic!("manifest key {key} is missing or not a string"))
}

fn json_u64(value: &vokra_core::json::JsonValue, key: &str) -> u64 {
    value
        .get(key)
        .and_then(vokra_core::json::JsonValue::as_u64)
        .unwrap_or_else(|| panic!("manifest key {key} is missing or not UINT64"))
}

fn json_bool(value: &vokra_core::json::JsonValue, key: &str) -> bool {
    match value.get(key) {
        Some(vokra_core::json::JsonValue::Bool(value)) => *value,
        _ => panic!("manifest key {key} is missing or not BOOL"),
    }
}

fn json_f64(value: &vokra_core::json::JsonValue, key: &str) -> f64 {
    match value.get(key) {
        Some(vokra_core::json::JsonValue::Float(number)) => *number,
        Some(vokra_core::json::JsonValue::Int(number)) => *number as f64,
        _ => panic!("manifest key {key} is missing or not a number"),
    }
}

fn json_object<'a>(
    value: &'a vokra_core::json::JsonValue,
    key: &str,
) -> &'a vokra_core::json::JsonValue {
    value
        .get(key)
        .and_then(|entry| entry.as_object().map(|_| entry))
        .unwrap_or_else(|| panic!("manifest key {key} is missing or not an object"))
}

fn exact_object_keys(value: &vokra_core::json::JsonValue, expected: &[&str], label: &str) {
    let object = value
        .as_object()
        .unwrap_or_else(|| panic!("{label} is not an object"));
    assert_eq!(object.len(), expected.len(), "{label} key count");
    for key in expected {
        assert!(
            object.iter().any(|(actual, _)| actual == key),
            "{label} key {key}"
        );
    }
}

fn reject_duplicate_json_keys(bytes: &[u8], label: &str) {
    let mut objects = Vec::<HashSet<String>>::new();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                let start = index + 1;
                index = start;
                while index < bytes.len() {
                    match bytes[index] {
                        b'\\' => index += 2,
                        b'"' => break,
                        _ => index += 1,
                    }
                }
                assert!(index < bytes.len(), "{label} has unterminated JSON string");
                let raw = String::from_utf8_lossy(&bytes[start..index]).into_owned();
                index += 1;
                while index < bytes.len() && bytes[index].is_ascii_whitespace() {
                    index += 1;
                }
                if index < bytes.len() && bytes[index] == b':' {
                    if let Some(keys) = objects.last_mut() {
                        assert!(keys.insert(raw), "{label} contains a duplicate JSON key");
                    }
                }
            }
            b'{' => {
                objects.push(HashSet::new());
                index += 1;
            }
            b'}' => {
                assert!(
                    objects.pop().is_some(),
                    "{label} has malformed JSON objects"
                );
                index += 1;
            }
            _ => index += 1,
        }
    }
    assert!(objects.is_empty(), "{label} has unclosed JSON object");
}

// Zero-dependency FIPS 180-4 SHA-256. The real-weight test is also run on
// macOS, where the GNU-only sha256sum executable is not guaranteed. Keep
// file hashing streaming: the staged GGUF can be multi-gigabyte.
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
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut x) = (
            self.h[0], self.h[1], self.h[2], self.h[3], self.h[4], self.h[5], self.h[6], self.h[7],
        );
        for i in 0..64 {
            let t1 = x
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ ((!e) & g))
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
        self.h.iter().map(|v| format!("{v:08x}")).collect()
    }
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(data);
    hash.finish()
}

fn sha256_file(path: &Path) -> String {
    let mut file = fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut hash = Sha256::new();
    let mut buf = [0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    hash.finish()
}

#[test]
fn sha256_known_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn sha256_file_streaming_crosses_chunk_boundary() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "vokra-cosyvoice2-llm-sha256-{}-{nonce}",
        std::process::id()
    ));
    let data = (0..(1 << 20) + 123)
        .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
        .collect::<Vec<_>>();
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .expect("create SHA-256 streaming fixture without clobbering");
    struct Cleanup<'a>(&'a Path);
    impl Drop for Cleanup<'_> {
        fn drop(&mut self) {
            let _ = fs::remove_file(self.0);
        }
    }
    let _cleanup = Cleanup(&path);
    file.write_all(&data)
        .expect("write SHA-256 streaming fixture");
    assert_eq!(sha256_file(&path), sha256_hex(&data));
}

fn authenticate_external_license_manifest(path: &Path) {
    required_file(path, "external owner-signed license manifest");
    let expected_sha =
        std::env::var(LICENSE_MANIFEST_SHA_ENV).expect("license manifest SHA-256 is required");
    assert_eq!(expected_sha.len(), 64);
    assert!(
        expected_sha
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    );
    let manifest_bytes = fs::read(path).expect("external owner-signed license manifest bytes");
    assert_eq!(
        sha256_hex(&manifest_bytes),
        expected_sha,
        "external license manifest digest"
    );
    reject_duplicate_json_keys(&manifest_bytes, "external owner-signed license manifest");
    let manifest = vokra_core::json::parse(&manifest_bytes)
        .expect("external owner-signed license manifest JSON");
    let expected_keys = [
        "format",
        "status",
        "owner_signoff",
        "publication",
        "blockers",
        "package_review",
        "native_payload_review",
        "weight_review",
        "source_review",
        "approval",
    ];
    exact_object_keys(&manifest, &expected_keys, "license manifest");
    assert_eq!(
        json_string(&manifest, "format"),
        "vokra-cosyvoice2-llm-license-gate-v1"
    );
    assert_eq!(json_string(&manifest, "status"), "APPROVED");
    assert_eq!(json_string(&manifest, "owner_signoff"), "OWNER_SIGNED_OFF");
    assert_eq!(json_string(&manifest, "publication"), "NO_UPLOAD");
    let blockers = manifest
        .get("blockers")
        .and_then(vokra_core::json::JsonValue::as_array)
        .expect("license blockers array");
    assert!(
        blockers.is_empty(),
        "approved license gate must have no blockers"
    );
    for (key, expected) in [
        ("package_review", "array"),
        ("native_payload_review", "string"),
        ("weight_review", "string"),
        ("source_review", "string"),
    ] {
        assert!(manifest.get(key).is_some(), "license {key} {expected}");
    }
    let package_review = manifest
        .get("package_review")
        .and_then(vokra_core::json::JsonValue::as_array)
        .expect("license package_review array");
    for row in package_review {
        exact_object_keys(
            row,
            &["name", "version", "status"],
            "license package review row",
        );
        assert!(!json_string(row, "name").trim().is_empty());
        assert!(!json_string(row, "version").trim().is_empty());
        assert_eq!(json_string(row, "status"), "APPROVED");
    }
    for key in ["native_payload_review", "weight_review", "source_review"] {
        assert_eq!(json_string(&manifest, key), "APPROVED", "license {key}");
    }
    let approval = json_object(&manifest, "approval");
    exact_object_keys(approval, &["signer", "scope_sha256"], "license approval");
    let signer = json_string(approval, "signer");
    assert!(!signer.trim().is_empty(), "license approval signer");
    let scope = json_string(approval, "scope_sha256");
    assert_eq!(scope.len(), 64, "license approval scope digest length");
    assert!(
        scope
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    );
}

fn authenticate_reference(reference: &Path) -> (Vec<u32>, Vec<f32>, (usize, usize)) {
    required_directory(reference, "reference directory");
    let manifest_path = reference.join("manifest.json");
    let token_path = reference.join("token-ids.json");
    let logits_path = reference.join("true_hf_logits.npy");
    let diagnostics_path = reference.join("diagnostics.json");
    let expected_names = [
        "diagnostics.json",
        "manifest.json",
        "token-ids.json",
        "true_hf_logits.npy",
    ];
    let actual_names = fs::read_dir(reference)
        .expect("reference directory entries")
        .map(|entry| {
            let entry = entry.expect("reference directory entry");
            let name = entry.file_name().to_string_lossy().into_owned();
            let metadata = fs::symlink_metadata(entry.path()).expect("reference entry metadata");
            assert!(
                metadata.file_type().is_file(),
                "reference entry must be a regular file: {name}"
            );
            assert!(
                !metadata.file_type().is_symlink(),
                "reference entry must not be a symlink: {name}"
            );
            name
        })
        .collect::<Vec<_>>();
    let mut actual_sorted = actual_names;
    actual_sorted.sort();
    assert_eq!(actual_sorted, expected_names, "reference artifact set");
    required_file(&manifest_path, "reference manifest");
    required_file(&token_path, "token IDs");
    required_file(&logits_path, "reference logits");
    required_file(&diagnostics_path, "reference diagnostics");

    let manifest_bytes = std::fs::read(&manifest_path).expect("reference manifest bytes");
    reject_duplicate_json_keys(&manifest_bytes, "reference manifest");
    let manifest = vokra_core::json::parse(&manifest_bytes).expect("reference manifest JSON");
    exact_object_keys(
        &manifest,
        &[
            "format",
            "status",
            "component",
            "model",
            "qwen_config",
            "source",
            "state",
            "execution",
            "artifacts",
        ],
        "reference manifest",
    );
    assert_eq!(
        json_string(&manifest, "format"),
        "vokra-cosyvoice2-llm-reference-v1"
    );
    assert_eq!(json_string(&manifest, "status"), "REFERENCE_READY");
    assert_eq!(json_string(&manifest, "component"), "llm");
    let model = json_object(&manifest, "model");
    exact_object_keys(
        model,
        &["repository", "revision", "path", "bytes", "sha256"],
        "model identity",
    );
    assert_eq!(json_string(model, "repository"), MODEL_REPOSITORY);
    assert_eq!(json_string(model, "revision"), MODEL_REVISION);
    assert_eq!(json_string(model, "path"), MODEL_PATH);
    assert_eq!(
        model.get("bytes").and_then(|value| value.as_u64()),
        Some(MODEL_BYTES)
    );
    assert_eq!(json_string(model, "sha256"), MODEL_SHA256);
    let qwen = json_object(&manifest, "qwen_config");
    exact_object_keys(
        qwen,
        &[
            "path",
            "bytes",
            "sha256",
            "git_blob_sha1",
            "verification",
            "fields",
        ],
        "Qwen config identity",
    );
    assert_eq!(json_string(qwen, "path"), "CosyVoice-BlankEN/config.json");
    assert_eq!(json_u64(qwen, "bytes"), 659);
    assert_eq!(
        json_string(qwen, "sha256"),
        "168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185"
    );
    assert_eq!(
        json_string(qwen, "git_blob_sha1"),
        "463b055262b6c66c4629a74a4b300bfe2ed31d3c"
    );
    assert_eq!(
        json_string(qwen, "verification"),
        "ACQUIRED_AND_HASH_VERIFIED"
    );
    let fields = json_object(qwen, "fields");
    exact_object_keys(
        fields,
        &[
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "vocab_size",
            "tie_word_embeddings",
        ],
        "Qwen config fields",
    );
    for (key, expected) in [
        ("hidden_size", 896),
        ("intermediate_size", 4864),
        ("num_hidden_layers", 24),
        ("num_attention_heads", 14),
        ("num_key_value_heads", 2),
        ("max_position_embeddings", 32768),
        ("vocab_size", 151936),
    ] {
        assert_eq!(json_u64(fields, key), expected);
    }
    assert_eq!(json_f64(fields, "rope_theta"), 1_000_000.0);
    assert_eq!(json_f64(fields, "rms_norm_eps"), 1e-6);
    assert!(json_bool(fields, "tie_word_embeddings"));
    let source = json_object(&manifest, "source");
    exact_object_keys(
        source,
        &["repository", "revision", "roles", "license"],
        "source identity",
    );
    assert_eq!(json_string(source, "repository"), SOURCE_REPOSITORY);
    assert_eq!(
        json_string(source, "revision"),
        "8555549e882236e6541748b1042d95693caa82ba"
    );
    let roles = json_object(source, "roles");
    let expected_roles = [
        (
            "cosyvoice/cli/cosyvoice.py",
            "cc443bed44c651a47492fc7e2142e3a88fb47627",
            "8e44f0f0144378561a00ebc065fdb15a843bc4650e68683bebb6624827731859",
            "CosyVoice2",
        ),
        (
            "cosyvoice/llm/llm.py",
            "59ebd48fde1f1b69240391fdac6e2afc1035e123",
            "6439d57fcf78bcdcad6d31812f3f4b02bd34f513333711ee317d71d1fd14d2de",
            "Qwen2LM",
        ),
        (
            "cosyvoice/tokenizer/tokenizer.py",
            "43fb39a2b543cc7ba4ec95fca9327596c34dcff0",
            "94340fc7cdf270c69a3aeb63290c5241044e20714e01fea736f361f9e5a56df2",
            "Qwen",
        ),
    ];
    exact_object_keys(
        roles,
        &[
            expected_roles[0].0,
            expected_roles[1].0,
            expected_roles[2].0,
        ],
        "source roles",
    );
    for (key, blob, sha, marker) in expected_roles {
        let role = json_object(roles, key);
        exact_object_keys(
            role,
            &["blob_sha1", "sha256", "marker", "bytes"],
            "source role",
        );
        assert_eq!(json_string(role, "blob_sha1"), blob);
        assert_eq!(json_string(role, "sha256"), sha);
        assert_eq!(json_string(role, "marker"), marker);
        assert!(json_u64(role, "bytes") > 0);
    }
    let source_license = json_object(source, "license");
    exact_object_keys(
        source_license,
        &["path", "bytes", "sha256", "git_blob_sha1", "declared"],
        "source license",
    );
    assert_eq!(json_string(source_license, "path"), "LICENSE");
    assert_eq!(json_u64(source_license, "bytes"), 11_357);
    assert_eq!(
        json_string(source_license, "sha256"),
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    );
    assert_eq!(
        json_string(source_license, "git_blob_sha1"),
        "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"
    );
    assert_eq!(json_string(source_license, "declared"), "Apache-2.0");
    let execution = json_object(&manifest, "execution");
    exact_object_keys(
        execution,
        &[
            "implementation",
            "attention",
            "dtype",
            "threads",
            "deterministic_algorithms",
            "model_execution",
            "publication",
        ],
        "reference execution",
    );
    assert_eq!(
        json_string(execution, "implementation"),
        "transformers.Qwen2ForCausalLM"
    );
    assert_eq!(json_string(execution, "attention"), "eager");
    assert_eq!(json_string(execution, "dtype"), "F32");
    assert_eq!(json_string(execution, "model_execution"), "RUN");
    assert_eq!(json_u64(execution, "threads"), 1);
    assert!(json_bool(execution, "deterministic_algorithms"));
    assert_eq!(json_string(execution, "publication"), "NO_UPLOAD");
    let state = json_object(&manifest, "state");
    exact_object_keys(
        state,
        &["tensor_count", "manifest_sha256"],
        "reference state",
    );
    assert_eq!(json_u64(state, "tensor_count"), TENSOR_COUNT as u64);
    assert_eq!(json_string(state, "manifest_sha256"), MANIFEST_SHA256);
    let artifacts = json_object(&manifest, "artifacts");
    exact_object_keys(
        artifacts,
        &["true_hf_logits.npy", "token-ids.json", "diagnostics.json"],
        "reference artifacts",
    );
    for (name, artifact_path) in [
        ("true_hf_logits.npy", &logits_path),
        ("token-ids.json", &token_path),
        ("diagnostics.json", &diagnostics_path),
    ] {
        let artifact = json_object(artifacts, name);
        exact_object_keys(artifact, &["bytes", "sha256"], "reference artifact");
        assert_eq!(
            artifact.get("bytes").and_then(|value| value.as_u64()),
            Some(std::fs::metadata(artifact_path).unwrap().len())
        );
        assert_eq!(json_string(artifact, "sha256"), sha256_file(artifact_path));
    }

    let token_bytes = std::fs::read(&token_path).expect("token IDs bytes");
    reject_duplicate_json_keys(&token_bytes, "token IDs");
    let token_root = vokra_core::json::parse(&token_bytes).expect("token IDs JSON");
    exact_object_keys(&token_root, &["ids", "text", "provenance"], "token IDs");
    let ids = token_root
        .get("ids")
        .and_then(vokra_core::json::JsonValue::as_array)
        .expect("token IDs array");
    let ids = ids
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .expect("u32 token ID")
        })
        .collect::<Vec<_>>();
    assert_eq!(ids, TOKEN_IDS, "reference token IDs drifted");
    assert_eq!(
        json_string(&token_root, "text"),
        "The quick brown fox jumps over the lazy dog."
    );
    assert_eq!(
        json_string(&token_root, "provenance"),
        "fixed documented ids; no tokenizer download"
    );

    let diagnostics_bytes = std::fs::read(&diagnostics_path).expect("diagnostics bytes");
    reject_duplicate_json_keys(&diagnostics_bytes, "reference diagnostics");
    let diagnostics = vokra_core::json::parse(&diagnostics_bytes).expect("diagnostics JSON");
    exact_object_keys(
        &diagnostics,
        &[
            "lm_head_vs_embed_max_abs_delta",
            "torch",
            "transformers",
            "numpy",
            "threads",
            "deterministic_algorithms",
            "attention_implementation",
            "state",
            "source",
        ],
        "reference diagnostics",
    );
    assert!(match diagnostics.get("lm_head_vs_embed_max_abs_delta") {
        Some(vokra_core::json::JsonValue::Float(value)) => value.is_finite(),
        Some(vokra_core::json::JsonValue::Int(_)) => true,
        _ => false,
    });
    for key in ["torch", "transformers", "numpy"] {
        assert!(
            !json_string(&diagnostics, key).trim().is_empty(),
            "diagnostics {key}"
        );
    }
    assert_eq!(json_u64(&diagnostics, "threads"), 1);
    assert!(json_bool(&diagnostics, "deterministic_algorithms"));
    assert_eq!(
        json_string(&diagnostics, "attention_implementation"),
        "eager"
    );
    let diagnostic_state = json_object(&diagnostics, "state");
    exact_object_keys(
        diagnostic_state,
        &["tensor_count", "manifest_sha256"],
        "diagnostic state",
    );
    assert_eq!(
        json_u64(diagnostic_state, "tensor_count"),
        TENSOR_COUNT as u64
    );
    assert_eq!(
        json_string(diagnostic_state, "manifest_sha256"),
        MANIFEST_SHA256
    );
    let diagnostic_source = json_object(&diagnostics, "source");
    exact_object_keys(
        diagnostic_source,
        &["repository", "revision", "roles", "license"],
        "diagnostic source",
    );
    assert_eq!(
        json_string(diagnostic_source, "repository"),
        SOURCE_REPOSITORY
    );
    assert_eq!(
        json_string(diagnostic_source, "revision"),
        "8555549e882236e6541748b1042d95693caa82ba"
    );
    assert_eq!(diagnostic_source, json_object(&manifest, "source"));

    read_npy_f32_2d(&logits_path)
        .map(|(shape, values)| (ids, values, shape))
        .unwrap_or_else(|error| panic!("reference logits: {error}"))
}

fn assert_outside_checkout(path: &Path, label: &str) {
    let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("canonical Vokra checkout root");
    let canonical = path
        .canonicalize()
        .unwrap_or_else(|error| panic!("{label} cannot be canonicalized: {error}"));
    assert!(
        !canonical.starts_with(&checkout),
        "{label} must be outside the Vokra checkout: {canonical:?}"
    );
}

fn read_npy_f32_2d(path: &Path) -> Result<((usize, usize), Vec<f32>), String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    if bytes.len() < 10 || bytes.get(..6).map(|value| value == b"\x93NUMPY") != Some(true) {
        return Err("bad NumPy magic".to_owned());
    }
    if bytes[7] != 0 {
        return Err("unsupported NumPy minor version".to_owned());
    }
    let (header_start, header_len) = match bytes[6] {
        1 => (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("truncated NumPy header".to_owned());
            }
            (
                12usize,
                u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            )
        }
        version => return Err(format!("unsupported NumPy version {version}")),
    };
    let header_end = header_start
        .checked_add(header_len)
        .ok_or("header overflow")?;
    let header = std::str::from_utf8(
        bytes
            .get(header_start..header_end)
            .ok_or("truncated header")?,
    )
    .map_err(|error| error.to_string())?;
    let compact = header
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect::<String>();
    if compact.matches("'descr'").count() != 1
        || compact.matches("'fortran_order'").count() != 1
        || compact.matches("'shape'").count() != 1
        || !matches!(
            compact.as_str(),
            "{'descr':'<f4','fortran_order':False,'shape':(11,151936),}"
                | "{'descr':'<f4','fortran_order':False,'shape':(11,151936)}"
        )
    {
        return Err("reference NPY header schema/dtype/shape mismatch".to_owned());
    }
    let shape = [TOKEN_IDS.len(), 151_936];
    let elements = shape[0].checked_mul(shape[1]).ok_or("shape overflow")?;
    let payload = bytes.get(header_end..).ok_or("missing payload")?;
    if payload.len() != elements.checked_mul(4).ok_or("payload overflow")? {
        return Err("NumPy payload size mismatch".to_owned());
    }
    let values = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("reference logits contain non-finite values".to_owned());
    }
    Ok(((shape[0], shape[1]), values))
}

#[test]
fn npy_header_rejects_duplicate_fields_and_minor_versions() {
    let duplicate = b"{'descr':'<f4','descr':'<f4','fortran_order':False,'shape':(11,151936),}";
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(duplicate.len() as u16).to_le_bytes());
    bytes.extend_from_slice(duplicate);
    assert!(read_npy_f32_2d_pathless(&bytes).is_err());
    bytes[7] = 1;
    assert!(read_npy_f32_2d_pathless(&bytes).is_err());
}

fn read_npy_f32_2d_pathless(bytes: &[u8]) -> Result<((usize, usize), Vec<f32>), String> {
    let path = std::env::temp_dir().join(format!(
        "vokra-cosyvoice2-llm-npy-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after Unix epoch")
            .as_nanos()
    ));
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .expect("create NPY self-test fixture without clobbering");
    file.write_all(bytes).expect("write NPY self-test fixture");
    drop(file);
    struct Cleanup<'a>(&'a Path);
    impl Drop for Cleanup<'_> {
        fn drop(&mut self) {
            let _ = fs::remove_file(self.0);
        }
    }
    let _cleanup = Cleanup(&path);
    read_npy_f32_2d(&path)
}

fn authenticate_gguf(file: &GgufFile) -> CosyVoice2Config {
    assert_eq!(file.tensors().len(), TENSOR_COUNT);
    assert_eq!(
        file.get(chunks::KEY_MODEL_ARCH)
            .and_then(|value| value.as_str()),
        Some("cosyvoice2")
    );
    for (key, expected) in [
        ("vokra.model.name", "cosyvoice2-0.5b"),
        ("vokra.model.category", "llm"),
        ("vokra.cosyvoice2.component", "llm"),
        ("vokra.cosyvoice2.composite_status", "INSPECTION_ONLY"),
        ("vokra.provenance.weight_license", "permissive"),
        ("vokra.provenance.license", "apache-2.0"),
        ("vokra.provenance.model_id", "cosyvoice2-0.5b"),
        ("vokra.provenance.upstream_hf", MODEL_REPOSITORY),
        ("vokra.provenance.upstream_revision", MODEL_REVISION),
        ("vokra.provenance.source", SOURCE_REPOSITORY),
        ("vokra.cosyvoice2.upstream_component.file", MODEL_PATH),
        ("vokra.cosyvoice2.upstream_component.sha256", MODEL_SHA256),
        ("vokra.cosyvoice2.tensor_manifest_sha256", MANIFEST_SHA256),
        (
            "vokra.cosyvoice2.prepared_input.authentication_status",
            "PREPARED_INPUT_DIGEST_RECORDED_NOT_PINNED",
        ),
    ] {
        assert_eq!(
            file.get(key).and_then(|value| value.as_str()),
            Some(expected),
            "GGUF metadata {key}"
        );
    }
    assert_eq!(
        file.get("vokra.cosyvoice2.upstream_component.bytes"),
        Some(&vokra_core::gguf::GgufMetadataValue::U32(
            MODEL_BYTES as u32
        ))
    );
    let prepared_sha = file
        .get("vokra.cosyvoice2.prepared_input.sha256")
        .and_then(|value| value.as_str())
        .expect("prepared input SHA-256 metadata");
    assert_eq!(prepared_sha.len(), 64);
    assert!(
        prepared_sha
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    );
    assert!(json_u64_from_gguf(file, "vokra.cosyvoice2.prepared_input.bytes") > 0);

    let config = CosyVoice2Config::from_gguf(file).expect("read GGUF CosyVoice2 config");
    assert_eq!(config.vocab_size, 151_936);
    assert_eq!(config.hidden_dim, 896);
    assert_eq!(config.n_layer, 24);
    assert_eq!(config.n_head, 14);
    assert_eq!(config.ffn_dim, 4_864);
    let llm_config = LlmBackboneConfig::from_gguf(file, &config).expect("read resolved LLM axes");
    assert_eq!(llm_config.n_head_q, 14);
    assert_eq!(llm_config.n_head_kv, 2);
    assert_eq!(llm_config.rope_base, 1_000_000.0);
    assert_eq!(llm_config.rms_norm_eps, 1.0e-6);
    assert_eq!(llm_config.n_ctx, 32_768);
    config
}

#[test]
#[ignore = "requires VAST CosyVoice2 LLM GGUF and official reference artifacts"]
fn parity_cosyvoice2_llm_component_real_transformers() {
    let gguf_path = required_path(GGUF_ENV);
    required_file(&gguf_path, "CosyVoice2 LLM GGUF");
    let reference = required_path(REFERENCE_ENV);
    let (token_ids, reference_logits, (rows, vocab)) = authenticate_reference(&reference);
    assert_eq!(rows, token_ids.len());

    let file = GgufFile::open(&gguf_path).expect("open CosyVoice2 LLM GGUF");
    let config = authenticate_gguf(&file);
    assert_eq!(config.vocab_size as usize, vocab);
    let backbone = LlmBackbone::from_gguf(&file, &config).expect("bind real GGUF backbone");
    assert!(!backbone.weights().is_synthesized);
    let report = parity::assert_vs_hf_reference(&backbone, &token_ids, &reference_logits, ATOL)
        .expect("official Transformers parity");
    assert_eq!(report.t, rows);
    assert_eq!(report.vocab, vocab);
    assert_eq!(report.argmax_matches, rows);
    eprintln!(
        "COSYVOICE2_LLM_PARITY_RESULT max_abs_delta={:.9e} mean_abs_delta={:.9e} argmax_matches={} argmax_total={} atol={ATOL:.9e}",
        report.max_abs_delta, report.mean_abs_delta, report.argmax_matches, report.t,
    );
}

fn argmax_first(values: &[f32]) -> usize {
    let (&first, rest) = values.split_first().expect("non-empty logits row");
    let mut best_index = 0;
    let mut best_value = first;
    for (index, &value) in rest.iter().enumerate() {
        if value > best_value {
            best_index = index + 1;
            best_value = value;
        }
    }
    best_index
}

fn compare_logits(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    rows: usize,
    vocab: usize,
) -> (f32, f64, usize) {
    assert_eq!(actual.len(), rows * vocab, "{label} logits shape");
    assert_eq!(expected.len(), actual.len(), "{label} comparison shape");
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut argmax_matches = 0usize;
    for row in 0..rows {
        let actual_row = &actual[row * vocab..(row + 1) * vocab];
        let expected_row = &expected[row * vocab..(row + 1) * vocab];
        assert!(
            actual_row.iter().all(|value| value.is_finite()),
            "{label} logits contain non-finite values"
        );
        for (&actual_value, &expected_value) in actual_row.iter().zip(expected_row) {
            let delta = (actual_value - expected_value).abs();
            max_abs = max_abs.max(delta);
            sum_abs += f64::from(delta);
        }
        if argmax_first(actual_row) == argmax_first(expected_row) {
            argmax_matches += 1;
        }
    }
    let mean_abs = sum_abs / (actual.len() as f64);
    assert!(
        max_abs <= ATOL,
        "{label} max_abs_delta={max_abs:.9e} exceeds ATOL={ATOL:.9e}"
    );
    assert_eq!(
        argmax_matches, rows,
        "{label} argmax mismatch: {argmax_matches}/{rows}"
    );
    (max_abs, mean_abs, argmax_matches)
}

#[test]
#[ignore = "requires VAST CosyVoice2 LLM GGUF, official reference artifacts, and Apple Metal"]
fn parity_cosyvoice2_llm_component_real_apple_cpu_metal() {
    if !cfg!(target_os = "macos") {
        panic!("Apple CPU/Metal parity must run on macOS; this explicit test must not skip");
    }
    if !cfg!(target_arch = "aarch64") {
        panic!("Apple CPU/Metal parity must run on arm64; this explicit test must not skip");
    }
    assert_eq!(
        std::env::var("VOKRA_REMOTE_APPLE_SILICON").as_deref(),
        Ok("1"),
        "Apple parity requires an explicitly designated remote host"
    );
    let gguf_path = required_path(GGUF_ENV);
    required_file(&gguf_path, "CosyVoice2 LLM GGUF");
    assert_outside_checkout(&gguf_path, "CosyVoice2 LLM GGUF");
    let gguf_sha = required_sha256(GGUF_SHA_ENV);
    assert_eq!(sha256_file(&gguf_path), gguf_sha, "GGUF digest");
    let reference = required_path(REFERENCE_ENV);
    required_directory(&reference, "reference directory");
    assert_outside_checkout(&reference, "reference directory");
    let reference_manifest_sha = required_sha256(REFERENCE_MANIFEST_SHA_ENV);
    assert_eq!(
        sha256_file(&reference.join("manifest.json")),
        reference_manifest_sha,
        "reference manifest digest"
    );
    let license_manifest = required_path(LICENSE_MANIFEST_ENV);
    required_file(&license_manifest, "external owner-signed license manifest");
    assert_outside_checkout(&license_manifest, "external owner-signed license manifest");
    let canonical_inputs = [
        gguf_path.canonicalize().expect("canonical GGUF path"),
        reference.canonicalize().expect("canonical reference path"),
        license_manifest
            .canonicalize()
            .expect("canonical license path"),
    ];
    for left in 0..canonical_inputs.len() {
        for right in (left + 1)..canonical_inputs.len() {
            assert!(
                !canonical_inputs[left].starts_with(&canonical_inputs[right])
                    && !canonical_inputs[right].starts_with(&canonical_inputs[left]),
                "Apple parity inputs must be pairwise disjoint"
            );
        }
    }
    authenticate_external_license_manifest(&license_manifest);
    let (token_ids, reference_logits, (rows, vocab)) = authenticate_reference(&reference);
    assert_eq!(rows, token_ids.len());

    let file = GgufFile::open(&gguf_path).expect("open CosyVoice2 LLM GGUF");
    let config = authenticate_gguf(&file);
    assert_eq!(config.vocab_size as usize, vocab);

    // Keep only the small CPU logits vector alive while binding the second
    // real-weight model. This proves both backends use the same authenticated
    // GGUF without retaining two complete weight stores unnecessarily.
    let cpu_model = LlmBackbone::from_gguf(&file, &config).expect("bind real CPU GGUF backbone");
    assert!(!cpu_model.weights().is_synthesized);
    assert_eq!(cpu_model.backend(), BackendKind::Cpu);
    let cpu_logits = cpu_model.forward(&token_ids, 0).expect("CPU logits");
    let cpu_report = compare_logits("CPU/reference", &cpu_logits, &reference_logits, rows, vocab);
    drop(cpu_model);

    let metal_model = LlmBackbone::from_gguf(&file, &config)
        .expect("bind real Metal GGUF backbone")
        .with_backend(BackendKind::Metal);
    assert!(!metal_model.weights().is_synthesized);
    assert_eq!(metal_model.backend(), BackendKind::Metal);
    assert_eq!(
        metal_model.backend_name(),
        "metal",
        "Metal leg must retain its requested backend"
    );
    // A missing device or uncovered Metal op is an error from forward(); the
    // test deliberately does not skip and cannot silently fall back to CPU.
    let metal_logits = metal_model.forward(&token_ids, 0).expect("Metal logits");
    let metal_report = compare_logits(
        "Metal/reference",
        &metal_logits,
        &reference_logits,
        rows,
        vocab,
    );
    let metal_cpu_report = compare_logits("Metal/CPU", &metal_logits, &cpu_logits, rows, vocab);
    assert_eq!(metal_cpu_report.2, rows);
    eprintln!(
        "COSYVOICE2_LLM_APPLE_PARITY backend=cpu,metal metal_device=present cpu_reference_max_abs={:.9e} metal_reference_max_abs={:.9e} metal_cpu_max_abs={:.9e} cpu_reference_mean_abs={:.9e} metal_reference_mean_abs={:.9e} metal_cpu_mean_abs={:.9e} argmax_matches={}/{} atol={ATOL:.9e} verdict=PASS",
        cpu_report.0,
        metal_report.0,
        metal_cpu_report.0,
        cpu_report.1,
        metal_report.1,
        metal_cpu_report.1,
        metal_cpu_report.2,
        rows,
    );
}

fn json_u64_from_gguf(file: &GgufFile, key: &str) -> u64 {
    match file.get(key) {
        Some(vokra_core::gguf::GgufMetadataValue::U64(value)) => *value,
        Some(value) => panic!("GGUF metadata {key} is not UINT64: {value:?}"),
        None => panic!("GGUF metadata {key} is missing"),
    }
}
