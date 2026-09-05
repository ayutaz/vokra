//! VAST-only real-checkpoint CosyVoice2 HiFT parity.
//!
//! The reference packet is produced by the independent upstream PyTorch
//! dumper (`tools/parity/cosyvoice2_hift_dump_reference.py`). This test is
//! deliberately ignored: enabling it requires an authenticated GGUF and
//! reference directory supplied by a VAST worker. Apple Metal parity is a
//! separate final Scaleway gate and is not covered by this Linux-only test.

use std::path::{Path, PathBuf};

use vokra_core::gguf::GgufFile;
use vokra_core::json::{self, JsonValue};
use vokra_models::cosyvoice2::HiFTChain;

const ATOL: f32 = 0.01;
const RTOL: f32 = 0.0;
const FORMAT: &str = "vokra-cosyvoice2-hift-reference-v1";
const MODEL_REVISION: &str = "eec1ae6c79877dbd9379285cf8789c9e0879293d";
const MODEL_SHA256: &str = "3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879";
const SOURCE_REVISION: &str = "8555549e882236e6541748b1042d95693caa82ba";
const CONFIG_SHA256: &str = "0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959";
const CONFIG_BLOB_SHA1: &str = "bc19267bbfd373c9a760b7667a74349ddd487db1";
const MANIFEST_SHA256: &str = "cecbb2d68f91337f263db0f0333c75573516e7087b6e75d6ea647b3f86afec7c";
const LICENSE_ENV: &str = "VOKRA_COSYVOICE2_HIFT_LICENSE_MANIFEST";
const GGUF_SHA_ENV: &str = "VOKRA_COSYVOICE2_HIFT_GGUF_SHA256";
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

fn required_path(var: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("{var} is required")));
    assert!(
        path.is_absolute(),
        "{var} must be absolute: {}",
        path.display()
    );
    assert!(
        !path.is_symlink() && path.is_file(),
        "{var} must be a regular non-symlink file: {}",
        path.display()
    );
    path
}

fn required_dir(var: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("{var} is required")));
    assert!(
        path.is_absolute(),
        "{var} must be absolute: {}",
        path.display()
    );
    assert!(
        !path.is_symlink() && path.is_dir(),
        "{var} must be a regular non-symlink directory: {}",
        path.display()
    );
    path
}

fn field<'a>(root: &'a JsonValue, path: &[&str]) -> &'a JsonValue {
    path.iter().fold(root, |value, key| {
        value
            .get(key)
            .unwrap_or_else(|| panic!("missing manifest field {}", path.join(".")))
    })
}
fn string(root: &JsonValue, path: &[&str]) -> &str {
    field(root, path)
        .as_str()
        .unwrap_or_else(|| panic!("manifest field {} is not a string", path.join(".")))
}
fn number(root: &JsonValue, path: &[&str]) -> usize {
    field(root, path).as_u64().unwrap_or_else(|| {
        panic!(
            "manifest field {} is not a nonnegative integer",
            path.join(".")
        )
    }) as usize
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

fn exact_keys(value: &JsonValue, expected: &[&str], label: &str) {
    let object = value
        .as_object()
        .unwrap_or_else(|| panic!("{label} must be an object"));
    assert_eq!(
        object.len(),
        expected.len(),
        "{label} has an unexpected key count"
    );
    for key in expected {
        assert!(
            object.iter().any(|(actual, _)| actual == key),
            "{label} is missing key {key}"
        );
    }
}

fn required_status(root: &JsonValue, path: &[&str], label: &str) {
    assert!(
        matches!(string(root, path), "APPROVED" | "REVIEWED"),
        "{label} must be APPROVED or REVIEWED"
    );
}

fn sha256_hex(data: &[u8]) -> String {
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
    let bit = (p.len() as u64) * 8;
    p.push(0x80);
    while p.len() % 64 != 56 {
        p.push(0)
    }
    p.extend_from_slice(&bit.to_be_bytes());
    for block in p.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, c) in block.chunks_exact(4).take(16).enumerate() {
            w[i] = u32::from_be_bytes(c.try_into().unwrap())
        }
        for i in 16..64 {
            let a = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let b = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(a)
                .wrapping_add(w[i - 7])
                .wrapping_add(b)
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut x) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = x
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            x = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2)
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(x)
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

fn validate_license_manifest(reference: &JsonValue, project_sha256: &str, lock_sha256: &str) {
    let license_path = required_path(LICENSE_ENV);
    let license_bytes =
        std::fs::read(&license_path).unwrap_or_else(|e| panic!("read license manifest: {e}"));
    let license_sha256 = sha256_hex(&license_bytes);
    assert_eq!(
        license_sha256,
        string(reference, &["license_manifest_sha256"]),
        "license manifest SHA-256"
    );
    let license = json::parse(&license_bytes).expect("parse license manifest");
    exact_keys(
        &license,
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
    assert_eq!(number(&license, &["gate_version"]), 1);
    assert_eq!(string(&license, &["status"]), "APPROVED");
    assert_eq!(string(&license, &["owner_signoff"]), "OWNER_SIGNED_OFF");
    assert_eq!(string(&license, &["publication"]), "NO_UPLOAD");
    assert_eq!(
        string(&license, &["decision"]),
        "DO_NOT_DISTRIBUTE_UNTIL_OWNER_SIGNOFF"
    );

    exact_keys(
        field(&license, &["source"]),
        &["repository", "revision", "license_status"],
        "license source",
    );
    assert_eq!(
        string(&license, &["source", "repository"]),
        "https://github.com/FunAudioLLM/CosyVoice.git"
    );
    assert_eq!(string(&license, &["source", "revision"]), SOURCE_REVISION);
    required_status(
        &license,
        &["source", "license_status"],
        "license source status",
    );

    exact_keys(
        field(&license, &["model"]),
        &[
            "repository",
            "revision",
            "file",
            "bytes",
            "sha256",
            "weight_license_status",
        ],
        "license model",
    );
    assert_eq!(
        string(&license, &["model", "repository"]),
        "FunAudioLLM/CosyVoice2-0.5B"
    );
    assert_eq!(string(&license, &["model", "revision"]), MODEL_REVISION);
    assert_eq!(string(&license, &["model", "file"]), "hift.pt");
    assert_eq!(number(&license, &["model", "bytes"]), 83_390_254);
    assert_eq!(string(&license, &["model", "sha256"]), MODEL_SHA256);
    required_status(
        &license,
        &["model", "weight_license_status"],
        "license model status",
    );

    exact_keys(
        field(&license, &["config"]),
        &["file", "bytes", "sha256", "git_blob_sha1"],
        "license config",
    );
    assert_eq!(string(&license, &["config", "file"]), "cosyvoice2.yaml");
    assert_eq!(number(&license, &["config", "bytes"]), 7_330);
    assert_eq!(string(&license, &["config", "sha256"]), CONFIG_SHA256);
    assert_eq!(
        string(&license, &["config", "git_blob_sha1"]),
        CONFIG_BLOB_SHA1
    );

    exact_keys(
        field(&license, &["python_closure"]),
        &[
            "runtime_pins",
            "platform",
            "license_status",
            "forbidden_packages",
        ],
        "license Python closure",
    );
    exact_keys(
        field(&license, &["python_closure", "runtime_pins"]),
        &["torch", "numpy", "scipy"],
        "license runtime pins",
    );
    assert_eq!(
        string(&license, &["python_closure", "runtime_pins", "torch"]),
        "2.7.1"
    );
    assert_eq!(
        string(&license, &["python_closure", "runtime_pins", "numpy"]),
        "2.3.5"
    );
    assert_eq!(
        string(&license, &["python_closure", "runtime_pins", "scipy"]),
        "1.16.3"
    );
    assert_eq!(
        string(&license, &["python_closure", "platform"]),
        "linux-x86_64-cpu"
    );
    required_status(
        &license,
        &["python_closure", "license_status"],
        "license Python closure status",
    );
    let forbidden = field(&license, &["python_closure", "forbidden_packages"])
        .as_array()
        .expect("license forbidden package list");
    let forbidden_names: Vec<&str> = forbidden
        .iter()
        .map(|value| value.as_str().expect("forbidden package name"))
        .collect();
    assert_eq!(
        forbidden_names,
        ["triton", "nvidia", "librosa", "soxr", "soundfile"]
    );

    exact_keys(
        field(&license, &["evidence"]),
        &[
            "tensor_count",
            "tensor_dtype",
            "tensor_manifest_sha256",
            "source_roles",
        ],
        "license evidence",
    );
    assert_eq!(number(&license, &["evidence", "tensor_count"]), 328);
    assert_eq!(string(&license, &["evidence", "tensor_dtype"]), "F32");
    assert_eq!(
        string(&license, &["evidence", "tensor_manifest_sha256"]),
        MANIFEST_SHA256
    );
    exact_keys(
        field(&license, &["evidence", "source_roles"]),
        &[
            "cosyvoice/hifigan/generator.py",
            "cosyvoice/hifigan/f0_predictor.py",
            "cosyvoice/transformer/activation.py",
            "cosyvoice/utils/common.py",
        ],
        "license source roles",
    );
    for (role, blob) in SOURCE_ROLES {
        assert_eq!(string(&license, &["evidence", "source_roles", role]), *blob);
    }

    exact_keys(
        field(&license, &["approval"]),
        &["schema", "signer", "scope_sha256"],
        "license approval",
    );
    assert_eq!(
        string(&license, &["approval", "schema"]),
        "cosyvoice2-hift-approval-scope-v1"
    );
    assert!(
        !string(&license, &["approval", "signer"]).trim().is_empty(),
        "license approval signer must be non-empty"
    );
    hash(
        string(&license, &["approval", "scope_sha256"]),
        "license approval scope",
    );
    assert_eq!(
        string(&license, &["approval", "scope_sha256"]),
        string(reference, &["approval_scope_sha256"]),
        "license approval scope SHA-256"
    );
    assert_eq!(
        project_sha256,
        string(reference, &["project_sha256"]),
        "reference project SHA-256"
    );
    assert_eq!(
        lock_sha256,
        string(reference, &["uv_lock_sha256"]),
        "reference uv.lock SHA-256"
    );
}

fn f32_file(
    path: &Path,
    expected_bytes: usize,
    expected_shape: &[usize],
    root: &JsonValue,
    section: &str,
    name: &str,
) -> Vec<f32> {
    assert!(
        !path.is_symlink() && path.is_file(),
        "artifact must be a regular non-symlink file: {}",
        path.display()
    );
    let record_path = if section == "input" {
        vec![section]
    } else {
        vec![section, name]
    };
    let record = field(root, &record_path);
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(bytes.len(), expected_bytes, "{} byte count", path.display());
    assert_eq!(
        record.get("bytes").and_then(JsonValue::as_u64).unwrap() as usize,
        expected_bytes
    );
    assert_eq!(record.get("dtype").and_then(JsonValue::as_str), Some("F32"));
    let shape = record
        .get("shape")
        .and_then(JsonValue::as_array)
        .expect("shape array");
    assert_eq!(
        shape
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>(),
        expected_shape
    );
    let expected_sha = record
        .get("sha256")
        .and_then(JsonValue::as_str)
        .expect("artifact sha256");
    hash(expected_sha, name);
    assert_eq!(
        sha256_hex(&bytes),
        expected_sha,
        "{} SHA-256",
        path.display()
    );
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!(
        values.iter().all(|v| v.is_finite()),
        "{} contains non-finite values",
        path.display()
    );
    assert!(
        !values.is_empty() && values.iter().any(|v| *v != 0.0),
        "{} is empty or vacuous",
        path.display()
    );
    values
}

fn max_diff(actual: &[f32], expected: &[f32]) -> (usize, f32) {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty(), "parity vectors must be non-empty");
    actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(i, (a, e))| (i, (*a - *e).abs()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

fn within(actual: &[f32], expected: &[f32]) -> bool {
    assert_eq!(
        actual.len(),
        expected.len(),
        "parity vectors must have equal lengths"
    );
    assert!(!actual.is_empty(), "parity vectors must be non-empty");
    actual
        .iter()
        .zip(expected)
        .all(|(a, e)| (*a - *e).abs() <= ATOL + RTOL * e.abs())
}

#[test]
#[ignore]
fn cosyvoice2_hift_real_cpu_parity() {
    assert_eq!(
        std::env::var("VOKRA_PUBLISH_ON_VAST").as_deref(),
        Ok("1"),
        "real parity is VAST-only"
    );
    let gguf_path = required_path("VOKRA_COSYVOICE2_HIFT_GGUF");
    let reference_dir = required_dir("VOKRA_COSYVOICE2_HIFT_REFERENCE_DIR");
    let mut packet_names = std::fs::read_dir(&reference_dir)
        .expect("read reference packet directory")
        .map(|entry| {
            let path = entry.expect("read reference packet entry").path();
            assert!(
                !path.is_symlink() && path.is_file(),
                "reference packet entries must be regular non-symlink files: {}",
                path.display()
            );
            path.file_name()
                .and_then(|name| name.to_str())
                .expect("reference packet filename")
                .to_owned()
        })
        .collect::<Vec<_>>();
    packet_names.sort();
    assert_eq!(
        packet_names,
        ["f0.f32", "manifest.json", "mel.f32", "pcm.f32"],
        "reference packet has unexpected files"
    );
    let manifest_path = reference_dir.join("manifest.json");
    assert!(
        !manifest_path.is_symlink() && manifest_path.is_file(),
        "manifest must be regular and non-symlink"
    );
    let manifest = json::parse(&std::fs::read(&manifest_path).expect("read manifest"))
        .expect("parse manifest");
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
    assert_eq!(string(&manifest, &["format"]), FORMAT);
    assert_eq!(string(&manifest, &["status"]), "AUTHENTICATED_REFERENCE");
    assert_eq!(string(&manifest, &["publication"]), "NO_UPLOAD");
    assert_eq!(
        string(&manifest, &["model", "repository"]),
        "FunAudioLLM/CosyVoice2-0.5B"
    );
    assert_eq!(string(&manifest, &["model", "revision"]), MODEL_REVISION);
    assert_eq!(string(&manifest, &["model", "path"]), "hift.pt");
    assert_eq!(number(&manifest, &["model", "bytes"]), 83_390_254);
    assert_eq!(string(&manifest, &["model", "sha256"]), MODEL_SHA256);
    assert_eq!(string(&manifest, &["config", "path"]), "cosyvoice2.yaml");
    assert_eq!(number(&manifest, &["config", "bytes"]), 7_330);
    assert_eq!(string(&manifest, &["config", "sha256"]), CONFIG_SHA256);
    assert_eq!(
        string(&manifest, &["config", "git_blob_sha1"]),
        CONFIG_BLOB_SHA1
    );
    assert_eq!(string(&manifest, &["source", "revision"]), SOURCE_REVISION);
    assert_eq!(field(&manifest, &["source", "clean"]).as_bool(), Some(true));
    assert_eq!(
        string(&manifest, &["source", "repository"]),
        "https://github.com/FunAudioLLM/CosyVoice.git"
    );
    for (role, blob) in SOURCE_ROLES {
        assert_eq!(
            string(&manifest, &["source", "roles", role, "git_blob_sha1"]),
            *blob
        );
        hash(
            string(&manifest, &["source", "roles", role, "sha256"]),
            role,
        );
    }
    assert_eq!(
        string(&manifest, &["checkpoint_tensor_manifest_sha256"]),
        MANIFEST_SHA256
    );
    for key in [
        "license_manifest_sha256",
        "approval_scope_sha256",
        "project_sha256",
        "uv_lock_sha256",
    ] {
        hash(string(&manifest, &[key]), key);
    }
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let project_path = project_root.join("tools/parity/cosyvoice2_hift_reference/pyproject.toml");
    let lock_path = project_root.join("tools/parity/cosyvoice2_hift_reference/uv.lock");
    for (path, label) in [
        (&project_path, "reference pyproject.toml"),
        (&lock_path, "reference uv.lock"),
    ] {
        assert!(
            !path.is_symlink() && path.is_file(),
            "{label} must be a regular non-symlink file: {}",
            path.display()
        );
    }
    let project_sha256 =
        sha256_hex(&std::fs::read(&project_path).expect("read reference pyproject.toml"));
    let lock_sha256 = sha256_hex(&std::fs::read(&lock_path).expect("read reference uv.lock"));
    assert_eq!(
        project_sha256,
        string(&manifest, &["project_sha256"]),
        "checked-in reference project SHA-256"
    );
    assert_eq!(
        lock_sha256,
        string(&manifest, &["uv_lock_sha256"]),
        "checked-in reference uv.lock SHA-256"
    );
    validate_license_manifest(&manifest, &project_sha256, &lock_sha256);
    assert_eq!(
        string(&manifest, &["execution", "uv_lock_sha256"]),
        string(&manifest, &["uv_lock_sha256"])
    );
    assert_eq!(string(&manifest, &["execution", "device"]), "cpu");
    assert_eq!(
        field(&manifest, &["execution", "torch_deterministic_algorithms"]).as_bool(),
        Some(true)
    );
    assert_eq!(number(&manifest, &["execution", "threads"]), 1);
    assert_eq!(
        string(&manifest, &["execution", "entropy_override"]),
        "torch.rand and torch.randn_like -> zeros only during official forward"
    );
    assert!(
        string(&manifest, &["execution", "python"]).starts_with("3.12."),
        "reference Python must be 3.12"
    );
    assert_eq!(string(&manifest, &["execution", "torch"]), "2.7.1+cpu");
    assert_eq!(string(&manifest, &["execution", "numpy"]), "2.3.5");
    assert_eq!(string(&manifest, &["execution", "scipy"]), "1.16.3");

    let expected_gguf_sha256 =
        std::env::var(GGUF_SHA_ENV).unwrap_or_else(|_| panic!("{GGUF_SHA_ENV} is required"));
    hash(&expected_gguf_sha256, GGUF_SHA_ENV);
    let gguf_bytes = std::fs::read(&gguf_path).expect("read authenticated HiFT GGUF");
    assert_eq!(
        sha256_hex(&gguf_bytes),
        expected_gguf_sha256,
        "GGUF SHA-256 does not match the VAST evidence binding"
    );

    let mel = f32_file(
        &reference_dir.join("mel.f32"),
        80 * 8 * 4,
        &[1, 80, 8],
        &manifest,
        "input",
        "input",
    );
    assert_eq!(number(&manifest, &["input", "seed"]), 2_026_0906);
    assert_eq!(
        string(&manifest, &["input", "formula"]),
        "torch.linspace(-0.25,0.25,640).reshape(1,80,8)"
    );
    let f0 = f32_file(
        &reference_dir.join("f0.f32"),
        8 * 4,
        &[1, 8],
        &manifest,
        "outputs",
        "f0",
    );
    let pcm = f32_file(
        &reference_dir.join("pcm.f32"),
        3_840 * 4,
        &[1, 3_840],
        &manifest,
        "outputs",
        "pcm",
    );
    let chain =
        HiFTChain::from_gguf(&GgufFile::open(&gguf_path).expect("open authenticated HiFT GGUF"))
            .expect("bind authenticated HiFT GGUF");
    let actual_f0 = chain
        .f0_predictor_forward(&mel, 8)
        .expect("native F0 predictor");
    let (f0_i, f0_d) = max_diff(&actual_f0, &f0);
    eprintln!("CosyVoice2 HiFT F0 CPU parity: max_abs={f0_d:.9e} at {f0_i}");
    assert!(
        within(&actual_f0, &f0),
        "F0 max diff {f0_d:.9e} at {f0_i} exceeds atol {ATOL}"
    );
    let actual_pcm = chain.forward(&mel, 8).expect("native HiFT CPU forward");
    let (pcm_i, pcm_d) = max_diff(&actual_pcm, &pcm);
    eprintln!("CosyVoice2 HiFT PCM CPU parity: max_abs={pcm_d:.9e} at {pcm_i}");
    assert!(
        within(&actual_pcm, &pcm),
        "PCM max diff {pcm_d:.9e} at {pcm_i} exceeds atol {ATOL}"
    );
    let mut perturbed = f0.clone();
    perturbed[0] += ATOL * 4.0;
    assert!(
        !within(&actual_f0, &perturbed),
        "negative control must reject a material perturbation"
    );
}
