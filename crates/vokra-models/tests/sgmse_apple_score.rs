//! Apple Silicon SGMSE score parity against the authenticated VAST packet.
//!
//! This is a single ignored, real-weight test reserved for a disposable
//! Scaleway Apple worker. It runs the same fixed state/condition through CPU
//! and Metal, compares both to the independent score planes, and records only
//! bounded evidence. No checkpoint conversion, upload, or reference creation
//! occurs here.

#![cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(dead_code, unused_imports)
)]

use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_core::json::{self, JsonValue};
use vokra_models::compute::Compute;
use vokra_models::sgmse::{SGMSE_HOT_OPS, SgmseModel};

#[cfg(not(all(feature = "metal", target_os = "macos")))]
use vokra_core::VokraError;

const ATOL: f32 = 0.01;
const HEIGHT: usize = 256;
const WIDTH: usize = 64;
const COUNT: usize = HEIGHT * WIDTH;
const BYTES: usize = COUNT * 4;
const GGUF_ENV: &str = "VOKRA_SGMSE_GGUF";
const GGUF_SHA_ENV: &str = "VOKRA_SGMSE_GGUF_SHA256";
const REFERENCE_ENV: &str = "VOKRA_SGMSE_REFERENCE_DIR";
const EVIDENCE_ENV: &str = "VOKRA_SGMSE_APPLE_EVIDENCE_DIR";
const REFERENCE_MANIFEST_SHA_ENV: &str = "VOKRA_SGMSE_REFERENCE_MANIFEST_SHA256";
const REMOTE_ENV: &str = "VOKRA_REMOTE_APPLE_SILICON";
const ARTIFACTS: &[(&str, &str)] = &[
    (
        "input_condition_imag",
        "37d4a9e7d1793aaef270cdbaddf69464fe3286171661c8f75380bd6f6e305893",
    ),
    (
        "input_condition_real",
        "8fa96184edbec9c85856eebabd6ba6102fee3e30debaf7aee2fffffd1e9599ea",
    ),
    (
        "input_noisy_imag",
        "a355948bcbafb8b89a3975d40ee333129216e730e153d8ef26d5419ed07f90ba",
    ),
    (
        "input_noisy_real",
        "c62e324c7826c752b2a8b567d184bca31cd9e1dd6b1ac04885eb78f1ccf325fa",
    ),
    (
        "score_imag",
        "ea029f909ed9eae729b2b52e51807847aece53ee574435b3b3c0f3bb713b25d5",
    ),
    (
        "score_real",
        "f15e232711181167317c820b3e0c12f07fcad8f30cd431031d196aedabbda16b",
    ),
];

fn env_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| panic!("{name} is required"))
}

fn require_abs(path: &Path, label: &str) {
    assert!(
        path.is_absolute(),
        "{label} must be absolute: {}",
        path.display()
    );
}

fn reject_symlink_ancestry(path: &Path, label: &str) {
    let mut current = path;
    loop {
        assert!(
            !current.is_symlink(),
            "{label} has symlink ancestry: {}",
            current.display()
        );
        if current == Path::new("/") {
            return;
        }
        current = current
            .parent()
            .unwrap_or_else(|| panic!("{label} has no parent"));
    }
}

fn overlap(left: &Path, right: &Path) -> bool {
    let canonical = |path: &Path| {
        if path.exists() {
            path.canonicalize()
        } else {
            let parent = path
                .parent()
                .unwrap_or_else(|| panic!("path has no parent"));
            parent
                .canonicalize()
                .map(|parent| parent.join(path.file_name().unwrap()))
        }
        .unwrap_or_else(|e| panic!("canonicalize {}: {e}", path.display()))
    };
    let left = canonical(left);
    let right = canonical(right);
    left == right || left.starts_with(&right) || right.starts_with(&left)
}

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
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x28877aaa,
            0x3b8b4c84, 0x4d2c6dfc, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x81c2c92e,
            0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
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

fn field<'a>(root: &'a JsonValue, key: &str) -> &'a JsonValue {
    root.get(key)
        .unwrap_or_else(|| panic!("manifest missing {key}"))
}
fn string_field(root: &JsonValue, key: &str) -> String {
    field(root, key)
        .as_str()
        .unwrap_or_else(|| panic!("manifest {key} is not a string"))
        .to_owned()
}
fn u64_field(root: &JsonValue, key: &str) -> u64 {
    field(root, key)
        .as_u64()
        .unwrap_or_else(|| panic!("manifest {key} is not an integer"))
}
fn bool_field(root: &JsonValue, key: &str) -> bool {
    match field(root, key) {
        JsonValue::Bool(value) => *value,
        _ => panic!("manifest {key} is not a boolean"),
    }
}
fn no_duplicate_json(root: &JsonValue) {
    if let JsonValue::Object(entries) = root {
        let mut seen = BTreeSet::new();
        for (key, value) in entries {
            assert!(seen.insert(key), "duplicate manifest key {key}");
            no_duplicate_json(value);
        }
    } else if let JsonValue::Array(items) = root {
        for item in items {
            no_duplicate_json(item);
        }
    }
}

fn read_plane(path: &Path, expected_sha: &str) -> Vec<f32> {
    assert!(
        path.is_file() && !path.is_symlink(),
        "invalid reference file {}",
        path.display()
    );
    assert_eq!(
        sha256_file(path),
        expected_sha,
        "reference digest mismatch: {}",
        path.display()
    );
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(bytes.len(), BYTES);
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(values.iter().all(|v| v.is_finite()));
    values
}

fn verify_reference(reference: &Path) {
    assert_eq!(
        fs::read_dir(reference)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<BTreeSet<_>>(),
        ["manifest.json", "run.log"]
            .into_iter()
            .chain(ARTIFACTS.iter().map(|(name, _)| format!("{name}.f32")))
            .collect()
    );
    let root = json::parse(&fs::read(reference.join("manifest.json")).unwrap())
        .expect("reference manifest JSON");
    no_duplicate_json(&root);
    assert_eq!(
        string_field(&root, "format"),
        "vokra-sgmse-score-reference-v1"
    );
    assert_eq!(
        string_field(&root, "status"),
        "REFERENCE_COMPLETE_NO_UPLOAD"
    );
    assert_eq!(string_field(&root, "publication"), "NO_UPLOAD");
    assert_eq!(string_field(&root, "fixtures"), "VAST_ONLY");
    assert_eq!(
        string_field(&root, "fixture_payload"),
        "retained_for_native_parity"
    );
    assert_eq!(
        string_field(&root, "model_repository"),
        "speechbrain/sgmse-voicebank"
    );
    assert_eq!(
        string_field(&root, "model_revision"),
        "8f4ff7b65284c49492a43349b8106e094ac0d365"
    );
    let source = field(&root, "source");
    assert_eq!(
        string_field(source, "repository"),
        "https://github.com/sp-uhh/sgmse.git"
    );
    assert_eq!(
        string_field(source, "revision"),
        "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e"
    );
    assert_eq!(
        string_field(source, "license_sha256"),
        "8748956d2e5afe9dfc8311188b4119dacc7c5293b0561e7cca7a21cf80e54caa"
    );
    let speech = field(&root, "speechbrain_source");
    assert_eq!(
        string_field(speech, "repository"),
        "https://github.com/speechbrain/speechbrain.git"
    );
    assert_eq!(
        string_field(speech, "revision"),
        "2b3f4f44351fd08a627c4ab307de5c420351bc19"
    );
    assert_eq!(
        string_field(speech, "license_sha256"),
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    );
    let checkpoint = field(&root, "checkpoint");
    assert_eq!(string_field(checkpoint, "filename"), "score_model_ema.ckpt");
    assert_eq!(u64_field(checkpoint, "size"), 262_593_305);
    assert_eq!(
        string_field(checkpoint, "sha256"),
        "7ca96321aca40cdca90c450d1450a5c7f343935e5b46ee34a1b575f9f774ccc3"
    );
    assert_eq!(string_field(source, "license_spdx"), "mit");
    assert_eq!(string_field(speech, "license_spdx"), "apache-2.0");
    let licenses = field(&root, "licenses");
    assert_eq!(string_field(field(licenses, "algorithm"), "spdx"), "mit");
    assert_eq!(
        string_field(field(licenses, "algorithm"), "sha256"),
        "8748956d2e5afe9dfc8311188b4119dacc7c5293b0561e7cca7a21cf80e54caa"
    );
    assert_eq!(
        string_field(field(licenses, "speechbrain"), "spdx"),
        "apache-2.0"
    );
    assert_eq!(
        string_field(field(licenses, "speechbrain"), "sha256"),
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    );
    assert_eq!(string_field(licenses, "checkpoint"), "apache-2.0");
    let identity = field(&root, "identity");
    assert_eq!(
        string_field(identity, "reference_format"),
        "vokra-sgmse-score-reference-v1"
    );
    assert_eq!(
        string_field(identity, "reference_tool"),
        "sgmse_dump_reference.py"
    );
    let ema = field(&root, "ema_route");
    assert_eq!(
        string_field(ema, "status"),
        "SOURCE_ROUTE_VERIFIED_STRICT_LOAD"
    );
    assert!(!bool_field(ema, "unsafe_pickle_fallback"));
    let input = field(&root, "input");
    assert_eq!(u64_field(input, "seed"), 20260901);
    assert_eq!(u64_field(input, "sample_rate"), 16000);
    assert_eq!(u64_field(input, "n_fft"), 510);
    assert_eq!(u64_field(input, "frequency_bins"), 256);
    assert_eq!(u64_field(input, "frames"), 64);
    assert_eq!(string_field(input, "forward_signature"), "(x_t, y, t)");
    let artifacts = field(&root, "artifacts")
        .as_object()
        .expect("manifest artifacts object");
    assert_eq!(artifacts.len(), ARTIFACTS.len());
    for (name, expected_sha) in ARTIFACTS {
        let metadata = artifacts
            .get(*name)
            .unwrap_or_else(|| panic!("missing artifact {name}"));
        assert_eq!(string_field(metadata, "path"), format!("{name}.f32"));
        assert_eq!(string_field(metadata, "dtype"), "float32");
        assert_eq!(u64_field(metadata, "count"), COUNT as u64);
        assert_eq!(u64_field(metadata, "bytes"), BYTES as u64);
        assert_eq!(string_field(metadata, "sha256"), *expected_sha);
        read_plane(&reference.join(format!("{name}.f32")), expected_sha);
    }
    let log = reference.join("run.log");
    assert!(log.is_file() && !log.is_symlink());
    let log_metadata = field(&root, "run_log");
    assert_eq!(string_field(log_metadata, "path"), "run.log");
    assert_eq!(
        u64_field(log_metadata, "size"),
        log.metadata().unwrap().len()
    );
    assert_eq!(string_field(log_metadata, "sha256"), sha256_file(&log));
    let text = String::from_utf8(fs::read(&log).unwrap()).expect("run log UTF-8");
    for marker in [
        "status=REFERENCE_COMPLETE_NO_UPLOAD",
        "fixture_payload=retained_for_native_parity",
        "publication=NO_UPLOAD",
    ] {
        assert!(text.contains(marker), "run log missing {marker}");
    }
}

#[test]
fn sha256_known_vectors() {
    let vectors: &[(&[u8], &str)] = &[
        (
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
    ];
    for (input, expected) in vectors {
        let mut hash = Sha256::new();
        hash.update(input);
        assert_eq!(hash.finish(), *expected);
    }
    let mut hash = Sha256::new();
    hash.update(&[b'a'; 64]);
    assert_eq!(
        hash.finish(),
        "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"
    );
    let mut hash = Sha256::new();
    hash.update(&[b'a'; 56]);
    assert_eq!(
        hash.finish(),
        "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"
    );
    let mut hash = Sha256::new();
    hash.update(&[b'a'; 57]);
    assert_eq!(
        hash.finish(),
        "f13b2d724659eb3bf47f2dd6af1accc87b81f09f59f2b75e5c0bed6589dfe8c6"
    );
}

fn max_abs(a: &[f32], b: &[f32]) -> (usize, f32) {
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(i, (x, y))| (i, (x - y).abs()))
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .unwrap()
}
fn assert_parity(label: &str, actual: &[f32], expected: &[f32]) {
    assert!(actual.iter().all(|v| v.is_finite()), "{label} non-finite");
    let (index, max) = max_abs(actual, expected);
    assert!(
        max <= ATOL,
        "{label} max_abs={max:.9e} index={index} exceeds atol={ATOL}"
    );
    eprintln!("SGMSE_APPLE_PARITY {label} max_abs={max:.9e} index={index} atol={ATOL:.9e} PASS");
}
fn write_evidence(path: &Path, values: &[f32]) {
    let mut f = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    for value in values {
        assert!(value.is_finite());
        f.write_all(&value.to_le_bytes()).unwrap();
    }
    f.sync_all().unwrap();
}

fn verify_evidence(path: &Path) {
    let expected: BTreeSet<&str> = [
        "cpu_score_real.f32",
        "cpu_score_imag.f32",
        "metal_score_real.f32",
        "metal_score_imag.f32",
        "backend.txt",
    ]
    .into_iter()
    .collect();
    let actual: BTreeSet<String> = fs::read_dir(path)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        actual,
        expected.iter().map(|name| (*name).to_owned()).collect()
    );
    for name in expected.iter().filter(|name| **name != "backend.txt") {
        let file = path.join(name);
        assert!(file.is_file() && !file.is_symlink());
        assert_eq!(file.metadata().unwrap().len() as usize, BYTES);
    }
    let backend = path.join("backend.txt");
    assert!(backend.is_file() && !backend.is_symlink());
    let text = String::from_utf8(fs::read(backend).unwrap()).unwrap();
    assert!(text.contains("backend=cpu,metal"));
    assert!(text.contains("metal_device=present"));
    assert!(text.contains("atol=0.01"));
    let manifest_sha = std::env::var(REFERENCE_MANIFEST_SHA_ENV).unwrap();
    assert!(text.contains(&format!("reference_manifest_sha256={manifest_sha}")));
    assert!(text.contains("verdict=PASS"));
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
#[test]
fn apple_sgmse_contract_is_unavailable_off_apple_metal() {
    let Err(error) = Compute::for_backend(BackendKind::Metal, SGMSE_HOT_OPS) else {
        panic!("off-feature Apple SGMSE must not claim Metal support");
    };
    assert!(matches!(error, VokraError::BackendUnavailable(_)));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "real Apple Silicon SGMSE score parity is reserved for disposable Scaleway"]
fn sgmse_apple_cpu_metal_score_matches_reference() {
    assert_eq!(std::env::consts::OS, "macos");
    assert_eq!(std::env::consts::ARCH, "aarch64");
    assert_eq!(std::env::var(REMOTE_ENV).as_deref(), Ok("1"));
    let gguf = env_path(GGUF_ENV);
    let reference = env_path(REFERENCE_ENV);
    let evidence = env_path(EVIDENCE_ENV);
    let expected_gguf = std::env::var(GGUF_SHA_ENV).expect("GGUF SHA required");
    let expected_manifest =
        std::env::var(REFERENCE_MANIFEST_SHA_ENV).expect("reference manifest SHA required");
    assert!(expected_gguf.len() == 64 && expected_gguf.bytes().all(|b| b.is_ascii_hexdigit()));
    assert!(
        expected_manifest.len() == 64 && expected_manifest.bytes().all(|b| b.is_ascii_hexdigit())
    );
    require_abs(&gguf, GGUF_ENV);
    require_abs(&reference, REFERENCE_ENV);
    require_abs(&evidence, EVIDENCE_ENV);
    reject_symlink_ancestry(&gguf, GGUF_ENV);
    reject_symlink_ancestry(&reference, REFERENCE_ENV);
    assert!(gguf.is_file() && !gguf.is_symlink());
    assert!(reference.is_dir() && !reference.is_symlink());
    assert!(!evidence.exists() && !evidence.is_symlink());
    let parent = evidence.parent().unwrap();
    assert!(parent.is_dir() && !parent.is_symlink());
    reject_symlink_ancestry(parent, "evidence parent");
    let evidence_candidate = parent.join(evidence.file_name().expect("evidence name"));
    assert!(!overlap(&gguf, &reference));
    assert!(!overlap(&gguf, &evidence_candidate));
    assert!(!overlap(&reference, &evidence_candidate));
    assert_eq!(sha256_file(&gguf), expected_gguf);
    assert_eq!(
        sha256_file(&reference.join("manifest.json")),
        expected_manifest
    );
    verify_reference(&reference);
    let noisy_real = read_plane(
        &reference.join("input_noisy_real.f32"),
        ARTIFACTS
            .iter()
            .find(|(n, _)| *n == "input_noisy_real")
            .unwrap()
            .1,
    );
    let noisy_imag = read_plane(
        &reference.join("input_noisy_imag.f32"),
        ARTIFACTS
            .iter()
            .find(|(n, _)| *n == "input_noisy_imag")
            .unwrap()
            .1,
    );
    let cond_real = read_plane(
        &reference.join("input_condition_real.f32"),
        ARTIFACTS
            .iter()
            .find(|(n, _)| *n == "input_condition_real")
            .unwrap()
            .1,
    );
    let cond_imag = read_plane(
        &reference.join("input_condition_imag.f32"),
        ARTIFACTS
            .iter()
            .find(|(n, _)| *n == "input_condition_imag")
            .unwrap()
            .1,
    );
    let mut state = Vec::with_capacity(2 * COUNT);
    state.extend_from_slice(&noisy_real);
    state.extend_from_slice(&noisy_imag);
    let mut condition = Vec::with_capacity(2 * COUNT);
    condition.extend_from_slice(&cond_real);
    condition.extend_from_slice(&cond_imag);
    let mut cpu_out = vec![0.0; 2 * COUNT];
    let mut metal_out = vec![0.0; 2 * COUNT];
    let file = vokra_mmap::open_gguf(&gguf).expect("open authenticated GGUF");
    let model = SgmseModel::from_gguf(&file).expect("bind authenticated SGMSE GGUF");
    model
        .score(&Compute::cpu(), &state, &condition, 0.5, &mut cpu_out)
        .expect("CPU score");
    let metal = Compute::for_backend(BackendKind::Metal, SGMSE_HOT_OPS)
        .unwrap_or_else(|e| panic!("Scaleway Metal unavailable: {e}"));
    model
        .score(&metal, &state, &condition, 0.5, &mut metal_out)
        .expect("Metal score");
    let ref_real = read_plane(
        &reference.join("score_real.f32"),
        ARTIFACTS
            .iter()
            .find(|(n, _)| *n == "score_real")
            .unwrap()
            .1,
    );
    let ref_imag = read_plane(
        &reference.join("score_imag.f32"),
        ARTIFACTS
            .iter()
            .find(|(n, _)| *n == "score_imag")
            .unwrap()
            .1,
    );
    assert_parity("cpu_score_real", &cpu_out[..COUNT], &ref_real);
    assert_parity("cpu_score_imag", &cpu_out[COUNT..], &ref_imag);
    assert_parity("metal_score_real", &metal_out[..COUNT], &ref_real);
    assert_parity("metal_score_imag", &metal_out[COUNT..], &ref_imag);
    assert_parity("metal_vs_cpu_real", &metal_out[..COUNT], &cpu_out[..COUNT]);
    assert_parity("metal_vs_cpu_imag", &metal_out[COUNT..], &cpu_out[COUNT..]);
    fs::create_dir(&evidence).expect("create absent evidence directory");
    write_evidence(&evidence.join("cpu_score_real.f32"), &cpu_out[..COUNT]);
    write_evidence(&evidence.join("cpu_score_imag.f32"), &cpu_out[COUNT..]);
    write_evidence(&evidence.join("metal_score_real.f32"), &metal_out[..COUNT]);
    write_evidence(&evidence.join("metal_score_imag.f32"), &metal_out[COUNT..]);
    fs::write(
        evidence.join("backend.txt"),
        format!(
            "backend=cpu,metal\nmetal_device=present\natol=0.01\nreference_manifest_sha256={}\nverdict=PASS\n",
            std::env::var(REFERENCE_MANIFEST_SHA_ENV).unwrap()
        ),
    )
    .unwrap();
    verify_evidence(&evidence);
    eprintln!(
        "SGMSE_APPLE_SCORE_PARITY backend=cpu+metal reference=verified atol=0.01 verdict=PASS evidence={}",
        evidence.display()
    );
}
