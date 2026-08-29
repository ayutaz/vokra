//! Gated real-public-GGUF parity for Ultravox v0.5.
//!
//! The reference is generated only on VAST by Fixie's authenticated official
//! custom model and processor in `tools/parity/ultravox/dump_reference.py`.
//! All input paths, the companion hash, and the backend must be configured;
//! partial configuration is a hard failure. An entirely unconfigured workspace
//! test skips honestly and is never accepted as VAST/Apple parity evidence;
//! those scripts require the named test, result, and independent-reference
//! sentinel. `metal` is reserved for the guarded remote Apple Silicon worker.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use vokra_core::LicenseClass;
use vokra_core::backend::BackendKind;
use vokra_core::compliance::CompliancePolicy;
use vokra_models::ultravox::{
    COMPANION_SOURCE_REVISION, COMPANION_UPSTREAM_HF, PUBLIC_FILE_BYTES, PUBLIC_FILE_SHA256,
    PUBLIC_FILENAME, PUBLIC_VOKRA_REVISION, SAMPLE_RATE, UPSTREAM_REVISION, UltravoxAudioTower,
    UltravoxGenerationOptions, UltravoxLlamaCompanion,
};
use vokra_models::whisper::mel::log_mel_variable;

const REFERENCE_SCHEMA: &str = "vokra-ultravox-reference-v1";
const FP32_ATOL: f32 = 0.01;
const EXPECTED_MODEL_SOURCE_SHA256: &str =
    "df618218561375da01bb53bd2764ea123e0cbf782f3326753f669f63ff6c6d3f";
const EXPECTED_PROCESSOR_SOURCE_SHA256: &str =
    "2ae6682f3deecb22539fae6a6631688fc1675282f1a5b31145d9f95d2347ff7b";
const EXPECTED_CONFIG_SOURCE_SHA256: &str =
    "99cf5ad911189f2351c2232234025db56b23763283583c0a848ebf2a1ecc40fc";
const EXPECTED_UPSTREAM_MODEL_SHA256: &str =
    "f3a3bf7e9137f3219a0d27ba71668deeee8c60aaf0ea587b48d8f71178763f31";

#[derive(Debug)]
struct Reference {
    pcm: Vec<f32>,
    input_features: Vec<f32>,
    audio_frames: usize,
    audio_embeddings: Vec<f32>,
    audio_token_len: usize,
    prompt_ids: Vec<u32>,
    audio_token_start_idx: usize,
    next_logits: Vec<f32>,
    generated_ids: Vec<u32>,
    stop_token_ids: Vec<u32>,
    max_new_tokens: usize,
}

fn read_manifest(path: &Path) -> BTreeMap<String, String> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read Ultravox manifest {}: {error}", path.display()));
    let mut values = BTreeMap::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .unwrap_or_else(|| panic!("manifest line {} has no '=': {line:?}", line_number + 1));
        assert!(
            !key.is_empty(),
            "empty manifest key at line {}",
            line_number + 1
        );
        assert!(
            values.insert(key.to_owned(), value.to_owned()).is_none(),
            "duplicate manifest key {key:?}"
        );
    }
    values
}

fn manifest_value<'a>(values: &'a BTreeMap<String, String>, key: &str) -> &'a str {
    values
        .get(key)
        .unwrap_or_else(|| panic!("Ultravox reference manifest is missing {key:?}"))
}

fn manifest_usize(values: &BTreeMap<String, String>, key: &str) -> usize {
    manifest_value(values, key)
        .parse()
        .unwrap_or_else(|_| panic!("Ultravox manifest {key:?} is not usize"))
}

fn manifest_u32_csv(values: &BTreeMap<String, String>, key: &str) -> Vec<u32> {
    let raw = manifest_value(values, key);
    assert!(!raw.is_empty(), "Ultravox manifest {key:?} is empty");
    raw.split(',')
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("Ultravox manifest {key:?} contains non-u32 {value:?}"))
        })
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(bytes.len() % 4, 0, "{path:?} is truncated");
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(bytes.len() % 4, 0, "{path:?} is truncated");
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{path:?} contains a non-finite reference value"
    );
    values
}

fn sha256_file(path: &Path) -> String {
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
    fn compress(h: &mut [u32; 8], block: &[u8], k: &[u32; 64]) {
        let mut w = [0u32; 64];
        for (index, word) in block.chunks_exact(4).enumerate() {
            w[index] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (value, add) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *value = value.wrapping_add(add);
        }
    }
    let mut h = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut file = File::open(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    let mut pending = Vec::with_capacity(64);
    let mut chunk = [0u8; 8 * 1024 * 1024];
    let mut bit_len = 0u64;
    loop {
        let read = file
            .read(&mut chunk)
            .unwrap_or_else(|error| panic!("read {path:?}: {error}"));
        if read == 0 {
            break;
        }
        bit_len = bit_len
            .checked_add((read as u64).checked_mul(8).unwrap())
            .expect("SHA-256 input exceeds u64 bit length");
        pending.extend_from_slice(&chunk[..read]);
        let complete = pending.len() / 64 * 64;
        for block in pending[..complete].chunks_exact(64) {
            compress(&mut h, block, &K);
        }
        pending.drain(..complete);
    }
    pending.push(0x80);
    while pending.len() % 64 != 56 {
        pending.push(0);
    }
    pending.extend_from_slice(&bit_len.to_be_bytes());
    for block in pending.chunks_exact(64) {
        compress(&mut h, block, &K);
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

fn manifest_hash(manifest: &BTreeMap<String, String>, file: &str) -> &str {
    manifest_value(manifest, &format!("sha256_{}", file.replace('.', "_")))
}

fn verify_manifest_hashes(directory: &Path, manifest: &BTreeMap<String, String>) {
    for file in [
        "pcm.f32le",
        "input_features.f32le",
        "audio_embeddings.f32le",
        "prompt_ids.u32le",
        "next_logits.f32le",
        "generated_ids.u32le",
        "source_files.json",
        "environment.json",
    ] {
        assert_eq!(
            sha256_file(&directory.join(file)),
            manifest_hash(manifest, file),
            "reference {file} differs from its authenticated manifest hash"
        );
    }
}

impl Reference {
    fn load(directory: &Path) -> Self {
        let manifest = read_manifest(&directory.join("manifest.txt"));
        assert_eq!(manifest_value(&manifest, "schema"), REFERENCE_SCHEMA);
        assert_eq!(
            manifest_value(&manifest, "upstream_repo"),
            "fixie-ai/ultravox-v0_5-llama-3_2-1b"
        );
        assert_eq!(
            manifest_value(&manifest, "upstream_revision"),
            UPSTREAM_REVISION
        );
        assert_eq!(
            manifest_value(&manifest, "companion_repo"),
            COMPANION_UPSTREAM_HF
        );
        assert_eq!(
            manifest_value(&manifest, "companion_revision"),
            COMPANION_SOURCE_REVISION
        );
        assert_eq!(manifest_value(&manifest, "transformers_version"), "5.5.0");
        assert_eq!(
            manifest_value(&manifest, "public_repo"),
            "vokra/ultravox-v0-5-llama-3-2-1b"
        );
        assert_eq!(
            manifest_value(&manifest, "public_revision"),
            PUBLIC_VOKRA_REVISION
        );
        assert_eq!(
            manifest_value(&manifest, "public_filename"),
            PUBLIC_FILENAME
        );
        assert_eq!(
            manifest_usize(&manifest, "public_file_bytes"),
            PUBLIC_FILE_BYTES as usize
        );
        assert_eq!(
            manifest_value(&manifest, "public_file_sha256"),
            PUBLIC_FILE_SHA256
        );
        assert_eq!(
            manifest_value(&manifest, "public_weights_sha256"),
            EXPECTED_UPSTREAM_MODEL_SHA256
        );
        assert_eq!(
            manifest_value(&manifest, "source_ultravox_model_sha256"),
            EXPECTED_MODEL_SOURCE_SHA256
        );
        assert_eq!(
            manifest_value(&manifest, "source_ultravox_processing_sha256"),
            EXPECTED_PROCESSOR_SOURCE_SHA256
        );
        assert_eq!(
            manifest_value(&manifest, "source_ultravox_config_sha256"),
            EXPECTED_CONFIG_SOURCE_SHA256
        );
        verify_manifest_hashes(directory, &manifest);
        assert_eq!(
            manifest_usize(&manifest, "sample_rate"),
            SAMPLE_RATE as usize
        );
        assert_eq!(manifest_usize(&manifest, "n_mels"), 128);
        assert_eq!(manifest_usize(&manifest, "vocab_size"), 128_256);
        assert_eq!(manifest_usize(&manifest, "audio_embedding_hidden"), 2_048);

        let pcm = read_f32(&directory.join("pcm.f32le"));
        let input_features = read_f32(&directory.join("input_features.f32le"));
        let audio_embeddings = read_f32(&directory.join("audio_embeddings.f32le"));
        let prompt_ids = read_u32(&directory.join("prompt_ids.u32le"));
        let next_logits = read_f32(&directory.join("next_logits.f32le"));
        let generated_ids = read_u32(&directory.join("generated_ids.u32le"));
        let audio_frames = manifest_usize(&manifest, "audio_frames");
        let audio_token_len = manifest_usize(&manifest, "audio_token_len");
        assert_eq!(pcm.len(), manifest_usize(&manifest, "sample_count"));
        assert_eq!(input_features.len(), 128 * audio_frames);
        assert_eq!(audio_embeddings.len(), audio_token_len * 2_048);
        assert_eq!(prompt_ids.len(), manifest_usize(&manifest, "prompt_tokens"));
        assert_eq!(next_logits.len(), 128_256);
        assert_eq!(
            generated_ids.len(),
            manifest_usize(&manifest, "generated_tokens")
        );
        assert_eq!(
            prompt_ids,
            manifest_u32_csv(&manifest, "prompt_ids_csv"),
            "binary and text prompt IDs differ"
        );
        Self {
            pcm,
            input_features,
            audio_frames,
            audio_embeddings,
            audio_token_len,
            prompt_ids,
            audio_token_start_idx: manifest_usize(&manifest, "audio_token_start_idx"),
            next_logits,
            generated_ids,
            stop_token_ids: manifest_u32_csv(&manifest, "stop_token_ids_csv"),
            max_new_tokens: manifest_usize(&manifest, "max_new_tokens"),
        }
    }
}

fn configured_paths() -> Option<(PathBuf, PathBuf, PathBuf, String)> {
    let gguf = std::env::var_os("VOKRA_ULTRAVOX_GGUF").map(PathBuf::from);
    let companion = std::env::var_os("VOKRA_ULTRAVOX_COMPANION_GGUF").map(PathBuf::from);
    let reference = std::env::var_os("VOKRA_ULTRAVOX_REFERENCE_DIR").map(PathBuf::from);
    let companion_sha = std::env::var_os("VOKRA_ULTRAVOX_COMPANION_GGUF_SHA256");
    let backend = std::env::var_os("VOKRA_ULTRAVOX_BACKEND");
    match (gguf, companion, reference) {
        (Some(gguf), Some(companion), Some(reference)) => {
            let companion_sha = companion_sha
                .unwrap_or_else(|| {
                    panic!("Ultravox parity requires VOKRA_ULTRAVOX_COMPANION_GGUF_SHA256")
                })
                .into_string()
                .unwrap_or_else(|_| panic!("companion SHA-256 must be valid UTF-8"));
            assert!(
                companion_sha.len() == 64
                    && companion_sha.bytes().all(|byte| byte.is_ascii_hexdigit())
                    && companion_sha.bytes().all(|byte| !byte.is_ascii_uppercase()),
                "VOKRA_ULTRAVOX_COMPANION_GGUF_SHA256 must be 64 lowercase hex characters"
            );
            Some((gguf, companion, reference, companion_sha))
        }
        (None, None, None) if companion_sha.is_none() && backend.is_none() => {
            eprintln!(
                "Ultravox real-weight parity inputs absent; skipping (not validation evidence)"
            );
            None
        }
        _ => {
            panic!("Ultravox parity is partially configured; set all paths, hash, and backend")
        }
    }
}

fn selected_backend() -> BackendKind {
    match std::env::var("VOKRA_ULTRAVOX_BACKEND").as_deref() {
        Ok("metal") => BackendKind::Metal,
        Ok("cpu") => BackendKind::Cpu,
        Err(_) => panic!("Ultravox parity requires VOKRA_ULTRAVOX_BACKEND=cpu or metal"),
        Ok(other) => panic!("VOKRA_ULTRAVOX_BACKEND must be cpu or metal, got {other:?}"),
    }
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    assert!(!actual.is_empty(), "{label} must not be empty");
    let mut worst_index = 0usize;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            left.is_finite(),
            "{label}: runtime value {index} is non-finite"
        );
        let delta = (left - right).abs();
        sum_abs += f64::from(delta);
        if delta > max_abs {
            max_abs = delta;
            worst_index = index;
        }
    }
    let mean_abs = sum_abs / actual.len() as f64;
    eprintln!(
        "ULTRAVOX_PARITY {label} max_abs={max_abs:.9e} index={worst_index} actual={:.9e} reference={:.9e} mean_abs={mean_abs:.9e} atol={FP32_ATOL:.9e}",
        actual[worst_index], expected[worst_index]
    );
    assert!(
        max_abs <= FP32_ATOL,
        "{label}: max_abs={max_abs} at {worst_index} exceeds FP32 atol={FP32_ATOL}; do not widen the bound to fit an observation"
    );
}

#[test]
fn ultravox_sha256_known_vector() {
    let path = std::env::temp_dir().join(format!("vokra-ultravox-sha256-{}", std::process::id()));
    std::fs::write(&path, b"abc").expect("write SHA-256 known vector");
    assert_eq!(
        sha256_file(&path),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    std::fs::remove_file(path).expect("remove SHA-256 known vector");
}

#[test]
fn ultravox_public_cpu_or_metal_matches_official_reference() {
    let Some((gguf, companion_gguf, reference_dir, companion_sha)) = configured_paths() else {
        return;
    };
    assert_eq!(
        std::fs::metadata(&gguf).expect("stat Ultravox GGUF").len(),
        PUBLIC_FILE_BYTES,
        "public file byte identity; expected {PUBLIC_FILENAME}@{PUBLIC_VOKRA_REVISION} SHA-256 {PUBLIC_FILE_SHA256}"
    );
    assert_eq!(
        sha256_file(&gguf),
        PUBLIC_FILE_SHA256,
        "public file SHA-256 identity; expected {PUBLIC_FILENAME}@{PUBLIC_VOKRA_REVISION}"
    );
    assert_eq!(
        sha256_file(&companion_gguf),
        companion_sha,
        "companion GGUF differs from the authenticated VAST input hash"
    );
    let reference = Reference::load(&reference_dir);
    let backend = selected_backend();
    let policy = CompliancePolicy::strict();
    let tower = UltravoxAudioTower::open_mapped_with_policy_and_backend(&gguf, &policy, backend)
        .unwrap_or_else(|error| panic!("bind Ultravox audio tower on {backend:?}: {error}"));
    let companion = UltravoxLlamaCompanion::open_mapped_with_policy_and_backend(
        &companion_gguf,
        &policy,
        backend,
    )
    .unwrap_or_else(|error| panic!("bind Ultravox Llama companion on {backend:?}: {error}"));
    assert_eq!(tower.backend(), backend);
    assert_eq!(companion.backend(), backend);
    assert_eq!(tower.weight_license(), LicenseClass::Permissive);
    assert_eq!(
        companion.weight_license(),
        LicenseClass::ConditionalCommercial
    );

    let (frontend, frontend_frames) = log_mel_variable(&reference.pcm, 128)
        .unwrap_or_else(|error| panic!("Ultravox variable log-mel: {error}"));
    assert_eq!(frontend_frames, reference.audio_frames);
    assert_close(
        &frontend,
        &reference.input_features,
        &format!("{backend:?}_frontend_vs_official"),
    );

    let runtime_audio = tower
        .encode_log_mel(&frontend, frontend_frames)
        .unwrap_or_else(|error| panic!("Ultravox end-to-end audio encode on {backend:?}: {error}"));
    assert_eq!(runtime_audio.frames(), reference.audio_token_len);
    assert_close(
        runtime_audio.values(),
        &reference.audio_embeddings,
        &format!("{backend:?}_audio_embeddings_vs_official"),
    );
    let logits = companion
        .next_token_logits_with_audio_embeddings(
            &reference.prompt_ids,
            reference.audio_token_start_idx,
            &runtime_audio,
        )
        .unwrap_or_else(|error| panic!("Ultravox logits on {backend:?}: {error}"));
    assert_close(
        &logits,
        &reference.next_logits,
        &format!("{backend:?}_next_logits_vs_official"),
    );

    let generated = companion
        .generate_with_audio_embeddings(
            &reference.prompt_ids,
            reference.audio_token_start_idx,
            &runtime_audio,
            &UltravoxGenerationOptions::greedy(reference.max_new_tokens, reference.stop_token_ids),
        )
        .unwrap_or_else(|error| panic!("Ultravox greedy generation on {backend:?}: {error}"));
    assert_eq!(
        generated.token_ids, reference.generated_ids,
        "official greedy token IDs"
    );
    eprintln!(
        "ULTRAVOX_PARITY {backend:?}_vs_official frontend_atol={FP32_ATOL} audio_embeddings_atol={FP32_ATOL} logits_atol={FP32_ATOL} greedy_ids=exact PASS"
    );
}
