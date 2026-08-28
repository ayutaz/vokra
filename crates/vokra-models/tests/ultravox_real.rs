//! Gated real-public-GGUF parity for Ultravox v0.5.
//!
//! The reference is generated only on VAST by Fixie's authenticated official
//! custom model and processor in `tools/parity/ultravox/dump_reference.py`.
//! An unset three-path configuration skips honestly; a partial configuration
//! fails. `VOKRA_ULTRAVOX_BACKEND=metal` is reserved for the guarded remote
//! Apple Silicon worker.

use std::collections::BTreeMap;
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
        assert_eq!(manifest_value(&manifest, "transformers_version"), "4.48.1");
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

fn configured_paths() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let gguf = std::env::var_os("VOKRA_ULTRAVOX_GGUF").map(PathBuf::from);
    let companion = std::env::var_os("VOKRA_ULTRAVOX_COMPANION_GGUF").map(PathBuf::from);
    let reference = std::env::var_os("VOKRA_ULTRAVOX_REFERENCE_DIR").map(PathBuf::from);
    match (gguf, companion, reference) {
        (None, None, None) => {
            eprintln!(
                "skip Ultravox official parity: set VOKRA_ULTRAVOX_GGUF, VOKRA_ULTRAVOX_COMPANION_GGUF and VOKRA_ULTRAVOX_REFERENCE_DIR"
            );
            None
        }
        (Some(gguf), Some(companion), Some(reference)) => Some((gguf, companion, reference)),
        _ => {
            panic!("Ultravox parity is partially configured; set all three VOKRA_ULTRAVOX_* paths")
        }
    }
}

fn selected_backend() -> BackendKind {
    match std::env::var("VOKRA_ULTRAVOX_BACKEND").as_deref() {
        Ok("metal") => BackendKind::Metal,
        Ok("cpu") | Err(_) => BackendKind::Cpu,
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
fn ultravox_public_cpu_or_metal_matches_official_reference() {
    let Some((gguf, companion_gguf, reference_dir)) = configured_paths() else {
        return;
    };
    assert_eq!(
        std::fs::metadata(&gguf).expect("stat Ultravox GGUF").len(),
        PUBLIC_FILE_BYTES,
        "public file byte identity; expected {PUBLIC_FILENAME}@{PUBLIC_VOKRA_REVISION} SHA-256 {PUBLIC_FILE_SHA256}"
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
