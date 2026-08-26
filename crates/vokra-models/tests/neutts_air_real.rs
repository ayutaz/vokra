//! Gated real-public-GGUF parity for NeuTTS Air.
//!
//! The reference is generated only on VAST by the fixed official Transformers
//! model and Neuphonic source method in
//! `tools/parity/neutts_air/dump_reference.py`. An unset model/reference pair
//! skips honestly; a partial pair fails. `VOKRA_NEUTTS_AIR_BACKEND=metal` is
//! reserved for the guarded remote Apple worker.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_core::compliance::CompliancePolicy;
use vokra_models::neutts_air::{
    NeuTtsAir, NeuTtsAirCompanion, NeuTtsAirGenerationOptions, PUBLIC_FILE_BYTES,
    PUBLIC_FILE_SHA256, PUBLIC_FILENAME, PUBLIC_VOKRA_REVISION, UPSTREAM_REVISION,
    UPSTREAM_SOURCE_REVISION,
};

const REFERENCE_SCHEMA: &str = "vokra-neutts-air-reference-v1";
const FP32_ATOL: f32 = 0.01;
const EXPECTED_SOURCE_SHA256: &str =
    "e68b87dae6718903337a08eff56afbd58ba261d829624ea5a00a343c8cefb7c1";

#[derive(Debug)]
struct Reference {
    prompt_ids: Vec<u32>,
    next_logits: Vec<f32>,
    generated_ids: Vec<u32>,
    max_new_tokens: usize,
}

fn read_manifest(path: &Path) -> BTreeMap<String, String> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read NeuTTS Air manifest {}: {error}", path.display()));
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
        .unwrap_or_else(|| panic!("NeuTTS Air reference manifest is missing {key:?}"))
}

fn manifest_usize(values: &BTreeMap<String, String>, key: &str) -> usize {
    manifest_value(values, key)
        .parse()
        .unwrap_or_else(|_| panic!("NeuTTS Air manifest {key:?} is not usize"))
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
            "neuphonic/neutts-air"
        );
        assert_eq!(
            manifest_value(&manifest, "upstream_revision"),
            UPSTREAM_REVISION
        );
        assert_eq!(
            manifest_value(&manifest, "source_revision"),
            UPSTREAM_SOURCE_REVISION
        );
        assert_eq!(
            manifest_value(&manifest, "source_sha256"),
            EXPECTED_SOURCE_SHA256
        );
        assert_eq!(manifest_value(&manifest, "transformers_version"), "4.57.6");
        assert_eq!(manifest_usize(&manifest, "vocab_size"), 217_652);
        let prompt_ids = read_u32(&directory.join("prompt_ids.u32le"));
        let next_logits = read_f32(&directory.join("next_logits.f32le"));
        let generated_ids = read_u32(&directory.join("generated_ids.u32le"));
        assert_eq!(prompt_ids.len(), manifest_usize(&manifest, "prompt_tokens"));
        assert_eq!(next_logits.len(), 217_652);
        assert_eq!(
            generated_ids.len(),
            manifest_usize(&manifest, "generated_tokens")
        );
        Self {
            prompt_ids,
            next_logits,
            generated_ids,
            max_new_tokens: manifest_usize(&manifest, "max_new_tokens"),
        }
    }
}

fn configured_paths() -> Option<(PathBuf, PathBuf)> {
    let gguf = std::env::var_os("VOKRA_NEUTTS_AIR_GGUF").map(PathBuf::from);
    let reference = std::env::var_os("VOKRA_NEUTTS_AIR_REFERENCE_DIR").map(PathBuf::from);
    match (gguf, reference) {
        (None, None) => {
            eprintln!(
                "skip NeuTTS Air official parity: set VOKRA_NEUTTS_AIR_GGUF and VOKRA_NEUTTS_AIR_REFERENCE_DIR"
            );
            None
        }
        (Some(gguf), Some(reference)) => Some((gguf, reference)),
        _ => panic!(
            "NeuTTS Air parity is partially configured; set both VOKRA_NEUTTS_AIR_GGUF and VOKRA_NEUTTS_AIR_REFERENCE_DIR"
        ),
    }
}

fn selected_backend() -> BackendKind {
    match std::env::var("VOKRA_NEUTTS_AIR_BACKEND").as_deref() {
        Ok("metal") => BackendKind::Metal,
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok(other) => panic!("VOKRA_NEUTTS_AIR_BACKEND must be cpu or metal, got {other:?}"),
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
        "NEUTTS_AIR_PARITY {label} max_abs={max_abs:.9e} index={worst_index} actual={:.9e} reference={:.9e} mean_abs={mean_abs:.9e} atol={FP32_ATOL:.9e}",
        actual[worst_index], expected[worst_index]
    );
    assert!(
        max_abs <= FP32_ATOL,
        "{label}: max_abs={max_abs} at {worst_index} exceeds FP32 atol={FP32_ATOL}; do not widen the bound to fit an observation"
    );
}

#[test]
fn neutts_air_public_cpu_or_metal_matches_official_reference() {
    let Some((gguf, reference_dir)) = configured_paths() else {
        return;
    };
    assert_eq!(
        std::fs::metadata(&gguf)
            .expect("stat NeuTTS Air GGUF")
            .len(),
        PUBLIC_FILE_BYTES,
        "public file byte identity; expected {PUBLIC_FILENAME}@{PUBLIC_VOKRA_REVISION} SHA-256 {PUBLIC_FILE_SHA256}"
    );
    let reference = Reference::load(&reference_dir);
    let backend = selected_backend();
    let model =
        NeuTtsAir::open_mapped_with_policy_and_backend(&gguf, &CompliancePolicy::strict(), backend)
            .unwrap_or_else(|error| panic!("bind NeuTTS Air on {backend:?}: {error}"));
    assert_eq!(model.backend(), backend);
    let logits = model
        .next_token_logits(&reference.prompt_ids)
        .unwrap_or_else(|error| panic!("NeuTTS Air logits on {backend:?}: {error}"));
    assert_close(
        &logits,
        &reference.next_logits,
        &format!("{backend:?}_vs_official next_logits"),
    );
    let generated = model
        .generate_codes(
            &reference.prompt_ids,
            &NeuTtsAirGenerationOptions::greedy(reference.max_new_tokens),
        )
        .unwrap_or_else(|error| panic!("NeuTTS Air greedy generation on {backend:?}: {error}"));
    assert_eq!(
        generated.token_ids, reference.generated_ids,
        "official greedy token ids"
    );
    eprintln!(
        "NEUTTS_AIR_PARITY {backend:?}_vs_official logits_atol={FP32_ATOL} greedy_ids=exact PASS"
    );

    if let Some(companion_path) = std::env::var_os("VOKRA_NEUTTS_AIR_COMPANION_GGUF") {
        let companion = NeuTtsAirCompanion::from_path_with_policy_and_backend(
            companion_path,
            &CompliancePolicy::strict(),
            backend,
        )
        .unwrap_or_else(|error| panic!("bind NeuCodec companion on {backend:?}: {error}"));
        let synthesis = model
            .synthesize_with_companion(
                &companion,
                &reference.prompt_ids,
                &NeuTtsAirGenerationOptions::greedy(reference.max_new_tokens),
            )
            .unwrap_or_else(|error| panic!("compose NeuTTS Air on {backend:?}: {error}"));
        assert!(!synthesis.generation.codes.is_empty());
        assert_eq!(
            synthesis.pcm.len(),
            synthesis.generation.codes.len() * 480,
            "NeuCodec 50 Hz / 24 kHz timebase"
        );
        assert!(synthesis.pcm.iter().all(|sample| sample.is_finite()));
        eprintln!(
            "NEUTTS_AIR_COMPOSITION {backend:?} codes={} samples={} PASS",
            synthesis.generation.codes.len(),
            synthesis.pcm.len()
        );
    } else {
        eprintln!("skip NeuTTS Air composition smoke: set VOKRA_NEUTTS_AIR_COMPANION_GGUF");
    }
}
