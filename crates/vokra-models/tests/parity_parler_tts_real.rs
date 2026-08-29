//! Gated real-public-GGUF parity against pinned official Parler-TTS.
//!
//! Generate references on VAST through
//! `scripts/publish/vast-ai/run-parler-tts-validation.sh`. Each variant
//! requires a complete GGUF/reference environment pair; an unset pair is a
//! documented skip and a partial pair fails loudly. The same fixture can run
//! on remote Apple Silicon with `VOKRA_PARLER_BACKEND=metal`.

use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_models::parler::{
    ParlerGeneratedCodes, ParlerGenerationConfig, ParlerModel, ParlerVariant,
};

// Same ceiling as the existing official Transformers T5-base parity gate.
const TEXT_HIDDEN_ATOL: f32 = 0.01;
const PCM_ATOL: f32 = 0.01;
const MAX_FRAMES: usize = 4;
const CODEBOOKS: usize = 9;
const FRAME_HOP: usize = 512;
const PARLER_SOURCE_REVISION: &str = "d108732cd57788ec86bc857d99a6cabd66663d68";
const ENGLISH_UPSTREAM_REVISION: &str = "0392b9451a601e528fd863bbb0598431fee810d9";
const MULTILINGUAL_UPSTREAM_REVISION: &str = "11b27d57855dec1ce0914ba1f12363bf2ea75ba3";

fn verify_manifest(reference_dir: &Path, prefix: &str) {
    let path = reference_dir.join("manifest.json");
    let manifest = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!("read official Parler manifest {}: {error}", path.display())
    });
    let (variant, upstream_revision) = match prefix {
        "ENGLISH" => ("english", ENGLISH_UPSTREAM_REVISION),
        "MULTILINGUAL" => ("multilingual", MULTILINGUAL_UPSTREAM_REVISION),
        other => panic!("unknown Parler parity prefix {other}"),
    };
    for required in [
        "\"format\": \"vokra-parler-tts-official-reference-v1\"".to_owned(),
        format!("\"variant\": \"{variant}\""),
        format!("\"upstream_revision\": \"{upstream_revision}\""),
        format!("\"parler_source_revision\": \"{PARLER_SOURCE_REVISION}\""),
        "\"transformers_version\": \"4.46.1\"".to_owned(),
    ] {
        assert!(
            manifest.contains(&required),
            "official Parler manifest {} is missing immutable provenance field {required:?}",
            path.display()
        );
    }
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read Parler fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read Parler fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn parity_paths(prefix: &str) -> Option<(PathBuf, PathBuf)> {
    let gguf_key = format!("VOKRA_PARLER_{prefix}_GGUF");
    let reference_key = format!("VOKRA_PARLER_{prefix}_PARITY_DIR");
    let gguf = std::env::var_os(&gguf_key).map(PathBuf::from);
    let reference = std::env::var_os(&reference_key).map(PathBuf::from);
    match (gguf, reference) {
        (None, None) => {
            eprintln!("skipping Parler {prefix} real parity: set {gguf_key} and {reference_key}");
            None
        }
        (Some(gguf), Some(reference)) => Some((gguf, reference)),
        _ => panic!(
            "Parler {prefix} real parity is partially configured; set both {gguf_key} and {reference_key}"
        ),
    }
}

fn selected_backend() -> BackendKind {
    match std::env::var("VOKRA_PARLER_BACKEND").as_deref() {
        Ok("metal") => BackendKind::Metal,
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok(other) => panic!("VOKRA_PARLER_BACKEND must be cpu or metal, got {other:?}"),
    }
}

fn error_metrics(actual: &[f32], expected: &[f32], label: &str) -> (f32, f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Parler {label} extent differs"
    );
    assert!(!actual.is_empty(), "Parler {label} is empty");
    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    for (&left, &right) in actual.iter().zip(expected) {
        assert!(
            left.is_finite() && right.is_finite(),
            "Parler {label} contains a non-finite value"
        );
        let delta = f64::from(left) - f64::from(right);
        max_abs = max_abs.max(delta.abs() as f32);
        squared_error += delta * delta;
    }
    (max_abs, (squared_error / actual.len() as f64).sqrt())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn compare_metal_with_cpu(
    prefix: &str,
    gguf_path: &Path,
    expected_variant: ParlerVariant,
    description: &[u32],
    prompt: &[u32],
    expected_hidden: &[f32],
    expected_codes: &[u32],
    expected_pcm: &[f32],
    frames: usize,
    official_packet: &ParlerGeneratedCodes,
    metal_hidden: &[f32],
    metal_generated: &ParlerGeneratedCodes,
    metal_official_pcm: &[f32],
    metal_end_to_end_pcm: &[f32],
) {
    let cpu_model = ParlerModel::open_mapped_with_backend(gguf_path, BackendKind::Cpu)
        .expect("strict CPU mapping for direct Parler Metal-vs-CPU comparison");
    assert_eq!(cpu_model.variant(), expected_variant);

    let cpu_hidden = cpu_model
        .encode_description(description, None)
        .expect("native Parler CPU T5 encoding for Metal-vs-CPU comparison");
    let (cpu_hidden_max_abs, cpu_hidden_rmse) =
        error_metrics(&cpu_hidden, expected_hidden, "CPU FLAN-T5 hidden state");
    assert!(
        cpu_hidden_max_abs <= TEXT_HIDDEN_ATOL,
        "Parler {prefix} CPU T5 max_abs={cpu_hidden_max_abs}, rmse={cpu_hidden_rmse}, exceeding {TEXT_HIDDEN_ATOL}"
    );
    let (hidden_max_abs, hidden_rmse) = error_metrics(
        &cpu_hidden,
        metal_hidden,
        "CPU-vs-Metal FLAN-T5 hidden state",
    );
    assert!(
        hidden_max_abs <= TEXT_HIDDEN_ATOL,
        "Parler {prefix} CPU-vs-Metal T5 max_abs={hidden_max_abs}, rmse={hidden_rmse}, exceeding {TEXT_HIDDEN_ATOL}"
    );

    let cpu_generated = cpu_model
        .generate_codes(
            description,
            None,
            prompt,
            &ParlerGenerationConfig::greedy(MAX_FRAMES),
        )
        .expect("native Parler CPU greedy generation for Metal-vs-CPU comparison");
    assert_eq!(
        cpu_generated.as_frame_major(),
        expected_codes,
        "Parler {prefix} CPU generated packet differs from the official oracle"
    );
    assert_eq!(
        cpu_generated.as_frame_major(),
        metal_generated.as_frame_major(),
        "Parler {prefix} CPU and Metal generated code packets differ"
    );

    let cpu_official_pcm = cpu_model
        .decode_codes(official_packet)
        .expect("native Parler CPU decode of the official packet");
    let (official_max_abs, official_rmse) = error_metrics(
        &cpu_official_pcm.samples,
        metal_official_pcm,
        "CPU-vs-Metal official PCM",
    );
    assert!(
        official_max_abs <= PCM_ATOL,
        "Parler {prefix} CPU-vs-Metal official PCM max_abs={official_max_abs}, rmse={official_rmse}, exceeding {PCM_ATOL}"
    );
    let (official_cpu_max_abs, official_cpu_rmse) = error_metrics(
        &cpu_official_pcm.samples,
        expected_pcm,
        "CPU official-packet PCM",
    );
    assert!(
        official_cpu_max_abs <= PCM_ATOL,
        "Parler {prefix} CPU official-packet PCM max_abs={official_cpu_max_abs}, rmse={official_cpu_rmse}, exceeding {PCM_ATOL}"
    );

    let cpu_end_to_end_pcm = cpu_model
        .decode_codes(&cpu_generated)
        .expect("native Parler CPU end-to-end decode");
    let (end_to_end_max_abs, end_to_end_rmse) = error_metrics(
        &cpu_end_to_end_pcm.samples,
        metal_end_to_end_pcm,
        "CPU-vs-Metal end-to-end PCM",
    );
    assert!(
        end_to_end_max_abs <= PCM_ATOL,
        "Parler {prefix} CPU-vs-Metal end-to-end PCM max_abs={end_to_end_max_abs}, rmse={end_to_end_rmse}, exceeding {PCM_ATOL}"
    );
    let (end_to_end_cpu_max_abs, end_to_end_cpu_rmse) = error_metrics(
        &cpu_end_to_end_pcm.samples,
        expected_pcm,
        "CPU end-to-end PCM",
    );
    assert!(
        end_to_end_cpu_max_abs <= PCM_ATOL,
        "Parler {prefix} CPU end-to-end PCM max_abs={end_to_end_cpu_max_abs}, rmse={end_to_end_cpu_rmse}, exceeding {PCM_ATOL}"
    );
    eprintln!(
        "PARLER_APPLE_PARITY variant={} metal_vs_cpu=PASS",
        prefix.to_ascii_lowercase()
    );
    assert_eq!(official_packet.frames(), frames);
}

fn run_variant(prefix: &str, expected_variant: ParlerVariant) {
    let Some((gguf_path, reference_dir)) = parity_paths(prefix) else {
        return;
    };
    verify_manifest(&reference_dir, prefix);
    let description = read_u32(&reference_dir.join("description_token_ids.u32le"));
    let prompt = read_u32(&reference_dir.join("prompt_token_ids.u32le"));
    let expected_hidden = read_f32(&reference_dir.join("text_hidden.f32"));
    let expected_codes = read_u32(&reference_dir.join("codes.u32le"));
    let expected_pcm = read_f32(&reference_dir.join("decoded_pcm.f32"));
    assert_eq!(expected_codes.len() % CODEBOOKS, 0);
    let frames = expected_codes.len() / CODEBOOKS;
    assert!((1..=MAX_FRAMES).contains(&frames));
    assert_eq!(expected_pcm.len(), frames * FRAME_HOP);

    let backend = selected_backend();
    let model = ParlerModel::open_mapped_with_backend(&gguf_path, backend)
        .expect("strict mapping-owned Parler bind and complete backend preflight");
    assert_eq!(model.variant(), expected_variant);

    let hidden = model
        .encode_description(&description, None)
        .expect("native Parler FLAN-T5 description encoding");
    let (hidden_max_abs, hidden_rmse) =
        error_metrics(&hidden, &expected_hidden, "FLAN-T5 hidden state");
    assert!(
        hidden_max_abs <= TEXT_HIDDEN_ATOL,
        "Parler {prefix} {backend:?} T5 max_abs {hidden_max_abs} exceeds FP32 ceiling {TEXT_HIDDEN_ATOL}"
    );

    let generated = model
        .generate_codes(
            &description,
            None,
            &prompt,
            &ParlerGenerationConfig::greedy(MAX_FRAMES),
        )
        .expect("native Parler greedy delayed generation");
    assert_eq!(
        generated.as_frame_major(),
        expected_codes.as_slice(),
        "Parler {prefix} {backend:?} frame-major code packet differs from the official oracle"
    );

    let official_packet = ParlerGeneratedCodes::from_frame_major(expected_codes.clone(), frames)
        .expect("official Parler frame-major packet");
    let decoded = model
        .decode_codes(&official_packet)
        .expect("native Parler decode of official packet");
    let (decode_max_abs, decode_rmse) = error_metrics(
        &decoded.samples,
        &expected_pcm,
        "official-packet decoded PCM",
    );
    assert!(
        decode_max_abs <= PCM_ATOL,
        "Parler {prefix} {backend:?} decoder max_abs {decode_max_abs} exceeds FP32 ceiling {PCM_ATOL}"
    );

    let end_to_end = model
        .decode_codes(&generated)
        .expect("native Parler end-to-end decode");
    let (end_to_end_max_abs, end_to_end_rmse) =
        error_metrics(&end_to_end.samples, &expected_pcm, "end-to-end PCM");
    eprintln!(
        "Parler {prefix} {backend:?}: frames={frames}, T5_max_abs={hidden_max_abs:.9e}, T5_rmse={hidden_rmse:.9e}, codes=exact, decode_max_abs={decode_max_abs:.9e}, decode_rmse={decode_rmse:.9e}, end_to_end_max_abs={end_to_end_max_abs:.9e}, end_to_end_rmse={end_to_end_rmse:.9e}"
    );
    assert!(
        end_to_end_max_abs <= PCM_ATOL,
        "Parler {prefix} {backend:?} end-to-end max_abs {end_to_end_max_abs} exceeds FP32 ceiling {PCM_ATOL}"
    );

    #[cfg(all(feature = "metal", target_os = "macos"))]
    if backend == BackendKind::Metal {
        compare_metal_with_cpu(
            prefix,
            &gguf_path,
            expected_variant,
            &description,
            &prompt,
            &expected_hidden,
            &expected_codes,
            &expected_pcm,
            frames,
            &official_packet,
            &hidden,
            &generated,
            &decoded.samples,
            &end_to_end.samples,
        );
    }
}

#[test]
fn real_parler_english_matches_official() {
    run_variant("ENGLISH", ParlerVariant::MiniV1English);
}

#[test]
fn real_parler_multilingual_matches_official() {
    run_variant("MULTILINGUAL", ParlerVariant::MiniMultilingualV11);
}
