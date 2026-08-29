//! Gated real-public-GGUF parity against pinned official Transformers 5.5.0.
//!
//! Generate references on VAST through
//! `scripts/publish/vast-ai/run-bark-validation.sh`. Each variant requires a
//! complete GGUF/reference environment pair; an unset pair is a documented
//! skip and a partial pair fails loudly. The same fixture can run on a remote
//! Apple Silicon host with `VOKRA_BARK_BACKEND=metal`.

use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_models::bark::{BarkGeneratedCodes, BarkGenerationConfig, BarkModel};

const FP32_ATOL: f32 = 0.01;
const MAX_SEMANTIC_TOKENS: usize = 4;
const CODEBOOKS: usize = 8;
const FRAME_HOP: usize = 320;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read Bark fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read Bark fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn parity_paths(prefix: &str) -> Option<(PathBuf, PathBuf)> {
    let gguf_key = format!("VOKRA_BARK_{prefix}_GGUF");
    let reference_key = format!("VOKRA_BARK_{prefix}_PARITY_DIR");
    let gguf = std::env::var_os(&gguf_key).map(PathBuf::from);
    let reference = std::env::var_os(&reference_key).map(PathBuf::from);
    match (gguf, reference) {
        (None, None) => {
            eprintln!("skipping Bark {prefix} real parity: set {gguf_key} and {reference_key}");
            None
        }
        (Some(gguf), Some(reference)) => Some((gguf, reference)),
        _ => panic!(
            "Bark {prefix} real parity is partially configured; set both {gguf_key} and {reference_key}"
        ),
    }
}

fn selected_backend() -> BackendKind {
    match std::env::var("VOKRA_BARK_BACKEND").as_deref() {
        Ok("metal") => BackendKind::Metal,
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok(other) => panic!("VOKRA_BARK_BACKEND must be cpu or metal, got {other:?}"),
    }
}

fn error_metrics(actual: &[f32], expected: &[f32], label: &str) -> (f32, f64) {
    assert_eq!(actual.len(), expected.len(), "Bark {label} extent differs");
    assert!(!actual.is_empty(), "Bark {label} is empty");
    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    for (&left, &right) in actual.iter().zip(expected) {
        assert!(
            left.is_finite() && right.is_finite(),
            "Bark {label} contains a non-finite value"
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
    text_tokens: &[u32],
    expected_codes: &[u32],
    frames: usize,
    official_packet: &BarkGeneratedCodes,
    metal_generated: &BarkGeneratedCodes,
    metal_official_pcm: &[f32],
    metal_end_to_end_pcm: &[f32],
    expected_pcm: &[f32],
) {
    let cpu_model = BarkModel::open_mapped_with_backend(gguf_path, BackendKind::Cpu)
        .expect("strict CPU mapping for direct Bark Metal-vs-CPU comparison");
    assert_eq!(
        cpu_model.variant().variant_tag(),
        prefix.to_ascii_lowercase()
    );
    let cpu_generated = cpu_model
        .generate_codes_from_tokens(
            text_tokens,
            None,
            &BarkGenerationConfig::greedy(MAX_SEMANTIC_TOKENS),
        )
        .expect("native Bark CPU greedy generation for Metal-vs-CPU comparison");
    assert_eq!(
        cpu_generated.as_frame_major(),
        expected_codes,
        "Bark {prefix} CPU generated packet differs from the official oracle"
    );
    assert_eq!(
        cpu_generated.as_frame_major(),
        metal_generated.as_frame_major(),
        "Bark {prefix} CPU and Metal generated code packets differ"
    );

    let cpu_official_pcm = cpu_model
        .decode_codes(official_packet)
        .expect("native Bark CPU decode of the official packet");
    let (official_max_abs, official_rmse) = error_metrics(
        &cpu_official_pcm,
        metal_official_pcm,
        "CPU-vs-Metal official PCM",
    );
    assert!(
        official_max_abs <= FP32_ATOL,
        "Bark {prefix} CPU-vs-Metal official PCM max_abs={official_max_abs}, rmse={official_rmse}, exceeding {FP32_ATOL}"
    );
    let (official_cpu_max_abs, official_cpu_rmse) =
        error_metrics(&cpu_official_pcm, expected_pcm, "CPU official-packet PCM");
    assert!(
        official_cpu_max_abs <= FP32_ATOL,
        "Bark {prefix} CPU official-packet PCM max_abs={official_cpu_max_abs}, rmse={official_cpu_rmse}, exceeding {FP32_ATOL}"
    );

    let cpu_end_to_end_pcm = cpu_model
        .decode_codes(&cpu_generated)
        .expect("native Bark CPU end-to-end decode");
    let (end_to_end_max_abs, end_to_end_rmse) = error_metrics(
        &cpu_end_to_end_pcm,
        metal_end_to_end_pcm,
        "CPU-vs-Metal end-to-end PCM",
    );
    assert!(
        end_to_end_max_abs <= FP32_ATOL,
        "Bark {prefix} CPU-vs-Metal end-to-end PCM max_abs={end_to_end_max_abs}, rmse={end_to_end_rmse}, exceeding {FP32_ATOL}"
    );
    let (end_to_end_cpu_max_abs, end_to_end_cpu_rmse) =
        error_metrics(&cpu_end_to_end_pcm, expected_pcm, "CPU end-to-end PCM");
    assert!(
        end_to_end_cpu_max_abs <= FP32_ATOL,
        "Bark {prefix} CPU end-to-end PCM max_abs={end_to_end_cpu_max_abs}, rmse={end_to_end_cpu_rmse}, exceeding {FP32_ATOL}"
    );
    eprintln!(
        "BARK_APPLE_PARITY variant={} metal_vs_cpu=PASS",
        prefix.to_ascii_lowercase()
    );
    assert_eq!(official_packet.frames(), frames);
}

fn run_variant(prefix: &str) {
    let Some((gguf_path, reference_dir)) = parity_paths(prefix) else {
        return;
    };
    let text_tokens = read_u32(&reference_dir.join("text_token_ids.u32le"));
    let expected_codes = read_u32(&reference_dir.join("codes.u32le"));
    let expected_pcm = read_f32(&reference_dir.join("decoded_pcm.f32"));
    assert_eq!(expected_codes.len() % CODEBOOKS, 0);
    let frames = expected_codes.len() / CODEBOOKS;
    assert!(frames > 0);
    assert_eq!(expected_pcm.len(), frames * FRAME_HOP);

    let backend = selected_backend();
    let model = BarkModel::open_mapped_with_backend(&gguf_path, backend)
        .expect("strict mapping-owned Bark bind and complete backend preflight");
    let expected_variant = prefix.to_ascii_lowercase();
    assert_eq!(model.variant().variant_tag(), expected_variant.as_str());

    let generated = model
        .generate_codes_from_tokens(
            &text_tokens,
            None,
            &BarkGenerationConfig::greedy(MAX_SEMANTIC_TOKENS),
        )
        .expect("native Bark greedy semantic/coarse/fine generation");
    assert_eq!(
        generated.as_frame_major(),
        expected_codes.as_slice(),
        "Bark {prefix} {backend:?} frame-major code packet differs from the official oracle"
    );

    // Isolate decoder parity from the autoregressive schedule by feeding the
    // official packet through the public validated frame-major boundary.
    let official_packet = BarkGeneratedCodes::from_frame_major(expected_codes.clone(), frames)
        .expect("official Bark frame-major packet");
    let decoded = model
        .decode_codes(&official_packet)
        .expect("native Bark decode of official packet");
    let (decode_max_abs, decode_rmse) =
        error_metrics(&decoded, &expected_pcm, "official-packet decoded PCM");
    assert!(
        decode_max_abs <= FP32_ATOL,
        "Bark {prefix} {backend:?} decoder max_abs {decode_max_abs} exceeds FP32 ceiling {FP32_ATOL}"
    );

    let end_to_end = model
        .decode_codes(&generated)
        .expect("native Bark end-to-end decode");
    let (end_to_end_max_abs, end_to_end_rmse) =
        error_metrics(&end_to_end, &expected_pcm, "end-to-end decoded PCM");
    eprintln!(
        "Bark {prefix} {backend:?}: frames={frames}, codes=exact, decode_max_abs={decode_max_abs:.9e}, decode_rmse={decode_rmse:.9e}, end_to_end_max_abs={end_to_end_max_abs:.9e}, end_to_end_rmse={end_to_end_rmse:.9e}"
    );
    assert!(
        end_to_end_max_abs <= FP32_ATOL,
        "Bark {prefix} {backend:?} end-to-end max_abs {end_to_end_max_abs} exceeds FP32 ceiling {FP32_ATOL}"
    );

    #[cfg(all(feature = "metal", target_os = "macos"))]
    if backend == BackendKind::Metal {
        compare_metal_with_cpu(
            prefix,
            &gguf_path,
            &text_tokens,
            &expected_codes,
            frames,
            &official_packet,
            &generated,
            &decoded,
            &end_to_end,
            &expected_pcm,
        );
    }
}

#[test]
fn real_bark_small_matches_official_transformers() {
    run_variant("SMALL");
}

#[test]
fn real_bark_full_matches_official_transformers() {
    run_variant("FULL");
}
