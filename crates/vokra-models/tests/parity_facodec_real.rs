//! Gated real-public-GGUF parity against pinned official Amphion FACodec V2.
//!
//! Generate the reference on VAST through
//! `scripts/publish/vast-ai/run-facodec-parity.sh`, then set both
//! `VOKRA_FACODEC_GGUF` and `VOKRA_FACODEC_PARITY_DIR`. An unset pair is a
//! documented skip; partial configuration fails loudly. The same comparison
//! can run on a remote Apple Silicon host with `VOKRA_FACODEC_BACKEND=metal`.

use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_models::facodec::{FacodecEncoded, FacodecV2};

const FP32_ATOL: f32 = 0.01;
const NUM_CODEBOOKS: usize = 6;
const SPEAKER_DIM: usize = 256;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read FACodec fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read FACodec fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn parity_paths() -> Option<(PathBuf, PathBuf)> {
    let gguf = std::env::var_os("VOKRA_FACODEC_GGUF").map(PathBuf::from);
    let reference = std::env::var_os("VOKRA_FACODEC_PARITY_DIR").map(PathBuf::from);
    match (gguf, reference) {
        (None, None) => {
            eprintln!(
                "skipping FACodec V2 real parity: set VOKRA_FACODEC_GGUF and VOKRA_FACODEC_PARITY_DIR"
            );
            None
        }
        (Some(gguf), Some(reference)) => Some((gguf, reference)),
        _ => panic!(
            "FACodec V2 real parity is partially configured; set both VOKRA_FACODEC_GGUF and VOKRA_FACODEC_PARITY_DIR"
        ),
    }
}

fn selected_backend() -> BackendKind {
    match std::env::var("VOKRA_FACODEC_BACKEND").as_deref() {
        Ok("metal") => BackendKind::Metal,
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok(other) => panic!("VOKRA_FACODEC_BACKEND must be cpu or metal, got {other:?}"),
    }
}

fn error_metrics(actual: &[f32], expected: &[f32], label: &str) -> (f32, f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "FACodec {label} extent differs"
    );
    assert!(!actual.is_empty(), "FACodec {label} is empty");
    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    for (&left, &right) in actual.iter().zip(expected) {
        assert!(
            left.is_finite() && right.is_finite(),
            "FACodec {label} contains a non-finite value"
        );
        let delta = f64::from(left) - f64::from(right);
        max_abs = max_abs.max(delta.abs() as f32);
        squared_error += delta * delta;
    }
    let rmse = (squared_error / actual.len() as f64).sqrt();
    (max_abs, rmse)
}

#[test]
fn real_facodec_v2_encode_decode_matches_official() {
    let Some((gguf_path, reference_dir)) = parity_paths() else {
        return;
    };
    let pcm = read_f32(&reference_dir.join("pcm.f32"));
    let expected_codes = read_u32(&reference_dir.join("codes.u32le"));
    let expected_speaker = read_f32(&reference_dir.join("speaker_embedding.f32"));
    let expected_pcm = read_f32(&reference_dir.join("decoded_pcm.f32"));
    assert_eq!(expected_codes.len() % NUM_CODEBOOKS, 0);
    assert_eq!(expected_speaker.len(), SPEAKER_DIM);
    let frames = expected_codes.len() / NUM_CODEBOOKS;
    assert_eq!(expected_pcm.len(), frames * 200);

    let backend = selected_backend();
    let file = GgufFile::open(gguf_path).expect("open audited public FACodec V2 GGUF");
    let model = FacodecV2::from_gguf_with_backend(&file, backend)
        .expect("strict FACodec V2 bind and complete backend preflight");

    let actual_encoded = model.encode(&pcm).expect("native FACodec V2 encode");
    assert_eq!(actual_encoded.frames, frames);
    assert_eq!(
        actual_encoded.codes, expected_codes,
        "FACodec V2 {backend:?} six-stream code matrix differs from the official oracle"
    );
    let (speaker_max_abs, speaker_rmse) = error_metrics(
        &actual_encoded.speaker_embedding,
        &expected_speaker,
        "speaker embedding",
    );
    assert!(
        speaker_max_abs <= FP32_ATOL,
        "FACodec V2 {backend:?} speaker max_abs {speaker_max_abs} exceeds FP32 ceiling {FP32_ATOL}"
    );

    // Isolate decoder parity with the official discrete codes and official
    // speaker embedding, then also verify the complete native reconstruction.
    let official_packet = FacodecEncoded {
        frames,
        codes: expected_codes,
        speaker_embedding: expected_speaker,
        input_samples: pcm.len(),
    };
    let decoded_official_packet = model
        .decode(&official_packet)
        .expect("native FACodec V2 decode of official packet");
    let (decode_max_abs, decode_rmse) = error_metrics(
        &decoded_official_packet,
        &expected_pcm,
        "official-packet decoded PCM",
    );
    assert!(
        decode_max_abs <= FP32_ATOL,
        "FACodec V2 {backend:?} decoder max_abs {decode_max_abs} exceeds FP32 ceiling {FP32_ATOL}"
    );

    let reconstructed = model
        .decode(&actual_encoded)
        .expect("native FACodec V2 end-to-end decode");
    let (end_to_end_max_abs, end_to_end_rmse) =
        error_metrics(&reconstructed, &expected_pcm, "end-to-end decoded PCM");
    eprintln!(
        "FACodec V2 {backend:?}: frames={frames}, codes=exact, speaker_max_abs={speaker_max_abs:.9e}, speaker_rmse={speaker_rmse:.9e}, decode_max_abs={decode_max_abs:.9e}, decode_rmse={decode_rmse:.9e}, end_to_end_max_abs={end_to_end_max_abs:.9e}, end_to_end_rmse={end_to_end_rmse:.9e}"
    );
    assert!(
        end_to_end_max_abs <= FP32_ATOL,
        "FACodec V2 {backend:?} end-to-end max_abs {end_to_end_max_abs} exceeds FP32 ceiling {FP32_ATOL}"
    );
}
