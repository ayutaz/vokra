//! Gated real-checkpoint parity against `vocos==0.1.0`.
//!
//! Generate inputs with `tools/parity/vocos_dump_reference.py`, convert the
//! matching prepared safetensors checkpoint, then set the environment paths
//! below.  An unset environment is a skip, never a fabricated pass.

use std::path::Path;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_models::vocos::{Vocos, VocosVariant};

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read parity f32 file");
    assert_eq!(bytes.len() % 4, 0, "f32 file must not be truncated");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn real_vocos_feature_decode_matches_official() {
    let (Ok(gguf), Ok(features), Ok(reference)) = (
        std::env::var("VOKRA_VOCOS_GGUF"),
        std::env::var("VOKRA_VOCOS_FEATURES"),
        std::env::var("VOKRA_VOCOS_REFERENCE"),
    ) else {
        eprintln!(
            "skipping Vocos real parity: set VOKRA_VOCOS_GGUF, VOKRA_VOCOS_FEATURES and VOKRA_VOCOS_REFERENCE"
        );
        return;
    };
    let file = GgufFile::open(&gguf).expect("open Vocos GGUF");
    let backend = match std::env::var("VOKRA_VOCOS_BACKEND").as_deref() {
        Ok("metal") => BackendKind::Metal,
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok(other) => panic!("VOKRA_VOCOS_BACKEND must be cpu or metal, got {other:?}"),
    };
    let model = Vocos::from_gguf(&file)
        .expect("strict Vocos bind")
        .with_backend(backend);
    let input = read_f32(Path::new(&features));
    assert_eq!(input.len() % model.config().n_input, 0);
    let frames = input.len() / model.config().n_input;
    let actual = match model.variant() {
        VocosVariant::Mel24khz => model.decode(&input, frames).expect("mel decode"),
        VocosVariant::Encodec24khz => {
            let bandwidth_id = std::env::var("VOKRA_VOCOS_BANDWIDTH_ID")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(2);
            model
                .decode_with_bandwidth(&input, frames, bandwidth_id)
                .expect("encodec decode")
        }
    };
    let expected = read_f32(Path::new(&reference));
    assert_eq!(actual.len(), expected.len());
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "Vocos {:?} {backend:?}: frames={frames}, samples={}, max_abs={max_abs:.9e}",
        model.variant(),
        actual.len()
    );
    assert!(max_abs <= 1.0e-5, "Vocos max_abs {max_abs} exceeds 1e-5");
}
