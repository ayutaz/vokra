//! Gated real-checkpoint parity against the pinned official FocalCodec package.
//!
//! Generate a fixture with `tools/parity/focalcodec/dump_reference.py`, point
//! `VOKRA_FOCALCODEC_GGUF` at the matching public Vokra GGUF, and set
//! `VOKRA_FOCALCODEC_PARITY_DIR` to the fixture directory.  Missing environment
//! variables are a documented skip; a partially configured run fails loudly.

use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_models::focalcodec::FocalCodec;

const FP32_ATOL: f32 = 0.01;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| {
        panic!(
            "failed to read FocalCodec fixture {}: {error}",
            path.display()
        )
    });
    assert_eq!(
        bytes.len() % 4,
        0,
        "FocalCodec fixture {} is truncated",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| {
        panic!(
            "failed to read FocalCodec fixture {}: {error}",
            path.display()
        )
    });
    assert_eq!(
        bytes.len() % 4,
        0,
        "FocalCodec fixture {} is truncated",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn parity_paths() -> Option<(PathBuf, PathBuf)> {
    let gguf = std::env::var_os("VOKRA_FOCALCODEC_GGUF").map(PathBuf::from);
    let fixture = std::env::var_os("VOKRA_FOCALCODEC_PARITY_DIR").map(PathBuf::from);
    match (gguf, fixture) {
        (None, None) => {
            eprintln!(
                "skipping FocalCodec real parity: set VOKRA_FOCALCODEC_GGUF and VOKRA_FOCALCODEC_PARITY_DIR"
            );
            None
        }
        (Some(gguf), Some(fixture)) => Some((gguf, fixture)),
        _ => panic!(
            "FocalCodec real parity is partially configured; set both VOKRA_FOCALCODEC_GGUF and VOKRA_FOCALCODEC_PARITY_DIR"
        ),
    }
}

#[test]
fn real_focalcodec_encode_decode_matches_official() {
    let Some((gguf_path, fixture_dir)) = parity_paths() else {
        return;
    };
    let file = GgufFile::open(&gguf_path).expect("open FocalCodec GGUF");
    let backend = match std::env::var("VOKRA_FOCALCODEC_BACKEND").as_deref() {
        Ok("metal") => BackendKind::Metal,
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok(other) => panic!("VOKRA_FOCALCODEC_BACKEND must be cpu or metal, got {other:?}"),
    };
    let model = FocalCodec::from_gguf(&file)
        .expect("strict FocalCodec bind")
        .with_backend(backend);

    let pcm = read_f32(&fixture_dir.join("pcm.f32"));
    let expected_tokens = read_u32(&fixture_dir.join("tokens.u32"));
    let actual_tokens = model.encode(&pcm).expect("FocalCodec encode");
    assert_eq!(
        actual_tokens,
        expected_tokens,
        "FocalCodec {:?} {backend:?} token sequence differs from official oracle",
        model.variant()
    );

    let actual_pcm = model
        .decode(&expected_tokens)
        .expect("FocalCodec decode official tokens");
    let expected_pcm = read_f32(&fixture_dir.join("decoded_pcm.f32"));
    assert_eq!(
        actual_pcm.len(),
        expected_pcm.len(),
        "FocalCodec {:?} decoded sample extent differs",
        model.variant()
    );
    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    for (&actual, &expected) in actual_pcm.iter().zip(&expected_pcm) {
        let delta = (actual - expected).abs();
        max_abs = max_abs.max(delta);
        squared_error += f64::from(delta) * f64::from(delta);
    }
    let rmse = (squared_error / actual_pcm.len() as f64).sqrt();
    eprintln!(
        "FocalCodec {:?} {backend:?}: tokens={}, samples={}, max_abs={max_abs:.9e}, rmse={rmse:.9e}",
        model.variant(),
        actual_tokens.len(),
        actual_pcm.len()
    );
    assert!(
        max_abs <= FP32_ATOL,
        "FocalCodec {:?} {backend:?} waveform max_abs {max_abs} exceeds FP32 atol {FP32_ATOL}",
        model.variant()
    );
}
