//! Gated real-public-GGUF parity against the pinned official FunCodec decoder.
//!
//! Generate the reference on VAST with
//! `tools/parity/funcodec/dump_reference.py`, then set both
//! `VOKRA_FUNCODEC_GGUF` and `VOKRA_FUNCODEC_REFERENCE_DIR`. An unset input is
//! a documented skip, never a fabricated pass. `VOKRA_FUNCODEC_BACKEND=metal`
//! selects Apple Metal; absent or `cpu` selects CPU.

use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_models::funcodec::{FunCodec, NUM_CODEBOOKS};

// Repository-wide FP32 waveform parity ceiling. Tightening requires a recorded
// official-reference measurement; widening requires a separate justification.
const FP32_ATOL: f32 = 0.01;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read FunCodec fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read FunCodec fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn selected_backend() -> BackendKind {
    match std::env::var("VOKRA_FUNCODEC_BACKEND").as_deref() {
        Ok("metal") => BackendKind::Metal,
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok(other) => panic!("VOKRA_FUNCODEC_BACKEND must be cpu or metal, got {other:?}"),
    }
}

#[test]
fn real_funcodec_decode_matches_official() {
    let Some(gguf_path) = std::env::var_os("VOKRA_FUNCODEC_GGUF") else {
        eprintln!("skipping FunCodec real parity: set VOKRA_FUNCODEC_GGUF");
        return;
    };
    let Some(reference_dir) = std::env::var_os("VOKRA_FUNCODEC_REFERENCE_DIR") else {
        eprintln!("skipping FunCodec real parity: set VOKRA_FUNCODEC_REFERENCE_DIR");
        return;
    };
    let reference_dir = PathBuf::from(reference_dir);
    let codes = read_u32(&reference_dir.join("codes.u32le"));
    let expected = read_f32(&reference_dir.join("decoded_pcm.f32"));
    assert!(!codes.is_empty());
    assert!(codes.len().is_multiple_of(NUM_CODEBOOKS));
    let frames = codes.len() / NUM_CODEBOOKS;

    let backend = selected_backend();
    let file = GgufFile::open(gguf_path).expect("open audited public FunCodec GGUF");
    let model = FunCodec::from_gguf_with_backend(&file, backend)
        .expect("strict public FunCodec bind and backend preflight");
    let actual = model
        .decode_frame_major(&codes, frames, NUM_CODEBOOKS)
        .expect("native FunCodec decode");
    assert_eq!(actual.len(), expected.len());

    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    for (&left, &right) in actual.iter().zip(&expected) {
        assert!(left.is_finite() && right.is_finite());
        let delta = f64::from(left) - f64::from(right);
        max_abs = max_abs.max(delta.abs() as f32);
        squared_error += delta * delta;
    }
    let rmse = (squared_error / actual.len() as f64).sqrt();
    eprintln!(
        "FunCodec {backend:?}: frames={frames}, quantizers={NUM_CODEBOOKS}, samples={}, max_abs={max_abs:.9e}, rmse={rmse:.9e}",
        actual.len()
    );
    assert!(
        max_abs <= FP32_ATOL,
        "FunCodec {backend:?} max_abs {max_abs} exceeds repository FP32 ceiling {FP32_ATOL}"
    );
}
