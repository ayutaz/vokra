//! Gated real-public-GGUF parity against the pinned official WavTokenizer.
//!
//! The small official fixture is committed. Set `VOKRA_WAVTOKENIZER_GGUF` to
//! either public Vokra WavTokenizer GGUF (both currently have the audited same
//! SHA-256). `VOKRA_WAVTOKENIZER_BACKEND=metal` selects Apple Metal; absent or
//! `cpu` selects the scalar/CPU kernel path. An unset GGUF is a documented
//! skip, never a fabricated pass.

use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_models::wavtokenizer::WavTokenizer;

const FP32_ATOL: f32 = 2.0e-5;
const COSINE_MIN: f64 = 0.999_999_999;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/wavtokenizer")
        .join(name)
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read WavTokenizer fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read WavTokenizer fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn real_wavtokenizer_decode_matches_official() {
    let Some(path) = std::env::var_os("VOKRA_WAVTOKENIZER_GGUF") else {
        eprintln!(
            "skipping WavTokenizer real parity: set VOKRA_WAVTOKENIZER_GGUF to either audited public GGUF"
        );
        return;
    };
    let backend = match std::env::var("VOKRA_WAVTOKENIZER_BACKEND").as_deref() {
        Ok("metal") => BackendKind::Metal,
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok(other) => panic!("VOKRA_WAVTOKENIZER_BACKEND must be cpu or metal, got {other:?}"),
    };
    let file = GgufFile::open(path).expect("open public WavTokenizer GGUF");
    let model = WavTokenizer::from_gguf(&file)
        .expect("strict public WavTokenizer bind")
        .with_backend(backend);
    let codes = read_u32(&fixture("codes.u32le"));
    let expected = read_f32(&fixture("decoded_pcm.f32"));
    let actual = model.decode_codes(&codes).expect("WavTokenizer decode");
    assert_eq!(actual.len(), expected.len());

    let mut max_abs = 0.0f32;
    let mut mean_abs = 0.0f64;
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    for (&left, &right) in actual.iter().zip(&expected) {
        let left = f64::from(left);
        let right = f64::from(right);
        let delta = (left - right).abs();
        max_abs = max_abs.max(delta as f32);
        mean_abs += delta;
        dot += left * right;
        actual_norm += left * left;
        expected_norm += right * right;
    }
    mean_abs /= actual.len() as f64;
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
    eprintln!(
        "WavTokenizer {backend:?}: codes={}, samples={}, max_abs={max_abs:.9e}, mean_abs={mean_abs:.9e}, cosine={cosine:.12}",
        codes.len(),
        actual.len()
    );
    assert!(
        max_abs <= FP32_ATOL,
        "WavTokenizer {backend:?} max_abs {max_abs} exceeds {FP32_ATOL}"
    );
    assert!(
        cosine >= COSINE_MIN,
        "WavTokenizer {backend:?} cosine {cosine} is below {COSINE_MIN}"
    );
}
