//! Gated real-public-GGUF parity against the pinned official NeuCodec decoder.
//!
//! The small official fixtures are committed. Set
//! `VOKRA_NEUCODEC_BASE_GGUF` and/or `VOKRA_NEUCODEC_DISTILL_GGUF` to the
//! corresponding audited public artifact. `VOKRA_NEUCODEC_BACKEND=metal`
//! selects Apple Metal; absent or `cpu` selects the scalar CPU path. An unset
//! artifact is a documented skip, never a fabricated pass.

use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_models::neucodec::{NeuCodec, NeuCodecVariant};

const FP32_MAX_ABS: f32 = 2.0e-4;
const FP32_RMSE: f64 = 2.0e-5;
const COSINE_MIN: f64 = 0.999_999;

fn fixture(variant: &str, name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/neucodec")
        .join(variant)
        .join(name)
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read NeuCodec fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read NeuCodec fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn selected_backend() -> BackendKind {
    match std::env::var("VOKRA_NEUCODEC_BACKEND").as_deref() {
        Ok("metal") => BackendKind::Metal,
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok(other) => panic!("VOKRA_NEUCODEC_BACKEND must be cpu or metal, got {other:?}"),
    }
}

fn assert_variant_parity(env_name: &str, fixture_variant: &str, variant: NeuCodecVariant) {
    let Some(path) = std::env::var_os(env_name) else {
        eprintln!("skipping NeuCodec {fixture_variant} real parity: set {env_name}");
        return;
    };
    let backend = selected_backend();
    let file = GgufFile::open(path).expect("open public NeuCodec GGUF");
    let model = NeuCodec::from_gguf(&file)
        .expect("strict public NeuCodec bind")
        .with_backend(backend);
    assert_eq!(model.variant(), variant);
    let codes = read_u32(&fixture(fixture_variant, "codes.u32le"));
    let expected = read_f32(&fixture(fixture_variant, "decoded_pcm.f32"));
    let actual = model.decode_codes(&codes).expect("NeuCodec decode");
    assert_eq!(actual.len(), expected.len());

    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    for (&left, &right) in actual.iter().zip(&expected) {
        let left = f64::from(left);
        let right = f64::from(right);
        let delta = left - right;
        max_abs = max_abs.max(delta.abs() as f32);
        squared_error += delta * delta;
        dot += left * right;
        actual_norm += left * left;
        expected_norm += right * right;
    }
    let rmse = (squared_error / actual.len() as f64).sqrt();
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
    eprintln!(
        "NeuCodec {fixture_variant} {backend:?}: codes={}, samples={}, max_abs={max_abs:.9e}, rmse={rmse:.9e}, cosine={cosine:.12}",
        codes.len(),
        actual.len()
    );
    assert!(
        max_abs <= FP32_MAX_ABS,
        "NeuCodec {fixture_variant} {backend:?} max_abs {max_abs} exceeds {FP32_MAX_ABS}"
    );
    assert!(
        rmse <= FP32_RMSE,
        "NeuCodec {fixture_variant} {backend:?} RMSE {rmse} exceeds {FP32_RMSE}"
    );
    assert!(
        cosine >= COSINE_MIN,
        "NeuCodec {fixture_variant} {backend:?} cosine {cosine} is below {COSINE_MIN}"
    );
}

#[test]
fn real_neucodec_base_decode_matches_official() {
    assert_variant_parity("VOKRA_NEUCODEC_BASE_GGUF", "base", NeuCodecVariant::Base);
}

#[test]
fn real_neucodec_distill_decode_matches_official() {
    assert_variant_parity(
        "VOKRA_NEUCODEC_DISTILL_GGUF",
        "distill",
        NeuCodecVariant::Distill,
    );
}
