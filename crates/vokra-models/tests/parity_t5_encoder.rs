//! Real T5-base CPU/Metal parity against the official Transformers forward.
//!
//! `tools/parity/t5_encoder_dump_reference.py` creates the independent oracle
//! from `transformers.T5EncoderModel.forward`. The test is deliberately
//! ignored: the VAST worker must supply both the exact composite GGUF and the
//! immutable reference directory. Missing inputs are a hard failure once the
//! ignored test is explicitly selected; no absent fixture is reported green.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_models::t5_encoder::{T5_BASE_CONFIG, T5Encoder};

const GGUF_ENV: &str = "VOKRA_T5_BASE_GGUF";
const REFERENCE_DIR_ENV: &str = "VOKRA_T5_BASE_REFERENCE_DIR";
const PREFIX_ENV: &str = "VOKRA_T5_BASE_PREFIX";
const OFFICIAL_FP32_ATOL: f32 = 0.01;
const OFFICIAL_SOURCE_REPO: &str = "google-t5/t5-base";
const OFFICIAL_SOURCE_REVISION: &str = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1";

fn verify_manifest(reference_dir: &Path) {
    let path = reference_dir.join("manifest.json");
    let manifest = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read official T5 manifest {path:?}: {error}"));
    for required in [
        "\"format\": \"vokra-t5-encoder-reference-v1\"".to_owned(),
        "\"oracle\": \"transformers.T5EncoderModel.forward\"".to_owned(),
        format!("\"source_repo\": \"{OFFICIAL_SOURCE_REPO}\""),
        format!("\"source_revision\": \"{OFFICIAL_SOURCE_REVISION}\""),
    ] {
        assert!(
            manifest.contains(&required),
            "official T5 manifest {path:?} is missing immutable provenance field {required:?}"
        );
    }
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(bytes.len() % 4, 0, "unaligned u32 fixture {path:?}");
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(bytes.len() % 4, 0, "unaligned f32 fixture {path:?}");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len(), "parity vector length");
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}

fn inputs() -> (PathBuf, PathBuf, String) {
    let gguf = env::var_os(GGUF_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{GGUF_ENV} must point at the exact composite T5-base GGUF"));
    let reference = env::var_os(REFERENCE_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!("{REFERENCE_DIR_ENV} must point at the independent official dump")
        });
    let prefix = env::var(PREFIX_ENV).unwrap_or_else(|_| "text_encoder".to_owned());
    (gguf, reference, prefix)
}

#[test]
#[ignore = "requires an immutable composite GGUF and independent official T5-base VAST dump"]
fn parity_t5_base_official_hidden_states_cpu_and_metal() {
    let (gguf_path, reference_dir, prefix) = inputs();
    verify_manifest(&reference_dir);
    let token_ids = read_u32(&reference_dir.join("input_ids.u32"));
    let mask_bytes =
        fs::read(reference_dir.join("attention_mask.u8")).expect("read official T5 attention mask");
    let attention_mask: Vec<bool> = mask_bytes
        .iter()
        .map(|&value| match value {
            0 => false,
            1 => true,
            other => panic!("official attention mask contains non-binary byte {other}"),
        })
        .collect();
    let expected = read_f32(&reference_dir.join("last_hidden_state.f32"));
    assert_eq!(token_ids, [71, 1234, 5, 0, 42, 9, 1]);
    assert_eq!(attention_mask.len(), token_ids.len());
    assert_eq!(expected.len(), token_ids.len() * T5_BASE_CONFIG.d_model);
    assert!(expected.iter().all(|value| value.is_finite()));

    let file = GgufFile::open(&gguf_path)
        .unwrap_or_else(|error| panic!("open opted-in T5 GGUF {gguf_path:?}: {error}"));
    let cpu = T5Encoder::t5_base_from_gguf(&file, &prefix)
        .expect("strictly bind canonical T5-base tensors")
        .encode_tokens(&token_ids, Some(&attention_mask))
        .expect("native T5-base CPU forward");
    let cpu_max_abs = max_abs(&cpu, &expected);
    assert!(
        cpu_max_abs <= OFFICIAL_FP32_ATOL,
        "T5-base CPU vs official max_abs={cpu_max_abs:.9e} exceeds {OFFICIAL_FP32_ATOL:.9e}"
    );

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    {
        let metal = T5Encoder::t5_base_from_gguf(&file, &prefix)
            .expect("rebind canonical T5-base tensors for Metal")
            .with_backend(BackendKind::Metal)
            .encode_tokens(&token_ids, Some(&attention_mask))
            .expect("native T5-base Metal forward");
        let metal_max_abs = max_abs(&metal, &expected);
        assert!(
            metal_max_abs <= OFFICIAL_FP32_ATOL,
            "T5-base Metal vs official max_abs={metal_max_abs:.9e} exceeds {OFFICIAL_FP32_ATOL:.9e}"
        );
        eprintln!(
            "T5_BASE_OFFICIAL_PARITY backend=metal max_abs={metal_max_abs:.9e} bound={OFFICIAL_FP32_ATOL:.9e} verdict=PASS"
        );
    }

    // Keep the import live on non-Metal builds; the real Metal branch above
    // is compiled only for an explicitly Metal-enabled Apple target.
    let _ = BackendKind::Cpu;
    eprintln!(
        "T5_BASE_OFFICIAL_PARITY backend=cpu max_abs={cpu_max_abs:.9e} bound={OFFICIAL_FP32_ATOL:.9e} verdict=PASS"
    );
}
