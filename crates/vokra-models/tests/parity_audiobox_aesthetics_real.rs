//! Real-checkpoint Audiobox Aesthetics parity.
//!
//! Generate the independent fixture with
//! `tools/parity/audiobox_aesthetics_dump_reference.py`, then set
//! `VOKRA_AUDIOBOX_GGUF` and `VOKRA_AUDIOBOX_REFERENCE_DIR`. The optional
//! `VOKRA_AUDIOBOX_METAL_PARITY=1` second pass reuses the same loaded weights
//! and compares Apple Metal directly to the CPU oracle.

use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_models::audiobox_aesthetics::{AudioboxAesthetics, SAMPLE_RATE};

const FP32_ATOL: f32 = 1.0e-2;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read Audiobox fixture {}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "truncated fixture {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn max_abs(actual: &[f32], expected: &[f32]) -> (usize, f32) {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty Audiobox scores")
}

#[test]
fn reference_contract_is_pinned_without_fabricated_values() {
    let readme = include_str!("../../../tests/fixtures/audiobox_aesthetics/README.md");
    for required in [
        "2618e9d451b456e9328b39495b5e6234678aa550",
        "9b1dd8e5df9af7216e836a98974fe3b82c56ded6",
        "AesMultiOutput",
        "0.01",
    ] {
        assert!(readme.contains(required), "Audiobox pin missing {required}");
    }
    assert!(
        !Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/audiobox_aesthetics/final_scores.f32le")
            .exists(),
        "do not land a numerical fixture without its VAST evidence and manifest review"
    );
}

#[test]
fn official_scores_and_optional_metal_match() {
    let (Some(gguf), Some(reference_dir)) = (
        std::env::var_os("VOKRA_AUDIOBOX_GGUF"),
        std::env::var_os("VOKRA_AUDIOBOX_REFERENCE_DIR"),
    ) else {
        eprintln!(
            "[parity_audiobox_aesthetics_real] SKIP: set VOKRA_AUDIOBOX_GGUF and VOKRA_AUDIOBOX_REFERENCE_DIR"
        );
        return;
    };
    let reference_dir = PathBuf::from(reference_dir);
    let manifest = std::fs::read_to_string(reference_dir.join("manifest.json"))
        .expect("read Audiobox official manifest");
    for required in [
        "vokra.audiobox-aesthetics.official-parity.v1",
        "2618e9d451b456e9328b39495b5e6234678aa550",
        "9b1dd8e5df9af7216e836a98974fe3b82c56ded6",
        "\"axes\": [",
    ] {
        assert!(manifest.contains(required), "manifest lost pin {required}");
    }
    let pcm = read_f32(&reference_dir.join("pcm.f32le"));
    let expected = read_f32(&reference_dir.join("final_scores.f32le"));
    assert_eq!(expected.len(), 4, "official CE/CU/PC/PQ score count");

    let model = AudioboxAesthetics::from_gguf(&gguf)
        .expect("strict public Audiobox bind")
        .with_backend(BackendKind::Cpu);
    let cpu = model
        .score_pcm(&pcm, SAMPLE_RATE)
        .expect("Audiobox CPU score")
        .as_array();
    let (index, delta) = max_abs(&cpu, &expected);
    eprintln!(
        "[parity_audiobox_aesthetics_real] CPU official max_abs={delta:.9e} axis={} actual={:.9e} reference={:.9e}",
        ["CE", "CU", "PC", "PQ"][index],
        cpu[index],
        expected[index]
    );
    assert!(
        delta <= FP32_ATOL,
        "Audiobox CPU/reference max_abs {delta:.9e} exceeds {FP32_ATOL:.9e}"
    );

    if std::env::var_os("VOKRA_AUDIOBOX_METAL_PARITY").is_some() {
        let model = model.with_backend(BackendKind::Metal);
        let metal = model
            .score_pcm(&pcm, SAMPLE_RATE)
            .expect("Audiobox Metal score")
            .as_array();
        let (index, delta) = max_abs(&metal, &cpu);
        eprintln!(
            "[parity_audiobox_aesthetics_real] Metal/CPU max_abs={delta:.9e} axis={} metal={:.9e} cpu={:.9e}",
            ["CE", "CU", "PC", "PQ"][index],
            metal[index],
            cpu[index]
        );
        assert!(
            delta <= FP32_ATOL,
            "Audiobox Metal/CPU max_abs {delta:.9e} exceeds {FP32_ATOL:.9e}"
        );
    }
}
