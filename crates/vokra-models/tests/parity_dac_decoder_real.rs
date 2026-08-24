//! Independent public-artifact parity for Descript DAC 44.1 kHz.
//!
//! The committed oracle was produced by `descript-audio-codec==1.0.0` from
//! the official release-tag 0.0.1 `weights.pth`, through the public
//! `ResidualVectorQuantize.from_codes` and `DAC.decode` APIs.  Vokra code does
//! not participate in reference generation.  The public GGUF is intentionally
//! env-gated because the 307 MB weight artifact is not committed.

use vokra_models::dac::{Dac, DacVariant};

const CODES: &[u8] = include_bytes!("fixtures/dac_44khz/codes.u32");
const FEATURES: &[u8] = include_bytes!("fixtures/dac_44khz/decoded_features.f32");
const PCM: &[u8] = include_bytes!("fixtures/dac_44khz/decoded_pcm.f32");
const MANIFEST: &str = include_str!("fixtures/dac_44khz/manifest.txt");

// The initial pre-measurement envelope was max=2e-4 / relative-L1=2e-3.
// VAST 48577185 then measured max=1.0133e-6 / relative-L1=1.0840e-6 / cosine
// 0.999999940 end-to-end, so the committed gate was tightened (never widened)
// to less than 2.5x the observed error and far below the project FP32 0.01.
const CPU_MAX_ABS_BOUND: f32 = 2.0e-6;
const CPU_RELATIVE_L1_BOUND: f32 = 2.5e-6;
const CPU_COSINE_BOUND: f32 = 0.999_999_5;

// The initial pre-measurement Metal envelope was max=5e-4 / relative-L1=5e-3.
// The M1 Mac run measured max=1.0133e-6 / relative-L1=1.0250e-6 / cosine 1.0,
// so these bounds were tightened to the same narrow envelope as CPU.
#[cfg(all(feature = "metal", target_os = "macos"))]
const METAL_MAX_ABS_BOUND: f32 = 2.0e-6;
#[cfg(all(feature = "metal", target_os = "macos"))]
const METAL_RELATIVE_L1_BOUND: f32 = 2.5e-6;
#[cfg(all(feature = "metal", target_os = "macos"))]
const METAL_COSINE_BOUND: f32 = 0.999_999_5;

fn f32s(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0, "truncated f32 fixture");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn u32s(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0, "truncated u32 fixture");
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[derive(Debug)]
struct Metrics {
    max_abs: f32,
    max_abs_index: usize,
    mean_abs: f32,
    relative_l1: f32,
    cosine: f32,
}

fn measure(label: &str, actual: &[f32], expected: &[f32]) -> Metrics {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    assert!(!actual.is_empty(), "{label} must be non-empty");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{label} finite"
    );
    let (max_abs_index, max_abs) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap();
    let sum_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>();
    let expected_l1 = expected.iter().map(|value| value.abs()).sum::<f32>();
    let dot = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| actual * expected)
        .sum::<f32>();
    let actual_norm = actual.iter().map(|value| value * value).sum::<f32>().sqrt();
    let expected_norm = expected
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let metrics = Metrics {
        max_abs,
        max_abs_index,
        mean_abs: sum_abs / actual.len() as f32,
        relative_l1: sum_abs / expected_l1.max(1.0e-20),
        cosine: dot / (actual_norm * expected_norm).max(1.0e-20),
    };
    eprintln!(
        "DAC {label}: max_abs={:.9e} at {} (actual={:.9e}, reference={:.9e}), mean_abs={:.9e}, relative_l1={:.9e}, cosine={:.9e}",
        metrics.max_abs,
        metrics.max_abs_index,
        actual[metrics.max_abs_index],
        expected[metrics.max_abs_index],
        metrics.mean_abs,
        metrics.relative_l1,
        metrics.cosine,
    );
    metrics
}

fn assert_cpu(metrics: &Metrics) {
    assert!(metrics.max_abs <= CPU_MAX_ABS_BOUND, "{metrics:?}");
    assert!(metrics.relative_l1 <= CPU_RELATIVE_L1_BOUND, "{metrics:?}");
    assert!(metrics.cosine >= CPU_COSINE_BOUND, "{metrics:?}");
}

#[test]
fn committed_reference_has_pinned_upstream_identity() {
    assert_eq!(u32s(CODES).len(), 9);
    assert_eq!(f32s(FEATURES).len(), 1_024);
    assert_eq!(f32s(PCM).len(), 512);
    assert!(MANIFEST.contains("descript-audio-codec==1.0.0"));
    assert!(MANIFEST.contains("official release tag 0.0.1 weights.pth"));
    assert!(MANIFEST.contains("a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa"));
    assert!(MANIFEST.contains(
        "sha256 decoded_pcm.f32 4967cc32a8dc5221d0d9a362c257a6877c2d1efcf9de5e87e48bcf7c0ae25d9e"
    ));
}

#[test]
fn public_dac_44khz_matches_official_decoder() {
    let Some(path) = std::env::var_os("VOKRA_DAC_44KHZ_GGUF") else {
        eprintln!(
            "[parity_dac_decoder_real] SKIP: set VOKRA_DAC_44KHZ_GGUF to the public canonical GGUF"
        );
        return;
    };
    let model = Dac::from_path(path).expect("strict public DAC 44.1 kHz bind");
    assert_eq!(model.variant(), DacVariant::Khz44);
    assert_eq!(model.sample_rate(), 44_100);
    assert_eq!(model.n_codebooks(), 9);

    let codes = u32s(CODES);
    let features = f32s(FEATURES);
    let expected = f32s(PCM);
    let cpu_decoder = model
        .decode_features(&features)
        .expect("CPU official-feature decoder");
    assert_cpu(&measure(
        "CPU decoder vs official DAC.decode",
        &cpu_decoder,
        &expected,
    ));
    let cpu_end_to_end = model.decode_codes(&codes).expect("CPU RVQ + decoder");
    assert_cpu(&measure(
        "CPU RVQ + decoder vs official",
        &cpu_end_to_end,
        &expected,
    ));

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let metal = model.with_backend(vokra_core::BackendKind::Metal);
        let metal_end_to_end = metal.decode_codes(&codes).expect("Metal RVQ + decoder");
        let metrics = measure(
            "Metal RVQ + decoder vs official",
            &metal_end_to_end,
            &expected,
        );
        assert!(metrics.max_abs <= METAL_MAX_ABS_BOUND, "{metrics:?}");
        assert!(
            metrics.relative_l1 <= METAL_RELATIVE_L1_BOUND,
            "{metrics:?}"
        );
        assert!(metrics.cosine >= METAL_COSINE_BOUND, "{metrics:?}");
    }
}
