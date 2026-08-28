//! Independent public-artifact parity for Descript DAC 16/24/44.1 kHz.
//!
//! The committed oracle was produced by `descript-audio-codec==1.0.0` from
//! the official release checkpoints, through the public
//! `ResidualVectorQuantize.from_codes` and `DAC.decode` APIs.  Vokra code does
//! not participate in reference generation.  The public GGUFs are
//! intentionally env-gated because their 298–307 MB weight artifacts are not
//! committed.

use vokra_models::dac::{Dac, DacVariant};

const CODES_16: &[u8] = include_bytes!("fixtures/dac_16khz/codes.u32");
const FEATURES_16: &[u8] = include_bytes!("fixtures/dac_16khz/decoded_features.f32");
const PCM_16: &[u8] = include_bytes!("fixtures/dac_16khz/decoded_pcm.f32");
const MANIFEST_16: &str = include_str!("fixtures/dac_16khz/manifest.txt");
const CODES_24: &[u8] = include_bytes!("fixtures/dac_24khz/codes.u32");
const FEATURES_24: &[u8] = include_bytes!("fixtures/dac_24khz/decoded_features.f32");
const PCM_24: &[u8] = include_bytes!("fixtures/dac_24khz/decoded_pcm.f32");
const MANIFEST_24: &str = include_str!("fixtures/dac_24khz/manifest.txt");
const CODES_44: &[u8] = include_bytes!("fixtures/dac_44khz/codes.u32");
const FEATURES_44: &[u8] = include_bytes!("fixtures/dac_44khz/decoded_features.f32");
const PCM_44: &[u8] = include_bytes!("fixtures/dac_44khz/decoded_pcm.f32");
const MANIFEST_44: &str = include_str!("fixtures/dac_44khz/manifest.txt");

// The initial pre-measurement envelope was max=2e-4 / relative-L1=2e-3.
// VAST 48577185 then measured max=1.0133e-6 / relative-L1=1.0840e-6 / cosine
// 0.999999940 end-to-end. The independent 16/24 kHz M1 measurements were
// smaller still (max 8.35e-7, relative-L1 1.30e-6). The shared committed gate
// is therefore below the project FP32 0.01 without treating a sibling result
// as the sibling's oracle.
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

struct Fixture {
    label: &'static str,
    env: &'static str,
    variant: DacVariant,
    sample_rate: u32,
    n_codebooks: usize,
    codes: &'static [u8],
    features: &'static [u8],
    pcm: &'static [u8],
    manifest: &'static str,
}

const FIXTURE_16: Fixture = Fixture {
    label: "16 kHz",
    env: "VOKRA_DAC_16KHZ_GGUF",
    variant: DacVariant::Khz16,
    sample_rate: 16_000,
    n_codebooks: 12,
    codes: CODES_16,
    features: FEATURES_16,
    pcm: PCM_16,
    manifest: MANIFEST_16,
};

const FIXTURE_24: Fixture = Fixture {
    label: "24 kHz",
    env: "VOKRA_DAC_24KHZ_GGUF",
    variant: DacVariant::Khz24,
    sample_rate: 24_000,
    n_codebooks: 32,
    codes: CODES_24,
    features: FEATURES_24,
    pcm: PCM_24,
    manifest: MANIFEST_24,
};

const FIXTURE_44: Fixture = Fixture {
    label: "44.1 kHz",
    env: "VOKRA_DAC_44KHZ_GGUF",
    variant: DacVariant::Khz44,
    sample_rate: 44_100,
    n_codebooks: 9,
    codes: CODES_44,
    features: FEATURES_44,
    pcm: PCM_44,
    manifest: MANIFEST_44,
};

fn assert_fixture_identity(
    fixture: &Fixture,
    checkpoint_pin: &str,
    checkpoint_sha256: &str,
    pcm_samples: usize,
    pcm_sha256: &str,
) {
    assert_eq!(u32s(fixture.codes).len(), fixture.n_codebooks);
    assert_eq!(f32s(fixture.features).len(), 1_024);
    assert_eq!(f32s(fixture.pcm).len(), pcm_samples);
    assert!(fixture.manifest.contains("descript-audio-codec==1.0.0"));
    assert!(fixture.manifest.contains(checkpoint_pin));
    assert!(fixture.manifest.contains(checkpoint_sha256));
    assert!(
        fixture
            .manifest
            .contains(&format!("sha256 decoded_pcm.f32 {pcm_sha256}"))
    );
}

#[test]
fn committed_references_have_pinned_upstream_identity() {
    assert_fixture_identity(
        &FIXTURE_16,
        "official release tag 0.0.5 weights_16khz.pth",
        "95ab7176b67137d4d4c6c54b8d6ef3cea797faec228cb03ad084badcad570b4d",
        312,
        "8003d2cb8c8e7b0b698f76ea2b5d8f3ccf0a86f790ea47ba5c1dfefb19795bf4",
    );
    assert_fixture_identity(
        &FIXTURE_24,
        "official release tag 0.0.4 weights_24khz.pth",
        "44bad592fc393e03eb0be7a5120b7d487fe9612fa41269dc03fca3d4b87e20ad",
        312,
        "ef1e5d2af37584c067461a0c3995b51ea64c08a6fb48ca4727f3652b89a7322f",
    );
    assert_fixture_identity(
        &FIXTURE_44,
        "official release tag 0.0.1 weights.pth",
        "a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa",
        512,
        "4967cc32a8dc5221d0d9a362c257a6877c2d1efcf9de5e87e48bcf7c0ae25d9e",
    );
}

fn run_public_artifact(fixture: &Fixture) {
    let Some(path) = std::env::var_os(fixture.env) else {
        eprintln!(
            "[parity_dac_decoder_real] SKIP {}: set {} to the public canonical GGUF",
            fixture.label, fixture.env
        );
        return;
    };
    let model = Dac::from_path(path)
        .unwrap_or_else(|error| panic!("strict public DAC {} bind: {error}", fixture.label));
    assert_eq!(model.variant(), fixture.variant);
    assert_eq!(model.sample_rate(), fixture.sample_rate);
    assert_eq!(model.n_codebooks(), fixture.n_codebooks);

    let codes = u32s(fixture.codes);
    let features = f32s(fixture.features);
    let expected = f32s(fixture.pcm);
    let cpu_decoder = model
        .decode_features(&features)
        .expect("CPU official-feature decoder");
    assert_cpu(&measure(
        &format!("{} CPU decoder vs official DAC.decode", fixture.label),
        &cpu_decoder,
        &expected,
    ));
    let cpu_end_to_end = model.decode_codes(&codes).expect("CPU RVQ + decoder");
    assert_cpu(&measure(
        &format!("{} CPU RVQ + decoder vs official", fixture.label),
        &cpu_end_to_end,
        &expected,
    ));

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let metal = model.with_backend(vokra_core::BackendKind::Metal);
        let metal_end_to_end = metal.decode_codes(&codes).expect("Metal RVQ + decoder");
        let metrics = measure(
            &format!("{} Metal RVQ + decoder vs official", fixture.label),
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

#[test]
fn public_dac_16khz_matches_official_decoder() {
    run_public_artifact(&FIXTURE_16);
}

#[test]
fn public_dac_24khz_matches_official_decoder() {
    run_public_artifact(&FIXTURE_24);
}

#[test]
fn public_dac_44khz_matches_official_decoder() {
    run_public_artifact(&FIXTURE_44);
}
