//! Independent real-weight DNS48 parity against the official Demucs forward.
//!
//! The reference directory must be produced by
//! `tools/parity/facebook_denoiser_dump_reference.py`. Bounds are deliberately
//! required from the environment until the first VAST measurement records
//! them; this test does not invent a tolerance before observing the oracle.

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_models::facebook_denoiser::FbDenoiser;

fn f32_file(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{} f32 alignment", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn required_bound(name: &str) -> f32 {
    let raw = std::env::var(name).unwrap_or_else(|_| {
        panic!(
            "{name} is required when VOKRA_FACEBOOK_DENOISER_GGUF is set; calibrate it from the recorded independent VAST comparison"
        )
    });
    let value = raw
        .parse::<f32>()
        .unwrap_or_else(|error| panic!("{name}={raw:?}: {error}"));
    assert!(
        value.is_finite() && value >= 0.0,
        "{name} must be finite and non-negative"
    );
    value
}

fn compare(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    max_abs_bound: f32,
    relative_l1_bound: f32,
) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{label} finite"
    );
    let (index, max_abs) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty DNS48 output");
    let absolute_l1 = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>();
    let reference_l1 = expected
        .iter()
        .map(|value| value.abs())
        .sum::<f32>()
        .max(1.0e-20);
    let relative_l1 = absolute_l1 / reference_l1;
    eprintln!(
        "Facebook Denoiser {label}: max_abs={max_abs:.9e} at {index} (actual={:.9e}, reference={:.9e}), relative_l1={relative_l1:.9e}",
        actual[index], expected[index]
    );
    assert!(
        max_abs <= max_abs_bound,
        "{label} max_abs={max_abs:.9e}, bound={max_abs_bound:.9e}"
    );
    assert!(
        relative_l1 <= relative_l1_bound,
        "{label} relative_l1={relative_l1:.9e}, bound={relative_l1_bound:.9e}"
    );
}

fn fixture() -> Option<(GgufFile, Vec<f32>, Vec<f32>)> {
    let Some(gguf_path) = std::env::var_os("VOKRA_FACEBOOK_DENOISER_GGUF") else {
        eprintln!(
            "[parity_facebook_denoiser_real] SKIP: set VOKRA_FACEBOOK_DENOISER_GGUF and VOKRA_FACEBOOK_DENOISER_REFERENCE_DIR after generating the official VAST reference"
        );
        return None;
    };
    let reference_dir = std::env::var_os("VOKRA_FACEBOOK_DENOISER_REFERENCE_DIR").expect(
        "VOKRA_FACEBOOK_DENOISER_REFERENCE_DIR is required when VOKRA_FACEBOOK_DENOISER_GGUF is set",
    );
    let directory = Path::new(&reference_dir);
    let pcm = f32_file(&directory.join("pcm.f32le"));
    let enhanced = f32_file(&directory.join("waveform.f32le"));
    let gguf = GgufFile::open(gguf_path).expect("open strict Facebook Denoiser GGUF");
    Some((gguf, pcm, enhanced))
}

#[test]
fn cpu_matches_official_dns48_forward() {
    let Some((gguf, pcm, reference)) = fixture() else {
        return;
    };
    let model = FbDenoiser::from_gguf(&gguf).expect("strict DNS48 bind");
    let actual = model.denoise(&pcm).expect("CPU DNS48 enhancement");
    compare(
        "CPU vs official",
        &actual,
        &reference,
        required_bound("VOKRA_FACEBOOK_DENOISER_MAX_ABS_BOUND"),
        required_bound("VOKRA_FACEBOOK_DENOISER_REL_L1_BOUND"),
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_cpu_dns48_forward() {
    use vokra_core::BackendKind;

    let Some((gguf, pcm, _reference)) = fixture() else {
        return;
    };
    let cpu = FbDenoiser::from_gguf(&gguf).expect("strict DNS48 CPU bind");
    let metal = FbDenoiser::from_gguf_with_backend(&gguf, BackendKind::Metal)
        .expect("strict DNS48 Metal bind");
    let cpu = cpu.denoise(&pcm).expect("CPU DNS48 enhancement");
    let metal = metal.denoise(&pcm).expect("Metal DNS48 enhancement");
    compare(
        "Metal vs CPU",
        &metal,
        &cpu,
        required_bound("VOKRA_FACEBOOK_DENOISER_METAL_MAX_ABS_BOUND"),
        required_bound("VOKRA_FACEBOOK_DENOISER_METAL_REL_L1_BOUND"),
    );
}
