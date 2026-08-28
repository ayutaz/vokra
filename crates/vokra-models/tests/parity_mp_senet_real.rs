//! Independent real-weight MP-SENet parity against the official PyTorch forward.
//!
//! The reference directory must be produced by
//! `tools/parity/mp_senet_dump_reference.py`. Bounds are deliberately required
//! from the environment until the first VAST measurement calibrates and
//! records them; this test does not invent a tolerance before observing the
//! independent oracle.

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_models::mp_senet::MpSenet;

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
            "{name} is required when VOKRA_MP_SENET_GGUF is set; calibrate it from the recorded independent VAST comparison"
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
        .expect("non-empty MP-SENet output");
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
        "MP-SENet {label}: max_abs={max_abs:.9e} at {index} (actual={:.9e}, reference={:.9e}), relative_l1={relative_l1:.9e}",
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
    let Some(gguf_path) = std::env::var_os("VOKRA_MP_SENET_GGUF") else {
        eprintln!(
            "[parity_mp_senet_real] SKIP: set VOKRA_MP_SENET_GGUF and VOKRA_MP_SENET_REFERENCE_DIR after generating an official VAST reference"
        );
        return None;
    };
    let reference_dir = std::env::var_os("VOKRA_MP_SENET_REFERENCE_DIR")
        .expect("VOKRA_MP_SENET_REFERENCE_DIR is required when VOKRA_MP_SENET_GGUF is set");
    let directory = Path::new(&reference_dir);
    let pcm = f32_file(&directory.join("pcm.f32le"));
    let enhanced = f32_file(&directory.join("waveform.f32le"));
    let gguf = GgufFile::open(gguf_path).expect("open strict MP-SENet GGUF");
    Some((gguf, pcm, enhanced))
}

#[test]
fn cpu_matches_official_mp_senet_forward() {
    let Some((gguf, pcm, reference)) = fixture() else {
        return;
    };
    let model = MpSenet::from_gguf(&gguf).expect("strict MP-SENet bind");
    let actual = model.enhance(&pcm).expect("CPU MP-SENet enhancement");
    compare(
        "CPU vs official",
        &actual,
        &reference,
        required_bound("VOKRA_MP_SENET_MAX_ABS_BOUND"),
        required_bound("VOKRA_MP_SENET_REL_L1_BOUND"),
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_cpu_mp_senet_forward() {
    use vokra_core::BackendKind;

    let Some((gguf, pcm, _reference)) = fixture() else {
        return;
    };
    let cpu = MpSenet::from_gguf(&gguf).expect("strict MP-SENet CPU bind");
    let metal = MpSenet::from_gguf_with_backend(&gguf, BackendKind::Metal)
        .expect("strict MP-SENet Metal bind");
    let cpu = cpu.enhance(&pcm).expect("CPU MP-SENet enhancement");
    let metal = metal.enhance(&pcm).expect("Metal MP-SENet enhancement");
    compare(
        "Metal vs CPU",
        &metal,
        &cpu,
        required_bound("VOKRA_MP_SENET_METAL_MAX_ABS_BOUND"),
        required_bound("VOKRA_MP_SENET_METAL_REL_L1_BOUND"),
    );
}
