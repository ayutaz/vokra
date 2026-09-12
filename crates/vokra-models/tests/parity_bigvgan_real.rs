//! Real-weight BigVGAN parity against NVIDIA's upstream Python forward.
//!
//! Every released BigVGAN variant has an explicit test entry. A test only
//! runs when both its GGUF and reference paths are opted in through the
//! variant-specific environment variables; missing paths are a clean skip,
//! never a pass for another variant.

use std::{env, fs};

use vokra_core::gguf::GgufFile;
use vokra_models::bigvgan::{BigVGan, BigVGanVariant};

const CPU_ATOL: f32 = 2.0e-5;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
const METAL_ATOL: f32 = 0.01;

struct VariantSpec {
    slug: &'static str,
    gguf_env: &'static str,
    reference_env: &'static str,
    variant: BigVGanVariant,
}

// D2-D5. The shell workers select one row explicitly; there is no
// default-to-base behaviour for a requested variant.
const VARIANTS: &[VariantSpec] = &[
    VariantSpec {
        slug: "v2_22khz_80band_256x",
        gguf_env: "VOKRA_BIGVGAN_V2_22KHZ_80BAND_256X_GGUF",
        reference_env: "VOKRA_BIGVGAN_V2_22KHZ_80BAND_256X_REFERENCE",
        variant: BigVGanVariant::V2_22khz80Band256x,
    },
    VariantSpec {
        slug: "v2_44khz_128band_512x",
        gguf_env: "VOKRA_BIGVGAN_V2_44KHZ_128BAND_512X_GGUF",
        reference_env: "VOKRA_BIGVGAN_V2_44KHZ_128BAND_512X_REFERENCE",
        variant: BigVGanVariant::V2_44khz128Band512x,
    },
    VariantSpec {
        slug: "v2_24khz_100band_256x",
        gguf_env: "VOKRA_BIGVGAN_V2_24KHZ_100BAND_256X_GGUF",
        reference_env: "VOKRA_BIGVGAN_V2_24KHZ_100BAND_256X_REFERENCE",
        variant: BigVGanVariant::V2_24khz100Band256x,
    },
    VariantSpec {
        slug: "base_v1_24khz_100band",
        gguf_env: "VOKRA_BIGVGAN_BASE_V1_24KHZ_100BAND_GGUF",
        reference_env: "VOKRA_BIGVGAN_BASE_V1_24KHZ_100BAND_REFERENCE",
        variant: BigVGanVariant::BaseV1_24khz100Band,
    },
];

fn run_real_parity(spec: &VariantSpec) {
    let Some(path) = env::var(spec.gguf_env).ok() else {
        eprintln!(
            "{} unset — skipping real BigVGAN {} parity; clean skip, not a fabricated pass",
            spec.gguf_env, spec.slug
        );
        return;
    };
    let file = GgufFile::open(&path).unwrap_or_else(|error| {
        panic!("open opted-in BigVGAN {} GGUF {}: {error}", spec.slug, path);
    });
    let model = BigVGan::from_gguf(&file).expect("bind complete BigVGAN tensor manifest");
    assert_eq!(
        model.variant(),
        spec.variant,
        "GGUF variant identity must be fixed"
    );

    let reference_path = env::var(spec.reference_env).unwrap_or_else(|_| {
        panic!(
            "{} must point to the VAST-generated official NVIDIA reference when {} is set; \
             a fixture from another variant is never a real-weight fallback",
            spec.reference_env, spec.gguf_env
        )
    });
    let fixture = fs::read_to_string(&reference_path).unwrap_or_else(|error| {
        panic!(
            "read opted-in BigVGAN {} reference {}: {error}",
            spec.slug, reference_path
        );
    });
    let mut rows = fixture.lines();
    let input_row: Vec<&str> = rows.next().expect("input row").split(',').collect();
    let output_row: Vec<&str> = rows.next().expect("output row").split(',').collect();
    assert_eq!(input_row[0], "input");
    assert_eq!(output_row[0], "output");
    assert!(
        rows.next().is_none(),
        "reference must contain exactly two rows"
    );

    let mel: Vec<f32> = input_row[1..]
        .iter()
        .map(|value| value.parse::<f32>().expect("input f32"))
        .collect();
    let expected: Vec<f32> = output_row[1..]
        .iter()
        .map(|value| value.parse::<f32>().expect("output f32"))
        .collect();
    assert_eq!(mel.len(), model.config().in_channels as usize);
    assert_eq!(
        expected.len(),
        model.config().total_upsample_factor() as usize
    );
    assert!(mel.iter().all(|value| value.is_finite()));
    assert!(expected.iter().all(|value| value.is_finite()));

    let actual = model.decode(&mel, 1).expect("native BigVGAN forward");
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|value| value.is_finite()));
    let max_abs = actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (value - reference).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "BIGVGAN_CPU_PARITY_METRICS variant={} samples={} max_abs={max_abs:.9} atol={CPU_ATOL:.9} \
         reference=NVIDIA.BigVGAN fixture=vast_generated_official",
        spec.slug,
        expected.len()
    );
    assert!(
        max_abs <= CPU_ATOL,
        "BigVGAN {} max |Δ| {max_abs:e} exceeds the registered {CPU_ATOL:e} FP32 bound",
        spec.slug
    );
    eprintln!(
        "BIGVGAN_CPU_PARITY_SENTINEL variant={} samples={} max_abs={max_abs:.9} atol={CPU_ATOL:.9} \
         reference=NVIDIA.BigVGAN fixture=vast_generated_official",
        spec.slug,
        expected.len()
    );

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    {
        let metal = BigVGan::from_gguf(&file)
            .expect("rebind BigVGAN for Metal")
            .with_backend(vokra_core::BackendKind::Metal)
            .decode(&mel, 1)
            .expect("real BigVGAN Metal forward; unsupported backends must fail, not fallback");
        assert_eq!(metal.len(), actual.len());
        assert!(metal.iter().all(|value| value.is_finite()));
        let gpu_max_abs = actual
            .iter()
            .zip(&metal)
            .map(|(cpu, gpu)| (cpu - gpu).abs())
            .fold(0.0f32, f32::max);
        assert!(
            gpu_max_abs <= METAL_ATOL,
            "BigVGAN {} CPU/Metal max |Δ| {gpu_max_abs:e} exceeds the established FP32 GPU gate",
            spec.slug
        );
        eprintln!(
            "BIGVGAN_METAL_PARITY_METRICS variant={} samples={} max_abs={gpu_max_abs:.9} atol={METAL_ATOL:.9} \
             route=resident_one_final_readback reference=CPU",
            spec.slug,
            actual.len()
        );
        eprintln!(
            "BIGVGAN_METAL_PARITY_SENTINEL variant={} samples={} max_abs={gpu_max_abs:.9} atol={METAL_ATOL:.9} \
             route=resident_one_final_readback reference=CPU",
            spec.slug,
            actual.len()
        );
    }
}

#[test]
fn parity_bigvgan_v2_22khz_80band_256x_real_weight_mel_to_waveform() {
    run_real_parity(&VARIANTS[0]);
}

#[test]
fn parity_bigvgan_v2_44khz_128band_512x_real_weight_mel_to_waveform() {
    run_real_parity(&VARIANTS[1]);
}

#[test]
fn parity_bigvgan_v2_24khz_100band_256x_real_weight_mel_to_waveform() {
    run_real_parity(&VARIANTS[2]);
}

#[test]
fn parity_bigvgan_base_v1_24khz_100band_real_weight_mel_to_waveform() {
    run_real_parity(&VARIANTS[3]);
}
