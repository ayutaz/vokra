use super::runtime::expected_manifest;
use super::*;
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile, chunks};

#[test]
fn released_variant_configs_match_upstream() {
    let hz24 = SnacConfig::for_variant(SnacVariant::Hz24);
    assert_eq!(hz24.sample_rate, 24_000);
    assert_eq!(hz24.active_vq_strides(), [4, 2, 1]);
    assert_eq!(hz24.n_stages, 3);

    let hz44 = SnacConfig::for_variant(SnacVariant::Hz44);
    assert_eq!(hz44.sample_rate, 44_100);
    assert_eq!(hz44.active_vq_strides(), [8, 4, 2, 1]);
    assert_eq!(hz44.n_stages, 4);
}

#[test]
fn released_manifests_have_revisioned_public_counts() {
    assert_eq!(expected_manifest(SnacVariant::Hz24).len(), 269);
    assert_eq!(expected_manifest(SnacVariant::Hz44).len(), 286);
}

#[test]
fn hz44_manifest_contains_both_attention_blocks_and_four_codebooks() {
    let manifest = expected_manifest(SnacVariant::Hz44);
    assert_eq!(
        manifest.get("encoder.block.5.to_qkv.weight"),
        Some(&vec![3072, 1024])
    );
    assert_eq!(
        manifest.get("decoder.model.2.to_qkv.weight"),
        Some(&vec![4608, 1536])
    );
    assert_eq!(
        manifest.get("quantizer.quantizers.3.codebook.weight"),
        Some(&vec![4096, 8])
    );
}

#[test]
fn hz24_manifest_has_no_attention_and_three_codebooks() {
    let manifest = expected_manifest(SnacVariant::Hz24);
    assert!(!manifest.contains_key("encoder.block.5.to_qkv.weight"));
    assert!(!manifest.contains_key("decoder.model.2.to_qkv.weight"));
    assert!(manifest.contains_key("quantizer.quantizers.2.codebook.weight"));
    assert!(!manifest.contains_key("quantizer.quantizers.3.codebook.weight"));
}

#[test]
fn odd_stride_decoder_manifest_pins_output_padding_weight_shape() {
    let manifest = expected_manifest(SnacVariant::Hz44);
    assert_eq!(
        manifest.get("decoder.model.5.block.1.parametrizations.weight.original1"),
        Some(&vec![384, 192, 6])
    );
}

fn tiny_file(arch: &str, variant: &str) -> GgufFile {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, arch);
    builder.add_string(KEY_SNAC_VARIANT, variant);
    builder.add_string(chunks::KEY_MODEL_NAME, "snac-24khz");
    builder.add_string("vokra.model.category", "codec");
    builder.add_string(chunks::KEY_PROVENANCE_MODEL_ID, "snac-24khz");
    builder.add_string(chunks::KEY_PROVENANCE_LICENSE, "mit");
    builder.add_string(
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        vokra_core::LicenseClass::Permissive.as_str(),
    );
    builder.add_string("vokra.provenance.upstream_hf", "hubertsiuzdak/snac_24khz");
    builder.add_string(chunks::KEY_PROVENANCE_SOURCE, "test source");
    builder
        .add_tensor(
            "unexpected",
            GgmlType::F32,
            vec![1],
            0.0_f32.to_le_bytes().to_vec(),
        )
        .unwrap();
    GgufFile::parse(builder.to_bytes().unwrap()).unwrap()
}

#[test]
fn strict_binder_rejects_wrong_arch_before_tensor_load() {
    let error = Snac::from_gguf(&tiny_file("dac", VARIANT_TAG_HZ24)).unwrap_err();
    assert!(matches!(error, vokra_core::VokraError::ModelLoad(_)));
    assert!(error.to_string().contains("vokra.model.arch"));
}

#[test]
fn strict_binder_rejects_unknown_variant() {
    let error = Snac::from_gguf(&tiny_file(ARCH, "16khz")).unwrap_err();
    assert!(matches!(error, vokra_core::VokraError::ModelLoad(_)));
    assert!(error.to_string().contains(KEY_SNAC_VARIANT));
}

#[test]
fn strict_binder_rejects_partial_tensor_manifest() {
    let error = Snac::from_gguf(&tiny_file(ARCH, VARIANT_TAG_HZ24)).unwrap_err();
    assert!(matches!(error, vokra_core::VokraError::ModelLoad(_)));
    assert!(error.to_string().contains("tensor manifest mismatch"));
}

const SNAC_24_PCM: &[u8] = include_bytes!("../../tests/fixtures/snac_24khz/pcm.f32");
const SNAC_24_ENCODED: &[u8] =
    include_bytes!("../../tests/fixtures/snac_24khz/encoded_features.f32");
const SNAC_24_DECODED_FEATURES: &[u8] =
    include_bytes!("../../tests/fixtures/snac_24khz/decoded_features_time_major.f32");
const SNAC_24_DECODED_PCM: &[u8] =
    include_bytes!("../../tests/fixtures/snac_24khz/decoded_pcm.f32");
const SNAC_24_CODES: [&[u8]; 3] = [
    include_bytes!("../../tests/fixtures/snac_24khz/codes_0.u32"),
    include_bytes!("../../tests/fixtures/snac_24khz/codes_1.u32"),
    include_bytes!("../../tests/fixtures/snac_24khz/codes_2.u32"),
];
const SNAC_24_NOISE: [&[u8]; 4] = [
    include_bytes!("../../tests/fixtures/snac_24khz/noise_0.f32"),
    include_bytes!("../../tests/fixtures/snac_24khz/noise_1.f32"),
    include_bytes!("../../tests/fixtures/snac_24khz/noise_2.f32"),
    include_bytes!("../../tests/fixtures/snac_24khz/noise_3.f32"),
];
const SNAC_24_MANIFEST: &str = include_str!("../../tests/fixtures/snac_24khz/manifest.json");

const SNAC_44_PCM: &[u8] = include_bytes!("../../tests/fixtures/snac_44khz/pcm.f32");
const SNAC_44_ENCODED: &[u8] =
    include_bytes!("../../tests/fixtures/snac_44khz/encoded_features.f32");
const SNAC_44_DECODED_FEATURES: &[u8] =
    include_bytes!("../../tests/fixtures/snac_44khz/decoded_features_time_major.f32");
const SNAC_44_DECODED_PCM: &[u8] =
    include_bytes!("../../tests/fixtures/snac_44khz/decoded_pcm.f32");
const SNAC_44_CODES: [&[u8]; 4] = [
    include_bytes!("../../tests/fixtures/snac_44khz/codes_0.u32"),
    include_bytes!("../../tests/fixtures/snac_44khz/codes_1.u32"),
    include_bytes!("../../tests/fixtures/snac_44khz/codes_2.u32"),
    include_bytes!("../../tests/fixtures/snac_44khz/codes_3.u32"),
];
const SNAC_44_NOISE: [&[u8]; 4] = [
    include_bytes!("../../tests/fixtures/snac_44khz/noise_0.f32"),
    include_bytes!("../../tests/fixtures/snac_44khz/noise_1.f32"),
    include_bytes!("../../tests/fixtures/snac_44khz/noise_2.f32"),
    include_bytes!("../../tests/fixtures/snac_44khz/noise_3.f32"),
];
const SNAC_44_MANIFEST: &str = include_str!("../../tests/fixtures/snac_44khz/manifest.json");

fn fixture_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0, "truncated SNAC f32 fixture");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn fixture_u32(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0, "truncated SNAC u32 fixture");
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[derive(Debug)]
struct ParityMetrics {
    max_abs: f64,
    max_abs_index: usize,
    relative_l1: f64,
    cosine: f64,
}

fn parity_metrics(label: &str, actual: &[f32], expected: &[f32]) -> ParityMetrics {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    assert!(!actual.is_empty(), "{label} must be non-empty");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{label} must be finite"
    );
    let mut max_abs = 0.0_f64;
    let mut max_abs_index = 0usize;
    let mut sum_abs = 0.0_f64;
    let mut expected_l1 = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut actual_sq = 0.0_f64;
    let mut expected_sq = 0.0_f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let error = (actual - expected).abs();
        if error > max_abs {
            max_abs = error;
            max_abs_index = index;
        }
        sum_abs += error;
        expected_l1 += expected.abs();
        dot += actual * expected;
        actual_sq += actual * actual;
        expected_sq += expected * expected;
    }
    let metrics = ParityMetrics {
        max_abs,
        max_abs_index,
        relative_l1: sum_abs / expected_l1.max(1.0e-30),
        cosine: dot / (actual_sq.sqrt() * expected_sq.sqrt()).max(1.0e-30),
    };
    eprintln!(
        "SNAC {label}: max_abs={:.9e} at {} (actual={:.9e}, reference={:.9e}), relative_l1={:.9e}, cosine={:.9e}",
        metrics.max_abs,
        metrics.max_abs_index,
        actual[metrics.max_abs_index],
        expected[metrics.max_abs_index],
        metrics.relative_l1,
        metrics.cosine,
    );
    metrics
}

fn assert_parity(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    max_abs_bound: f64,
    relative_l1_bound: f64,
    cosine_bound: f64,
) {
    let metrics = parity_metrics(label, actual, expected);
    assert!(metrics.max_abs <= max_abs_bound, "{metrics:?}");
    assert!(metrics.relative_l1 <= relative_l1_bound, "{metrics:?}");
    assert!(metrics.cosine >= cosine_bound, "{metrics:?}");
}

struct OfficialFixture {
    variant: SnacVariant,
    env: &'static str,
    sample_rate: u32,
    pcm: &'static [u8],
    encoded: &'static [u8],
    decoded_features: &'static [u8],
    decoded_pcm: &'static [u8],
    codes: &'static [&'static [u8]],
    noises: &'static [&'static [u8]],
}

fn run_official_parity(fixture: &OfficialFixture) {
    let Some(path) = std::env::var_os(fixture.env) else {
        eprintln!(
            "[snac official parity] SKIP: set {} to the canonical public GGUF",
            fixture.env
        );
        return;
    };
    let model = Snac::from_path(path).expect("strict public SNAC bind");
    assert_eq!(model.variant(), fixture.variant);
    let pcm = fixture_f32(fixture.pcm);
    let encoded = fixture_f32(fixture.encoded);
    let decoded_features = fixture_f32(fixture.decoded_features);
    let decoded_pcm = fixture_f32(fixture.decoded_pcm);
    let codes: Vec<Vec<u32>> = fixture
        .codes
        .iter()
        .map(|bytes| fixture_u32(bytes))
        .collect();
    let noises: Vec<Vec<f32>> = fixture
        .noises
        .iter()
        .map(|bytes| fixture_f32(bytes))
        .collect();

    let actual_encoded = model
        .encode_features_for_parity(&pcm, fixture.sample_rate)
        .expect("CPU official-input encoder");
    // VAST 48584151 measured max_abs=7.8201e-5 / relative-L1=2.2713e-6
    // at 24 kHz and max_abs=3.1018e-4 / relative-L1=4.2411e-6 at 44.1 kHz.
    // The committed envelope leaves less than 1.3x max-error headroom while
    // remaining far below the project-wide FP32 0.01 default.
    assert_parity(
        "CPU encoder vs official",
        &actual_encoded,
        &encoded,
        4.0e-4,
        1.0e-5,
        0.999_999,
    );
    assert_eq!(
        model
            .encode(&pcm, fixture.sample_rate)
            .expect("CPU official-input RVQ"),
        codes,
        "SNAC CPU encode codes must match the official package exactly"
    );
    let actual_features = model
        .decode_codes_to_features(&codes)
        .expect("CPU official-code RVQ decode");
    assert_parity(
        "CPU RVQ decode vs official",
        &actual_features,
        &decoded_features,
        5.0e-6,
        1.0e-7,
        0.999_999_9,
    );
    let actual_pcm = model
        .decode_with_noise_for_parity(&codes, &noises)
        .expect("CPU official-noise decoder");
    assert_parity(
        "CPU decoder vs official",
        &actual_pcm,
        &decoded_pcm,
        1.5e-6,
        4.0e-6,
        0.999_999_9,
    );

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let metal = model.with_backend(vokra_core::BackendKind::Metal);
        let metal_features = metal
            .decode_codes_to_features(&codes)
            .expect("Metal official-code RVQ decode");
        assert_parity(
            "Metal RVQ decode vs official",
            &metal_features,
            &decoded_features,
            5.0e-5,
            5.0e-4,
            0.999_99,
        );
        let metal_pcm = metal
            .decode_with_noise_for_parity(&codes, &noises)
            .expect("Metal official-noise decoder");
        assert_parity(
            "Metal decoder vs official",
            &metal_pcm,
            &decoded_pcm,
            2.0e-3,
            1.0e-2,
            0.999,
        );
    }
}

#[test]
fn official_reference_fixtures_pin_source_checkpoints_and_shapes() {
    assert_eq!(fixture_f32(SNAC_24_PCM).len(), 1_567);
    assert_eq!(fixture_f32(SNAC_24_ENCODED).len(), 768 * 4);
    assert_eq!(fixture_f32(SNAC_24_DECODED_PCM).len(), 2_048);
    assert_eq!(fixture_f32(SNAC_44_PCM).len(), 5_003);
    assert_eq!(fixture_f32(SNAC_44_ENCODED).len(), 1_024 * 32);
    assert_eq!(fixture_f32(SNAC_44_DECODED_PCM).len(), 12_288);
    for manifest in [SNAC_24_MANIFEST, SNAC_44_MANIFEST] {
        assert!(manifest.contains("vokra-snac-reference-v1"));
        assert!(manifest.contains("8f79a718f1ad71f94f79999f0071348227aff22e"));
        assert!(manifest.contains("snac.SNAC.from_pretrained/encode/decode"));
    }
    assert!(SNAC_24_MANIFEST.contains("d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"));
    assert!(
        SNAC_24_MANIFEST
            .contains("4b8164cc6606bfa627f1a784734c1e539891518f1191ed9194fe1e3b9b4bff40")
    );
    assert!(SNAC_44_MANIFEST.contains("873ebef9718b89660340c6f55a2b515e98cfa1d9"));
    assert!(
        SNAC_44_MANIFEST
            .contains("b0a676cbdc8d1cc53186f6d777bc956fb7932ceacdc657a4c3741646e9e7ead0")
    );
}

#[test]
fn public_snac_24khz_matches_official_codec() {
    run_official_parity(&OfficialFixture {
        variant: SnacVariant::Hz24,
        env: "VOKRA_SNAC_24KHZ_GGUF",
        sample_rate: 24_000,
        pcm: SNAC_24_PCM,
        encoded: SNAC_24_ENCODED,
        decoded_features: SNAC_24_DECODED_FEATURES,
        decoded_pcm: SNAC_24_DECODED_PCM,
        codes: &SNAC_24_CODES,
        noises: &SNAC_24_NOISE,
    });
}

#[test]
fn public_snac_44khz_matches_official_codec() {
    run_official_parity(&OfficialFixture {
        variant: SnacVariant::Hz44,
        env: "VOKRA_SNAC_44KHZ_GGUF",
        sample_rate: 44_100,
        pcm: SNAC_44_PCM,
        encoded: SNAC_44_ENCODED,
        decoded_features: SNAC_44_DECODED_FEATURES,
        decoded_pcm: SNAC_44_DECODED_PCM,
        codes: &SNAC_44_CODES,
        noises: &SNAC_44_NOISE,
    });
}
