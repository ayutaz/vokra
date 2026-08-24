//! Independent official parity for SpeechBrain SepFormer WHAM16k enhancement.

use vokra_core::gguf::GgufFile;
use vokra_models::sepformer::{
    KEY_MODEL_CATEGORY, KEY_SEPFORMER_N_OUT, SepFormer, SepformerVariant,
};

const PCM: &[u8] = include_bytes!("fixtures/sepformer/pcm.f32.bin");
const ENCODER: &[u8] = include_bytes!("fixtures/sepformer/encoder.f32.bin");
const SEPARATED: &[u8] = include_bytes!("fixtures/sepformer/separated.f32.bin");
const MAX_ABS_BOUND: f32 = 0.01;
const MEAN_ABS_BOUND: f32 = 0.001;

fn f32s(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn compare(label: &str, actual: &[f32], expected: &[f32]) {
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
        .unwrap();
    let mean_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>()
        / actual.len() as f32;
    eprintln!(
        "SepFormer {label}: max_abs={max_abs:.9e} at {index} (actual={:.9e}, \
         reference={:.9e}), mean_abs={mean_abs:.9e}",
        actual[index], expected[index]
    );
    assert!(max_abs <= MAX_ABS_BOUND, "{label} max_abs={max_abs:.9e}");
    assert!(
        mean_abs <= MEAN_ABS_BOUND,
        "{label} mean_abs={mean_abs:.9e}"
    );
}

#[test]
fn committed_reference_has_pinned_shape_and_finite_values() {
    let pcm = f32s(PCM);
    let encoder = f32s(ENCODER);
    let separated = f32s(SEPARATED);
    assert_eq!(pcm.len(), 4_096);
    assert_eq!(encoder.len(), 256 * 511);
    assert_eq!(separated.len(), 4_096);
    assert!(pcm.iter().all(|value| value.is_finite()));
    assert!(encoder.iter().all(|value| value.is_finite()));
    assert!(separated.iter().all(|value| value.is_finite()));
}

#[test]
fn public_gguf_matches_official_encoder_and_enhanced_waveform() {
    let Some(path) = std::env::var_os("VOKRA_SEPFORMER_GGUF") else {
        eprintln!(
            "[parity_sepformer_real] SKIP: set VOKRA_SEPFORMER_GGUF to \
             vokra/sepformer-wham16k-enhancement/model.gguf"
        );
        return;
    };
    let file = GgufFile::open(&path).expect("open public SepFormer GGUF");
    let model = SepFormer::from_gguf(&file).expect("strict SepFormer bind");
    assert_eq!(model.variant(), SepformerVariant::Wham16kEnhancement);
    assert_eq!(model.n_out(), 1);
    assert_eq!(model.sample_rate(), 16_000);
    assert_eq!(model.tensor_count(), 417);

    let pcm = f32s(PCM);
    let expected_encoder = f32s(ENCODER);
    let expected_separated = f32s(SEPARATED);
    let (encoder, frames) = model.encode_features(&pcm).expect("CPU SepFormer encoder");
    assert_eq!(frames, 511);
    compare("CPU encoder vs official", &encoder, &expected_encoder);
    let separated = model.separate(&pcm).expect("CPU SepFormer separation");
    assert_eq!(separated.len(), 1);
    compare(
        "CPU enhanced waveform vs official",
        &separated[0],
        &expected_separated,
    );

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use vokra_core::BackendKind;

        let metal = SepFormer::from_gguf(&file)
            .expect("strict SepFormer bind for Metal")
            .with_backend(BackendKind::Metal);
        let (metal_encoder, metal_frames) = metal
            .encode_features(&pcm)
            .expect("Metal SepFormer encoder");
        assert_eq!(metal_frames, frames);
        compare("Metal encoder vs CPU", &metal_encoder, &encoder);
        let metal_separated = metal.separate(&pcm).expect("Metal SepFormer separation");
        assert_eq!(metal_separated.len(), 1);
        compare(
            "Metal enhanced waveform vs CPU",
            &metal_separated[0],
            &separated[0],
        );
    }
}

#[test]
fn all_seven_public_artifacts_strictly_bind() {
    let Some(directory) = std::env::var_os("VOKRA_SEPFORMER_GGUF_DIR") else {
        eprintln!(
            "[parity_sepformer_real] SKIP: set VOKRA_SEPFORMER_GGUF_DIR to a directory \
             containing all seven public SepFormer GGUFs"
        );
        return;
    };
    let rows = [
        (
            "sepformer-wsj02mix.gguf",
            SepformerVariant::Wsj02mix,
            2,
            8_000,
            None,
            "separation",
        ),
        (
            "sepformer-libri2mix.gguf",
            SepformerVariant::Libri2Mix,
            2,
            8_000,
            Some(2),
            "separation",
        ),
        (
            "sepformer-libri3mix.gguf",
            SepformerVariant::Libri3Mix,
            3,
            8_000,
            Some(3),
            "separation",
        ),
        (
            "sepformer-wham16k-enhancement.gguf",
            SepformerVariant::Wham16kEnhancement,
            1,
            16_000,
            Some(1),
            "enhancement",
        ),
        (
            "sepformer-whamr16k.gguf",
            SepformerVariant::Whamr16k,
            2,
            16_000,
            Some(1),
            "enhancement",
        ),
        (
            "sepformer-whamr-8khz.gguf",
            SepformerVariant::Whamr8k,
            2,
            8_000,
            Some(1),
            "enhancement",
        ),
        (
            "sepformer-dns4.gguf",
            SepformerVariant::Dns4Enhancement,
            1,
            16_000,
            Some(1),
            "enhancement",
        ),
    ];
    for (
        file_name,
        expected_variant,
        expected_outputs,
        expected_rate,
        published_outputs,
        published_category,
    ) in rows
    {
        let path = std::path::Path::new(&directory).join(file_name);
        let file = GgufFile::open(&path)
            .unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
        let model = SepFormer::from_gguf(&file)
            .unwrap_or_else(|error| panic!("strict bind {}: {error}", path.display()));
        assert_eq!(model.variant(), expected_variant, "{file_name}");
        assert_eq!(model.n_out(), expected_outputs, "{file_name}");
        assert_eq!(model.sample_rate(), expected_rate, "{file_name}");
        assert_eq!(model.tensor_count(), 417, "{file_name}");
        assert_eq!(
            file.get(KEY_SEPFORMER_N_OUT)
                .and_then(|value| value.as_u64()),
            published_outputs,
            "pin the audited public metadata, including legacy repairs"
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY)
                .and_then(|value| value.as_str()),
            Some(published_category),
            "pin the audited public category stamp"
        );
    }
}
