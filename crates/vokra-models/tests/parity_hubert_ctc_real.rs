//! Independent official parity for HuBERT-Large-LS960 CTC.

use std::path::Path;

use vokra_models::hubert::HubertCtc;

const PCM: &[u8] = include_bytes!("fixtures/hubert_ctc/pcm.f32.bin");
const ENCODER: &[u8] = include_bytes!("fixtures/hubert_ctc/encoder.f32.bin");
const LOGITS: &[u8] = include_bytes!("fixtures/hubert_ctc/logits.f32.bin");
const TOKENS: &[u8] = include_bytes!("fixtures/hubert_ctc/tokens.u32.bin");
const TEXT: &str = include_str!("fixtures/hubert_ctc/text.txt");
const MAX_ABS_BOUND: f32 = 0.01;
const MEAN_ABS_BOUND: f32 = 0.001;

fn f32s(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn u32s(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn compare(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
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
        "HuBERT {label}: max_abs={max_abs:.9e} at {index} (actual={:.9e}, \
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
fn committed_reference_has_pinned_shape() {
    assert_eq!(f32s(PCM).len(), 32_000);
    assert_eq!(f32s(ENCODER).len(), 99 * 1024);
    assert_eq!(f32s(LOGITS).len(), 99 * 32);
    assert_eq!(u32s(TOKENS).len(), 23);
    assert_eq!(TEXT.trim_end(), "SO MY FELLOW AMERICANS");
}

#[test]
fn public_gguf_matches_official_encoder_logits_tokens_and_text() {
    let Some(path) = std::env::var_os("VOKRA_HUBERT_LARGE_GGUF") else {
        eprintln!(
            "[parity_hubert_ctc_real] SKIP: set VOKRA_HUBERT_LARGE_GGUF to \
             vokra/hubert-large-ls960/model.gguf"
        );
        return;
    };
    let model = HubertCtc::from_gguf(Path::new(&path)).expect("strict HuBERT bind");
    assert_eq!(model.config().model_id, "hubert-large-ls960");
    let pcm = f32s(PCM);
    let expected_encoder = f32s(ENCODER);
    let expected_logits = f32s(LOGITS);
    let expected_tokens = u32s(TOKENS);
    let (encoder, frames) = model.encode_features(&pcm).expect("CPU HuBERT encoder");
    assert_eq!(frames, 99);
    compare("CPU encoder vs official", &encoder, &expected_encoder);
    let (logits, logits_frames) = model.logits(&pcm).expect("CPU HuBERT logits");
    assert_eq!(logits_frames, frames);
    compare("CPU logits vs official", &logits, &expected_logits);
    assert_eq!(model.transcribe_tokens(&pcm).unwrap(), expected_tokens);
    assert_eq!(model.transcribe_text(&pcm).unwrap(), TEXT.trim_end());

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use vokra_core::BackendKind;

        let metal = HubertCtc::from_gguf(Path::new(&path))
            .expect("strict HuBERT bind for Metal")
            .with_backend(BackendKind::Metal);
        let (metal_encoder, metal_frames) = metal.encode_features(&pcm).expect("Metal encoder");
        assert_eq!(metal_frames, frames);
        compare("Metal encoder vs CPU", &metal_encoder, &encoder);
        let (metal_logits, metal_logits_frames) = metal.logits(&pcm).expect("Metal logits");
        assert_eq!(metal_logits_frames, frames);
        compare("Metal logits vs CPU", &metal_logits, &logits);
        assert_eq!(metal.transcribe_tokens(&pcm).unwrap(), expected_tokens);
    }
}
