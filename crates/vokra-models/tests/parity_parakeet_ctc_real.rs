//! Gated real-checkpoint parity against the pinned upstream Transformers
//! Parakeet-CTC implementation. Unset environment variables skip honestly.

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_models::parakeet_ctc::ParakeetCtcAsr;

// Declared before the first VAST measurement. These are deliberately below
// the repository-wide FP32 atol=0.01 while leaving room for 42 layers of
// different-but-valid GEMM accumulation order. A failure is investigated; the
// bounds are not widened to fit an observation.
const ENCODER_MAX_ABS_BOUND: f32 = 2.0e-4;
const ENCODER_MEAN_ABS_BOUND: f32 = 2.0e-5;
const LOGITS_MAX_ABS_BOUND: f32 = 1.0e-3;
const LOGITS_MEAN_ABS_BOUND: f32 = 1.0e-4;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read f32 fixture");
    assert_eq!(bytes.len() % 4, 0, "f32 fixture is truncated");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).expect("read u32 fixture");
    assert_eq!(bytes.len() % 4, 0, "u32 fixture is truncated");
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn compare(label: &str, actual: &[f32], expected: &[f32], max_bound: f32, mean_bound: f32) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let (max_index, max_abs) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (left, right))| (index, (left - right).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty parity tensor");
    let mean_abs = actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / actual.len() as f32;
    eprintln!(
        "Parakeet-CTC {label}: max_abs={max_abs:.9e} at {max_index} (actual={:.9e}, reference={:.9e}), mean_abs={mean_abs:.9e}",
        actual[max_index], expected[max_index]
    );
    assert!(max_abs <= max_bound, "{label} max {max_abs} > {max_bound}");
    assert!(
        mean_abs <= mean_bound,
        "{label} mean {mean_abs} > {mean_bound}"
    );
}

#[test]
fn real_parakeet_ctc_pcm_encoder_logits_tokens_and_text_match_official() {
    let Ok(gguf) = std::env::var("VOKRA_PARAKEET_CTC_GGUF") else {
        eprintln!("skipping Parakeet-CTC real parity: set VOKRA_PARAKEET_CTC_GGUF");
        return;
    };
    let reference_dir = std::env::var("VOKRA_PARAKEET_CTC_REFERENCE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parakeet_ctc")
        });
    let file = GgufFile::open(gguf).expect("open Parakeet-CTC GGUF");
    let model = ParakeetCtcAsr::from_gguf(&file).expect("strict Parakeet-CTC bind");
    assert_eq!(model.bound_tensor_count(), Some(1_652));
    assert!(model.has_tokenizer());

    let pcm = read_f32(&reference_dir.join("pcm.f32"));
    let expected_encoder = read_f32(&reference_dir.join("encoder.f32"));
    let expected_logits = read_f32(&reference_dir.join("logits.f32"));
    let expected_raw_argmax = read_u32(&reference_dir.join("raw_argmax.u32"));
    let expected_tokens = read_u32(&reference_dir.join("tokens.u32"));
    let expected_text = std::fs::read_to_string(reference_dir.join("text.txt"))
        .expect("read text fixture")
        .trim_end()
        .to_owned();

    let (encoder, frames) = model.encode_pcm(&pcm).expect("native encoder");
    assert_eq!(
        frames,
        expected_encoder.len() / model.config().encoder.d_model,
        "fixture encoder frame count"
    );
    compare(
        "encoder",
        &encoder,
        &expected_encoder,
        ENCODER_MAX_ABS_BOUND,
        ENCODER_MEAN_ABS_BOUND,
    );
    let head_from_reference_encoder = model
        .logits(&expected_encoder, frames)
        .expect("native CTC head on reference encoder");
    compare(
        "head_from_reference_encoder",
        &head_from_reference_encoder,
        &expected_logits,
        LOGITS_MAX_ABS_BOUND,
        LOGITS_MEAN_ABS_BOUND,
    );
    let logits = model.logits(&encoder, frames).expect("native CTC head");
    compare(
        "logits",
        &logits,
        &expected_logits,
        LOGITS_MAX_ABS_BOUND,
        LOGITS_MEAN_ABS_BOUND,
    );
    let actual_raw_argmax = logits
        .chunks_exact(model.config().head.vocab_size)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index as u32)
                .expect("non-empty CTC logits row")
        })
        .collect::<Vec<_>>();
    assert_eq!(actual_raw_argmax, expected_raw_argmax, "raw CTC argmax ids");
    assert_eq!(
        model.transcribe_tokens(&pcm).expect("native CTC decode"),
        expected_tokens,
        "collapsed CTC token ids"
    );
    assert_eq!(
        model
            .transcribe_text(&pcm)
            .expect("native tokenizer decode"),
        expected_text,
        "official BPE + Metaspace text"
    );
}
