//! Gated decoder/head parity against the official Transformers Parakeet-TDT
//! implementation. An unset environment skips rather than fabricating data.

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_models::parakeet::{ParakeetAsr, ParakeetTokenizer};

/// Honest FP32 GEMV/LSTM accumulation envelope against PyTorch eager.
///
/// The 2026-08-21 VAST calibration used the audited upstream revision and
/// public 699-tensor GGUF with token ids 0, 1, 4096, and 8192. The measured
/// worst max-|Δ| was 5.493164062e-4 (on a logit whose magnitude was about
/// 6.26e2), and the worst mean-|Δ| was 9.052013047e-5. The fixed bounds are
/// roughly 2× those measured GEMV-order floors; joint argmax is checked
/// independently so a head-layout or activation error still fails loudly.
const MAX_ABS_BOUND: f32 = 1.2e-3;
const MEAN_ABS_BOUND: f32 = 2.0e-4;

/// Full raw-PCM encoder accumulation envelope against PyTorch eager.
///
/// The 2026-08-22 VAST calibration used the pinned upstream revision and a
/// deterministic 16,000-sample three-tone fixture. Across 13x1024 outputs it
/// measured max-|Delta| 1.012161374e-5 and mean-|Delta| 1.245580734e-6. The
/// fixed gates are about 2.5x those independently measured f32-order floors.
const PCM_ENCODER_MAX_ABS_BOUND: f32 = 2.5e-5;
const PCM_ENCODER_MEAN_ABS_BOUND: f32 = 3.0e-6;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read parity f32 file");
    assert_eq!(bytes.len() % 4, 0, "f32 file must not be truncated");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).expect("read parity u32 file");
    assert_eq!(bytes.len() % 4, 0, "u32 file must not be truncated");
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn real_parakeet_tdt_head_step_matches_official() {
    let (Ok(gguf), Ok(encoder_hidden), Ok(reference)) = (
        std::env::var("VOKRA_PARAKEET_TDT_GGUF"),
        std::env::var("VOKRA_PARAKEET_TDT_ENCODER_HIDDEN"),
        std::env::var("VOKRA_PARAKEET_TDT_REFERENCE"),
    ) else {
        eprintln!(
            "skipping Parakeet-TDT real parity: set VOKRA_PARAKEET_TDT_GGUF, VOKRA_PARAKEET_TDT_ENCODER_HIDDEN and VOKRA_PARAKEET_TDT_REFERENCE"
        );
        return;
    };
    let file = GgufFile::open(&gguf).expect("open Parakeet-TDT GGUF");
    let model = ParakeetAsr::from_gguf(&file).expect("strict Parakeet-TDT bind");
    assert_eq!(model.tensor_count(), 699);
    let input = read_f32(Path::new(&encoder_hidden));
    let token_id = std::env::var("VOKRA_PARAKEET_TDT_TOKEN_ID")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8192);
    let actual = model
        .tdt_head_step(&input, token_id)
        .expect("real decoder/head step");
    let expected = read_f32(Path::new(&reference));
    assert_eq!(actual.len(), expected.len());
    let (max_index, max_abs) = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .map(|(index, (left, right))| (index, (left - right).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty joint output");
    let mean_abs = actual
        .iter()
        .zip(&expected)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / actual.len() as f32;
    let actual_argmax = actual
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .expect("non-empty actual output");
    let expected_argmax = expected
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .expect("non-empty reference output");
    eprintln!(
        "Parakeet-TDT: tensors={}, joint_width={}, max_abs={max_abs:.9e} at {max_index} (actual={:.9e}, reference={:.9e}), mean_abs={mean_abs:.9e}, argmax={actual_argmax}",
        model.tensor_count(),
        actual.len(),
        actual[max_index],
        expected[max_index],
    );
    assert_eq!(actual_argmax, expected_argmax, "joint argmax must match");
    assert!(
        max_abs <= MAX_ABS_BOUND,
        "Parakeet-TDT max_abs {max_abs} exceeds fixed {MAX_ABS_BOUND} bound"
    );
    assert!(
        mean_abs <= MEAN_ABS_BOUND,
        "Parakeet-TDT mean_abs {mean_abs} exceeds fixed {MEAN_ABS_BOUND} bound"
    );
}

/// Public-artifact CPU smoke that needs no generated TDT reference directory.
///
/// This is deliberately not called parity: the independent upstream encoder
/// and token gates below remain the numerical authority. It proves that the
/// multi-gigabyte Hub file strictly binds and executes the learned raw-PCM
/// encoder plus one real prediction-LSTM/joint-head step. The independently
/// calibrated test below remains the authority for the complete token loop.
#[test]
fn real_parakeet_tdt_public_artifact_cpu_smoke() {
    let Ok(gguf) = std::env::var("VOKRA_PARAKEET_TDT_GGUF") else {
        eprintln!("skipping Parakeet-TDT public-artifact smoke: set VOKRA_PARAKEET_TDT_GGUF");
        return;
    };
    let pcm_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parakeet_ctc/pcm.f32");
    let pcm = read_f32(&pcm_path);
    let file = GgufFile::open(&gguf).expect("open public Parakeet-TDT GGUF");
    let model = ParakeetAsr::from_gguf(&file).expect("strict public Parakeet-TDT bind");
    assert_eq!(model.tensor_count(), 699);

    let (encoder, frames) = model.encode_pcm(&pcm).expect("public TDT CPU encoder");
    assert!(frames > 0);
    assert_eq!(encoder.len(), frames * model.config().encoder.d_model);
    assert!(encoder.iter().all(|value| value.is_finite()));

    let logits = model
        .tdt_head_step(
            &encoder[..model.config().encoder.d_model],
            model.config().joint.blank_token_id,
        )
        .expect("public TDT prediction/joint step");
    assert_eq!(
        logits.len(),
        model.config().joint.vocab_size + model.config().joint.durations.len()
    );
    assert!(logits.iter().all(|value| value.is_finite()));
    if !model.has_tokenizer() {
        eprintln!(
            "Parakeet-TDT public artifact has no embedded tokenizer; learned PCM/head path passed, CLI text requires gated replacement"
        );
    }
}

#[test]
fn real_parakeet_tdt_pcm_encoder_and_tokens_match_official() {
    let (Ok(gguf), Ok(reference_dir)) = (
        std::env::var("VOKRA_PARAKEET_TDT_GGUF"),
        std::env::var("VOKRA_PARAKEET_TDT_PCM_REFERENCE_DIR"),
    ) else {
        eprintln!(
            "skipping Parakeet-TDT PCM parity: set VOKRA_PARAKEET_TDT_GGUF and VOKRA_PARAKEET_TDT_PCM_REFERENCE_DIR"
        );
        return;
    };
    let reference_dir = Path::new(&reference_dir);
    let file = GgufFile::open(&gguf).expect("open Parakeet-TDT GGUF");
    let model = ParakeetAsr::from_gguf(&file).expect("strict Parakeet-TDT bind");
    let pcm = read_f32(&reference_dir.join("pcm.f32"));
    let expected_encoder = read_f32(&reference_dir.join("encoder.f32"));
    let expected_tokens = read_u32(&reference_dir.join("tokens.u32"));

    let (actual_encoder, frames) = model.encode_pcm(&pcm).expect("native PCM encoder");
    assert_eq!(frames, 13, "fixture encoder frame count");
    assert_eq!(actual_encoder.len(), expected_encoder.len());
    let (max_index, max_abs) = actual_encoder
        .iter()
        .zip(&expected_encoder)
        .enumerate()
        .map(|(index, (left, right))| (index, (left - right).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty encoder output");
    let mean_abs = actual_encoder
        .iter()
        .zip(&expected_encoder)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / actual_encoder.len() as f32;
    eprintln!(
        "Parakeet-TDT PCM encoder: frames={frames}, max_abs={max_abs:.9e} at {max_index} (actual={:.9e}, reference={:.9e}), mean_abs={mean_abs:.9e}",
        actual_encoder[max_index], expected_encoder[max_index]
    );
    assert!(
        max_abs <= PCM_ENCODER_MAX_ABS_BOUND,
        "Parakeet-TDT PCM encoder max_abs {max_abs} exceeds fixed {PCM_ENCODER_MAX_ABS_BOUND} bound"
    );
    assert!(
        mean_abs <= PCM_ENCODER_MEAN_ABS_BOUND,
        "Parakeet-TDT PCM encoder mean_abs {mean_abs} exceeds fixed {PCM_ENCODER_MEAN_ABS_BOUND} bound"
    );

    let actual_tokens = model.transcribe(&pcm).expect("native TDT greedy decode");
    assert_eq!(actual_tokens, expected_tokens, "TDT emitted token ids");
    let tokenizer = ParakeetTokenizer::from_gguf(&file, 8193).expect("embedded tokenizer");
    assert_eq!(
        tokenizer
            .decode(&actual_tokens, 8192, 2, 3)
            .expect("native tokenizer decode"),
        "Oh"
    );
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn real_parakeet_tdt_metal_matches_cpu() {
    let (Ok(gguf), Ok(reference_dir)) = (
        std::env::var("VOKRA_PARAKEET_TDT_GGUF"),
        std::env::var("VOKRA_PARAKEET_TDT_PCM_REFERENCE_DIR"),
    ) else {
        eprintln!(
            "skipping Parakeet-TDT Metal parity: set VOKRA_PARAKEET_TDT_GGUF and VOKRA_PARAKEET_TDT_PCM_REFERENCE_DIR"
        );
        return;
    };
    let file = GgufFile::open(gguf).expect("open Parakeet-TDT GGUF");
    let model = ParakeetAsr::from_gguf(&file).expect("strict Parakeet-TDT bind");
    let pcm = read_f32(&Path::new(&reference_dir).join("pcm.f32"));
    let (cpu_encoder, cpu_frames) = model.encode_pcm(&pcm).expect("CPU encoder");
    let cpu_tokens = model.transcribe(&pcm).expect("CPU tokens");
    let model = model.with_backend(vokra_core::BackendKind::Metal);
    let (metal_encoder, metal_frames) = model.encode_pcm(&pcm).expect("Metal encoder");
    assert_eq!(metal_frames, cpu_frames);
    let max_abs = metal_encoder
        .iter()
        .zip(&cpu_encoder)
        .map(|(metal, cpu)| (metal - cpu).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 0.01,
        "Parakeet-TDT Metal encoder max_abs {max_abs} > 0.01"
    );
    assert_eq!(
        model.transcribe(&pcm).expect("Metal tokens"),
        cpu_tokens,
        "Metal TDT token sequence must match CPU"
    );
}
