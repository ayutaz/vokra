//! Gated decoder/head parity against the official Transformers Parakeet-TDT
//! implementation. An unset environment skips rather than fabricating data.

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_models::parakeet::ParakeetAsr;

/// Honest FP32 GEMV/LSTM accumulation envelope against PyTorch eager.
///
/// The 2026-08-21 VAST calibration used the audited upstream revision and
/// public 699-tensor GGUF with token ids 0, 1, 4096, and 8192. The measured
/// worst max-|Δ| was 5.493164062e-4 (on a logit whose magnitude was about
/// 6.26e2), and the worst mean-|Δ| was 8.496245573e-5. The fixed bounds are
/// roughly 2× those measured GEMV-order floors; joint argmax is checked
/// independently so a head-layout or activation error still fails loudly.
const MAX_ABS_BOUND: f32 = 1.2e-3;
const MEAN_ABS_BOUND: f32 = 2.0e-4;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read parity f32 file");
    assert_eq!(bytes.len() % 4, 0, "f32 file must not be truncated");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
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
