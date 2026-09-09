//! Gated real-checkpoint SpeechT5 TTS parity against Transformers 5.5.0.
//!
//! The reference directory is produced only by the official
//! `SpeechT5ForTextToSpeech.generate_speech` path. The Python oracle injects
//! the documented SplitMix64 masks into the official always-on decoder-prenet
//! dropout; it does not mirror any encoder, attention, decoder or postnet math.

use std::path::{Path, PathBuf};

use vokra_core::gguf::GgufFile;
use vokra_models::speecht5::{
    NUM_MEL_BINS, REDUCTION_FACTOR, SPEAKER_EMBEDDING_DIM, SpeechT5GenerationOptions, SpeechT5Tts,
};

/// Default project FP32 bound. It is deliberately not calibrated upward
/// before the first independent VAST observation.
const FP32_ATOL: f32 = 0.01;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| {
        panic!("read parity f32 {}: {error}", path.display());
    });
    assert_eq!(
        bytes.len() % 4,
        0,
        "{} must be whole little-endian f32 values",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| {
        panic!("read parity u32 {}: {error}", path.display());
    });
    assert_eq!(
        bytes.len() % 4,
        0,
        "{} must be whole little-endian u32 values",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte u32")))
        .collect()
}

fn read_usize(path: &Path) -> usize {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        .trim()
        .parse()
        .unwrap_or_else(|error| panic!("parse decimal {}: {error}", path.display()))
}

fn inputs() -> Option<(String, PathBuf)> {
    let gguf = std::env::var("VOKRA_SPEECHT5_TTS_GGUF").ok();
    let reference = std::env::var("VOKRA_SPEECHT5_TTS_REFERENCE_DIR").ok();
    match (gguf, reference) {
        (Some(gguf), Some(reference)) => Some((gguf, PathBuf::from(reference))),
        _ => {
            eprintln!(
                "skipping SpeechT5 TTS real parity: set VOKRA_SPEECHT5_TTS_GGUF and VOKRA_SPEECHT5_TTS_REFERENCE_DIR (clean skip, not a fabricated pass)"
            );
            None
        }
    }
}

fn max_abs(actual: &[f32], expected: &[f32]) -> (usize, f32) {
    assert_eq!(actual.len(), expected.len(), "parity vector length");
    actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty parity vector")
}

#[test]
fn released_cpu_mel_matches_official_transformers() {
    let Some((gguf, reference)) = inputs() else {
        return;
    };
    assert!(reference.is_dir(), "reference path must be a directory");
    let text =
        std::fs::read_to_string(reference.join("text.txt")).expect("read official reference text");
    let text = text.strip_suffix('\n').unwrap_or(&text);
    assert!(
        !text.is_empty(),
        "official reference text must not be empty"
    );
    let tokens = read_u32(&reference.join("tokens.u32"));
    let speaker = read_f32(&reference.join("speaker.f32"));
    let expected_before = read_f32(&reference.join("before_postnet.f32"));
    let expected_after = read_f32(&reference.join("after_postnet.f32"));
    let frames = read_usize(&reference.join("frames.txt"));
    let decoder_steps = read_usize(&reference.join("decoder_steps.txt"));
    assert_eq!(speaker.len(), SPEAKER_EMBEDDING_DIM);
    assert_eq!(frames, decoder_steps * REDUCTION_FACTOR);
    assert_eq!(expected_before.len(), frames * NUM_MEL_BINS);
    assert_eq!(expected_after.len(), frames * NUM_MEL_BINS);

    let file = GgufFile::open(&gguf).expect("open complete SpeechT5 TTS GGUF");
    let model = SpeechT5Tts::from_gguf(&file).expect("strict released SpeechT5 bind");
    assert_eq!(
        model.tokenizer().encode(text).expect("native tokenizer"),
        tokens,
        "native tokenizer IDs must exactly match official SpeechT5Tokenizer"
    );
    let cpu = model
        .generate_tokens_mel(&tokens, &speaker, SpeechT5GenerationOptions::default())
        .expect("native CPU SpeechT5 text-to-mel");
    assert_eq!(cpu.frames(), frames, "official stop frame");
    assert_eq!(cpu.decoder_steps(), decoder_steps, "official stop step");
    let (before_index, before_max_abs) = max_abs(cpu.before_postnet(), &expected_before);
    let (after_index, after_max_abs) = max_abs(cpu.values(), &expected_after);
    eprintln!(
        "SPEECHT5_TTS_OFFICIAL_PARITY backend=cpu frames={frames} decoder_steps={decoder_steps} before_max_abs={before_max_abs:.9e} before_index={before_index} after_max_abs={after_max_abs:.9e} after_index={after_index} bound={FP32_ATOL:.9e} verdict={}",
        if before_max_abs <= FP32_ATOL && after_max_abs <= FP32_ATOL {
            "PASS"
        } else {
            "FAIL"
        }
    );
    assert!(
        before_max_abs <= FP32_ATOL,
        "pre-postnet max_abs {before_max_abs} at {before_index} exceeds default FP32 bound {FP32_ATOL}; diagnose before changing the bound"
    );
    assert!(
        after_max_abs <= FP32_ATOL,
        "postnet max_abs {after_max_abs} at {after_index} exceeds default FP32 bound {FP32_ATOL}; diagnose before changing the bound"
    );

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let metal = model
            .with_backend(vokra_core::BackendKind::Metal)
            .generate_tokens_mel(&tokens, &speaker, SpeechT5GenerationOptions::default())
            .expect("native Metal SpeechT5 text-to-mel");
        assert_eq!(metal.frames(), frames, "Metal stop frame");
        assert_eq!(metal.decoder_steps(), decoder_steps, "Metal stop step");
        let (metal_before_index, metal_before_abs) =
            max_abs(metal.before_postnet(), &expected_before);
        let (metal_after_index, metal_after_abs) = max_abs(metal.values(), &expected_after);
        let (_, metal_cpu_abs) = max_abs(metal.values(), cpu.values());
        eprintln!(
            "SPEECHT5_TTS_OFFICIAL_PARITY backend=metal frames={frames} decoder_steps={decoder_steps} before_max_abs={metal_before_abs:.9e} before_index={metal_before_index} after_max_abs={metal_after_abs:.9e} after_index={metal_after_index} cpu_max_abs={metal_cpu_abs:.9e} bound={FP32_ATOL:.9e} verdict={}",
            if metal_before_abs <= FP32_ATOL
                && metal_after_abs <= FP32_ATOL
                && metal_cpu_abs <= FP32_ATOL
            {
                "PASS"
            } else {
                "FAIL"
            }
        );
        assert!(
            metal_before_abs <= FP32_ATOL,
            "Metal pre-postnet max_abs {metal_before_abs} at {metal_before_index} exceeds {FP32_ATOL}"
        );
        assert!(
            metal_after_abs <= FP32_ATOL,
            "Metal postnet max_abs {metal_after_abs} at {metal_after_index} exceeds {FP32_ATOL}"
        );
        assert!(
            metal_cpu_abs <= FP32_ATOL,
            "SpeechT5 Metal/CPU postnet max_abs {metal_cpu_abs} exceeds {FP32_ATOL}"
        );
    }
}
