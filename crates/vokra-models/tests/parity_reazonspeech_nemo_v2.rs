//! Gated real-checkpoint parity for ReazonSpeech NeMo v2.
//!
//! The reference directory is generated exclusively by the official NVIDIA
//! NeMo `EncDecRNNTBPEModel`; no values are embedded or synthesized here.

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_models::reazonspeech_nemo_v2::ReazonSpeechNemoV2;

/// Default project FP32 bound. This is intentionally not calibrated upward
/// before the first independent VAST measurement.
const FP32_ATOL: f32 = 0.01;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read parity f32 file");
    assert_eq!(bytes.len() % 4, 0, "f32 file must not be truncated");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).expect("read parity u32 file");
    assert_eq!(bytes.len() % 4, 0, "u32 file must not be truncated");
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte u32")))
        .collect()
}

fn inputs() -> Option<(String, String)> {
    let gguf = std::env::var("VOKRA_REAZONSPEECH_NEMO_V2_GGUF").ok();
    let reference = std::env::var("VOKRA_REAZONSPEECH_NEMO_V2_REFERENCE_DIR").ok();
    match (gguf, reference) {
        (Some(gguf), Some(reference)) => Some((gguf, reference)),
        _ => {
            eprintln!(
                "skipping ReazonSpeech-NeMo-v2 real parity: set VOKRA_REAZONSPEECH_NEMO_V2_GGUF and VOKRA_REAZONSPEECH_NEMO_V2_REFERENCE_DIR"
            );
            None
        }
    }
}

#[test]
fn released_cpu_encoder_and_tokens_match_official_nemo() {
    let Some((gguf, reference)) = inputs() else {
        return;
    };
    let reference = Path::new(&reference);
    assert!(reference.is_dir(), "reference path must be a directory");
    let file = GgufFile::open(&gguf).expect("open complete ReazonSpeech GGUF");
    let model = ReazonSpeechNemoV2::from_gguf(&file).expect("strict released checkpoint bind");
    assert_eq!(model.tensor_count(), 965);
    assert!(
        model.has_tokenizer(),
        "text artifact must embed tokenizer.vocab"
    );

    let pcm = read_f32(&reference.join("pcm.f32"));
    let expected_encoder = read_f32(&reference.join("encoder.f32"));
    let expected_tokens = read_u32(&reference.join("tokens.u32"));
    assert!(
        !expected_tokens.is_empty(),
        "official tokens must not be empty"
    );
    let expected_frames = std::fs::read_to_string(reference.join("encoder.frames.txt"))
        .expect("read official encoder frame count")
        .trim()
        .parse::<usize>()
        .expect("decimal encoder frame count");

    let (actual_encoder, actual_frames) = model.encode_pcm(&pcm).expect("native CPU encoder");
    assert_eq!(actual_frames, expected_frames, "encoder frame count");
    assert_eq!(actual_encoder.len(), expected_encoder.len());
    let (max_index, max_abs) = actual_encoder
        .iter()
        .zip(&expected_encoder)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty encoder output");
    let mean_abs = actual_encoder
        .iter()
        .zip(&expected_encoder)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>()
        / actual_encoder.len() as f32;
    eprintln!(
        "ReazonSpeech-NeMo-v2 CPU encoder: frames={actual_frames}, max_abs={max_abs:.9e} at {max_index} (actual={:.9e}, official={:.9e}), mean_abs={mean_abs:.9e}",
        actual_encoder[max_index], expected_encoder[max_index]
    );
    assert!(
        max_abs <= FP32_ATOL,
        "encoder max_abs {max_abs} exceeds default FP32 bound {FP32_ATOL}; diagnose the worst bin before changing the bound"
    );

    assert_eq!(
        model
            .transcribe_tokens(&pcm)
            .expect("native CPU RNN-T decode"),
        expected_tokens,
        "greedy emitted token IDs must exactly match official NeMo"
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn released_metal_matches_cpu_encoder_and_tokens() {
    let Some((gguf, reference)) = inputs() else {
        return;
    };
    let file = GgufFile::open(&gguf).expect("open complete ReazonSpeech GGUF");
    let pcm = read_f32(&Path::new(&reference).join("pcm.f32"));
    let cpu = ReazonSpeechNemoV2::from_gguf(&file).expect("strict CPU bind");
    let (cpu_encoder, cpu_frames) = cpu.encode_pcm(&pcm).expect("CPU encoder");
    let cpu_tokens = cpu.transcribe_tokens(&pcm).expect("CPU tokens");
    let metal = cpu.with_backend(vokra_core::BackendKind::Metal);
    let (metal_encoder, metal_frames) = metal.encode_pcm(&pcm).expect("Metal encoder");
    assert_eq!(metal_frames, cpu_frames);
    let max_abs = metal_encoder
        .iter()
        .zip(&cpu_encoder)
        .map(|(metal, cpu)| (metal - cpu).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= FP32_ATOL,
        "Metal encoder max_abs {max_abs} exceeds {FP32_ATOL}"
    );
    assert_eq!(
        metal.transcribe_tokens(&pcm).expect("Metal tokens"),
        cpu_tokens,
        "Metal RNN-T token sequence must match CPU exactly"
    );
}
