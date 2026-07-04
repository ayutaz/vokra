//! Zero-shot voice-cloning wiring: reference audio → CAM++ embedding → piper
//! (M0-08 stage 3 integration).
//!
//! Exercises the full replacement of the zero-embedding fallback: a real WAV is
//! turned into a 192-d CAM++ speaker embedding by
//! [`PiperPlusTts::embed_reference`], and that embedding is shown to (a) be
//! non-zero, (b) change the global conditioning `g` versus the zero fallback,
//! and (c) change the synthesized PCM. It needs BOTH a zero-shot v7 voice and a
//! CAM++ encoder, so it is gated on `VOKRA_PIPER_V7_GGUF` **and**
//! `VOKRA_CAMPLUS_GGUF` and skips cleanly (CI stays green) when either is unset:
//!
//! ```text
//! VOKRA_PIPER_V7_GGUF=v7.gguf VOKRA_CAMPLUS_GGUF=campplus.gguf \
//!   cargo test -p vokra-models clone_integration -- --nocapture
//! ```
//!
//! The fbank front-end itself has no numeric oracle yet (a Kaldi oracle is a
//! follow-up); this test validates the *wiring and behavioural contract*, while
//! the CAM++ network is separately parity-checked against onnxruntime
//! (`speaker::parity`) and the fbank op structurally (`vokra_ops::kaldi_fbank`).

use std::path::PathBuf;

use super::PiperPlusTts;
use crate::silero_vad::wav::read_wav_f32;
use crate::speaker::{EMBED_DIM, SpeakerEncoder};

/// The committed real 16 kHz mono WAV (shared with the Silero VAD fixtures).
fn reference_wav() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("silero_vad")
        .join("test_16k.wav")
}

fn load_voice() -> Option<PiperPlusTts> {
    let path = std::env::var("VOKRA_PIPER_V7_GGUF").ok()?;
    Some(PiperPlusTts::from_path(&path).expect("load v7 voice GGUF"))
}

fn load_encoder() -> Option<SpeakerEncoder> {
    let path = std::env::var("VOKRA_CAMPLUS_GGUF").ok()?;
    Some(SpeakerEncoder::from_path(&path).expect("load CAM++ GGUF"))
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

#[test]
fn reference_audio_embeds_and_changes_synthesis() {
    let (Some(voice), Some(encoder)) = (load_voice(), load_encoder()) else {
        eprintln!(
            "skipping clone integration: set VOKRA_PIPER_V7_GGUF and VOKRA_CAMPLUS_GGUF to run"
        );
        return;
    };
    assert_eq!(
        voice.speaker_embedding_dim(),
        EMBED_DIM,
        "v7 voice spk_proj expects the CAM++ 192-d embedding"
    );

    let wav = read_wav_f32(reference_wav()).expect("read reference WAV");
    assert!(!wav.samples.is_empty(), "empty reference audio");

    // (a) reference audio → non-zero 192-d embedding.
    let emb = voice
        .embed_reference(&encoder, &wav.samples, wav.sample_rate)
        .expect("embed reference audio");
    assert_eq!(emb.len(), EMBED_DIM);
    assert!(emb.iter().all(|v| v.is_finite()), "embedding has NaN/Inf");
    let energy: f32 = emb.iter().map(|v| v.abs()).sum();
    assert!(energy > 0.0, "embedding is the all-zero fallback");

    // (b) the embedding changes the global conditioning `g` versus the zero
    // fallback (spk_proj(zeros) ≠ spk_proj(embed)).
    let lid = 0;
    let g_zero = voice.global_g(None, lid);
    let g_clone = voice.global_g(Some(&emb), lid);
    let cos_g = cosine(&g_zero, &g_clone);
    eprintln!("clone: cos(g_zero, g_clone) = {cos_g:.6} (embed |·|₁ = {energy:.3})");
    assert!(
        cos_g < 0.999,
        "conditioning unchanged by the speaker embedding (cos {cos_g})"
    );

    // (c) the pipeline runs with the embedding and the PCM differs from the
    // zero-embedding default (deterministic synthesis, noise off).
    let n = voice.config().num_symbols;
    assert!(
        n >= 8,
        "voice phoneme table too small for the smoke sequence"
    );
    let ids: Vec<i64> = [1, 2, 3, 4, 5, 6, 7]
        .into_iter()
        .filter(|&id| (id as usize) < n)
        .collect();
    let length_scale = voice.config().length_scale;

    let pcm_zero = voice
        .synthesize_phonemes(&ids, lid, None, None, 0.0, length_scale, 0.0)
        .expect("synthesize with zero embedding");
    let pcm_clone = voice
        .synthesize_phonemes(&ids, lid, Some(&emb), None, 0.0, length_scale, 0.0)
        .expect("synthesize with cloned embedding");

    assert!(
        !pcm_clone.samples.is_empty(),
        "cloned synthesis produced no audio"
    );
    assert!(
        pcm_clone.samples.iter().all(|s| s.is_finite()),
        "cloned PCM has NaN/Inf"
    );
    // Different length (durations shifted) or different samples both prove the
    // embedding flows through the whole native pipeline.
    let differs = pcm_zero.samples.len() != pcm_clone.samples.len()
        || pcm_zero
            .samples
            .iter()
            .zip(&pcm_clone.samples)
            .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        differs,
        "cloned synthesis is identical to the zero-embedding default"
    );
    eprintln!(
        "clone: pcm_zero {} samples, pcm_clone {} samples — differ = {differs}",
        pcm_zero.samples.len(),
        pcm_clone.samples.len()
    );
}
