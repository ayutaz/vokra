//! Real-artifact regression for public piper-plus voices without the optional
//! zero-shot `spk_proj` MLP.
//!
//! Set `VOKRA_PIPER_LEGACY_GGUF` to either public GGUF. CI skips when unset;
//! owner verification runs this once per published artifact.

use vokra_core::VokraError;
use vokra_models::piper_plus::PiperPlusTts;

#[test]
fn public_legacy_voice_loads_synthesizes_and_rejects_speaker_embedding() {
    let Ok(path) = std::env::var("VOKRA_PIPER_LEGACY_GGUF") else {
        eprintln!("skipping public legacy piper regression: VOKRA_PIPER_LEGACY_GGUF unset");
        return;
    };
    let voice = PiperPlusTts::from_path(&path).expect("load public legacy piper voice");
    assert_eq!(
        voice.speaker_embedding_dim(),
        0,
        "legacy voice must report that zero-shot speaker input is unsupported"
    );

    let n = voice.config().num_symbols;
    let ids = [1_i64, 5, 9].map(|id| id % n as i64);
    let audio = voice
        .synthesize_phonemes(&ids, 0, None, None, 0.0, 1.0, 0.0)
        .expect("legacy language-only synthesis");
    assert!(!audio.samples.is_empty(), "legacy voice produced no PCM");
    assert!(
        audio.samples.iter().all(|sample| sample.is_finite()),
        "legacy voice produced non-finite PCM"
    );

    let error = voice
        .synthesize_phonemes(&ids, 0, Some(&[0.0]), None, 0.0, 1.0, 0.0)
        .expect_err("legacy voice must reject unsupported speaker embedding");
    assert!(
        matches!(error, VokraError::InvalidArgument(ref message) if message.contains("has no spk_proj")),
        "unexpected legacy speaker error: {error}"
    );
}
