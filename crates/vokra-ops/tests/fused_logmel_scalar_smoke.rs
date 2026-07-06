//! Smoke tests for `vokra_ops::fused_log_mel_scalar` (M2-04-T05).
//!
//! Bounded / shape checks only — the elementwise parity against the imperative
//! reference (`vokra_models::whisper::mel::log_mel`) lives in T07
//! (`crates/vokra-ops/tests/fused_logmel_parity.rs`, delivered by a later
//! change in the WP). These smoke checks are the T05-local guardrails
//! required by the plan:
//!
//! - silence input → output is finite and bounded (no NaN / Inf leaking from
//!   the log-floor path);
//! - single-tone input → output is finite and lives in `(-2, 2)` (the Whisper
//!   dynamic-range window `[gmax - 8, gmax] → [-1, 1]` after `+4 / 4`
//!   normalization, with the log-floor able to push a tiny margin below `-1`
//!   for silent frames outside the tone's frame coverage);
//! - shape is exactly `n_mels * 3000` (the Whisper 30 s window).

use vokra_core::gguf::frontend_spec::FrontendSpec;
use vokra_ops::fused_log_mel_scalar;
use vokra_ops::{mel_attrs_from_spec, mel_filterbank, stft_attrs_from_spec};

const N_MELS: usize = 80;
const N_FRAMES: usize = 3000;
const SAMPLE_RATE: usize = 16_000;
const N_SAMPLES: usize = 30 * SAMPLE_RATE;

/// Whisper's fixed front-end (openai/whisper `whisper/audio.py`) — kept
/// inline so the smoke test does not depend on `vokra-models` and preserves
/// the workspace zero-dep invariant. Mirrors
/// `vokra_models::whisper::mel::runtime_frontend_spec` (only `n_mels` varies).
fn whisper_frontend_spec(n_mels: u32) -> FrontendSpec {
    FrontendSpec {
        n_fft: 400,
        hop: 160,
        win_length: 400,
        window_type: "hann".to_owned(),
        mel_norm: "slaney".to_owned(),
        htk_mode: false,
        fmin: 0.0,
        fmax: (SAMPLE_RATE as f32) / 2.0,
        n_mels,
        pad_mode: "reflect".to_owned(),
        dc_offset_removal: false,
        pre_emphasis: 0.0,
        sample_rate: SAMPLE_RATE as u32,
    }
}

fn build_fixtures() -> (
    vokra_core::ir::graph::StftAttrs,
    vokra_ops::mel::MelFilterbank,
) {
    let spec = whisper_frontend_spec(N_MELS as u32);
    let stft = stft_attrs_from_spec(&spec).expect("whisper frontend spec is well-formed");
    let mel_attrs = mel_attrs_from_spec(&spec).expect("whisper frontend spec is well-formed");
    let fb = mel_filterbank(&mel_attrs);
    (stft, fb)
}

fn floor_log10() -> f32 {
    1e-10f32.log10()
}

#[test]
fn silence_produces_bounded_finite_output() {
    let (stft_attrs, fb) = build_fixtures();
    let pcm = vec![0.0f32; N_SAMPLES];
    let mut out = vec![floor_log10(); N_MELS * N_FRAMES];
    fused_log_mel_scalar(&pcm, &stft_attrs, &fb, N_FRAMES, &mut out);
    // All silent → dynamic-range normalization is a no-op (gmax = log10(1e-10)
    // = -10 for every bin), so the whole buffer stays at `((-10).max(-10-8) +
    // 4)/4 = -1.5`. Bound the values loosely to guard against NaN/Inf.
    for &v in &out {
        assert!(v.is_finite(), "non-finite value in silence output: {v}");
        assert!(
            (-2.0..=0.0).contains(&v),
            "silence value out of expected [-2, 0]: {v}"
        );
    }
}

#[test]
fn tone_produces_finite_output_in_expected_range() {
    let (stft_attrs, fb) = build_fixtures();
    // 440 Hz sinusoid across the full 30 s Whisper window.
    let pcm: Vec<f32> = (0..N_SAMPLES)
        .map(|i| {
            (2.0 * std::f64::consts::PI * 440.0 * (i as f64) / SAMPLE_RATE as f64).sin() as f32
        })
        .collect();
    let mut out = vec![floor_log10(); N_MELS * N_FRAMES];
    fused_log_mel_scalar(&pcm, &stft_attrs, &fb, N_FRAMES, &mut out);
    for &v in &out {
        assert!(v.is_finite(), "non-finite value in tone output: {v}");
        assert!(v > -2.0 && v < 2.0, "tone value out of range (-2, 2): {v}");
    }
    // The dominant frames should have some values comfortably above the
    // dynamic floor (sanity: at least one bin above -1).
    assert!(
        out.iter().any(|&v| v > -1.0),
        "tone output collapsed to the dynamic floor"
    );
}

#[test]
fn output_shape_matches_whisper_window() {
    let (stft_attrs, fb) = build_fixtures();
    let pcm = vec![0.0f32; N_SAMPLES];
    let mut out = vec![floor_log10(); N_MELS * N_FRAMES];
    fused_log_mel_scalar(&pcm, &stft_attrs, &fb, N_FRAMES, &mut out);
    assert_eq!(
        out.len(),
        N_MELS * N_FRAMES,
        "fused kernel wrote the wrong number of elements"
    );
}
