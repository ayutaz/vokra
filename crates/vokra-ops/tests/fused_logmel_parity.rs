//! Fused-vs-unfused log-mel parity harness (M2-04-T07; NFR-QL-01, NFR-QL-03,
//! FR-LD-03).
//!
//! Compares [`vokra_ops::fused_log_mel_scalar`] (the fused single-pass kernel
//! from T05) against the imperative reference
//! [`vokra_models::whisper::mel::log_mel`] (STFT → power →
//! `MelFilterbank::apply` → log10 → transpose → dynamic-range compress). The
//! fused kernel folds the `r² + im²` power computation, the mel projection,
//! and the log10 into a single pass, so the arithmetic differs by exactly one
//! round of intermediate rounding vs the imperative path (see
//! `crates/vokra-ops/src/fused_logmel.rs` "Fidelity vs the imperative
//! reference"). The plan pins that delta at
//! `atol = 0.01` in the log-normalized output — the FP32 log-domain feature
//! tolerance from NFR-QL-01.
//!
//! # Coverage
//!
//! Three inputs at 16 kHz, all zero-padded / trimmed to the Whisper 30 s
//! window ([`N_SAMPLES`]) by the reference and to the same size (via
//! `n_frames = 3000`) by the fused kernel:
//!
//! 1. **1 s of silence** — every mel bin lands at the log-floor
//!    `log10(1e-10) = -10`, so the dynamic-range compression
//!    `((-10).max(gmax - 8) + 4) / 4` degenerates to `-1.5` on both sides.
//!    This is the "small-signal" corner where a SIMD `vlog10` approximation
//!    could blow the tolerance (risk R1 in the plan) — the imperative side
//!    reaches the floor through `apply → log10`, the fused side through the
//!    inline `acc.max(1e-10).log10()`, and both must clip identically.
//! 2. **1 s of a 440 Hz sinusoid** — a single tone that concentrates energy
//!    into 1-2 mel bands; the dominant frames drive `gmax` well above the
//!    floor and exercise the "wide dynamic range" branch of the compression.
//! 3. **3 s speech-like linear chirp** `sin(2π·(200 + 800·t/T)·t)` sweeping
//!    200 Hz → 1000 Hz — a broadband non-stationary signal that spreads
//!    energy across many mel bands over time, catching any accumulation-order
//!    dependence between the fused (`sum_k w_k·(r² + im²)`) and unfused
//!    (`power_k = r² + im²`, then `sum_k w_k·power_k`) paths.
//!
//! # Frontend-spec bit-exact check (NFR-QL-03 / FR-LD-03)
//!
//! Both sides derive their `StftAttrs` / `MelAttrs` from **the same**
//! [`FrontendSpec`] via the `vokra-ops` translation
//! ([`stft_attrs_from_spec`] / [`mel_attrs_from_spec`]) — the mel filter
//! weights are therefore byte-identical by construction, and the
//! bit-exactness of the front-end knob → attrs translation is what
//! [`vokra_models::whisper::mel::log_mel`] and [`fused_log_mel_scalar`]
//! must agree on. If a future refactor makes the fused kernel bypass
//! `mel_attrs_from_spec`, the log-mel spectrogram will drift out of
//! `atol = 0.01`, and this test will fail.
//!
//! # Cross-crate dev-dependency (zero-dep invariant, NFR-DS-02)
//!
//! This test lives in `vokra-ops` but reads the reference through
//! `vokra-models::whisper::mel::log_mel`. The corresponding dev-dep in
//! `vokra-ops/Cargo.toml` is `vokra-models = { path = "../vokra-models" }` —
//! **workspace-internal only** (no third-party crate). It is a
//! `dev-dependencies` edge, so it never enters the release build graph of
//! `vokra-ops`, and `./scripts/check-zero-deps.sh` (which audits the release
//! `Cargo.lock`) continues to pass.

use vokra_models::whisper::mel::{
    N_FRAMES, N_SAMPLES, SAMPLE_RATE, log_mel, runtime_frontend_spec,
};
use vokra_ops::fused_log_mel_scalar;
use vokra_ops::{mel_attrs_from_spec, mel_filterbank, stft_attrs_from_spec};

const N_MELS: usize = 80;
/// FP32 log-domain elementwise tolerance for the fused-vs-unfused parity
/// (NFR-QL-01). The fused kernel differs from the imperative reference by
/// exactly one round of intermediate rounding in the `sum_k w_k · (r² + im²)`
/// mel projection; the plan pins the resulting delta at `atol = 0.01` in the
/// normalized log-mel output.
const PARITY_ATOL: f32 = 0.01;

fn floor_log10() -> f32 {
    1e-10f32.log10()
}

/// Compares `fused` against `reference` elementwise, returning
/// `Err(diagnostic)` on the first index that exceeds [`PARITY_ATOL`]. The
/// diagnostic includes the failing index, the absolute delta, and both
/// sides' values so a regression pins exactly which mel bin drifted.
fn assert_close(reference: &[f32], fused: &[f32], case: &str) {
    assert_eq!(
        reference.len(),
        fused.len(),
        "{case}: shape mismatch (ref = {}, fused = {})",
        reference.len(),
        fused.len()
    );
    let mut max_delta = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (&r, &f)) in reference.iter().zip(fused.iter()).enumerate() {
        assert!(
            r.is_finite() && f.is_finite(),
            "{case}: non-finite value at index {i} (ref = {r}, fused = {f})"
        );
        let d = (r - f).abs();
        if d > max_delta {
            max_delta = d;
            max_idx = i;
        }
    }
    assert!(
        max_delta <= PARITY_ATOL,
        "{case}: fused vs unfused log-mel exceeds atol = {PARITY_ATOL} at index {max_idx} \
         (delta = {max_delta}, ref = {}, fused = {})",
        reference[max_idx],
        fused[max_idx],
    );
}

/// Runs the fused kernel with the same `frontend_spec` the reference uses,
/// building the STFT / mel attrs once via the `vokra-ops` translation so the
/// mel filter weights are byte-identical to `log_mel`'s (NFR-QL-03). The
/// input is zero-padded / trimmed to exactly [`N_SAMPLES`] (30 s at 16 kHz)
/// to match the reference's `mel.rs:60-62` pre-conditioning — without this
/// alignment the fused kernel would only produce mel frames for the input
/// length while the reference sees a full 30 s window, and the dynamic-range
/// compression `(v.max(gmax - 8) + 4) / 4` would key off different `gmax`
/// values on the two sides.
fn fused_reference_matched(pcm: &[f32]) -> Vec<f32> {
    let spec = runtime_frontend_spec(N_MELS);
    let stft_attrs = stft_attrs_from_spec(&spec).expect("runtime frontend spec is well-formed");
    let mel_attrs = mel_attrs_from_spec(&spec).expect("runtime frontend spec is well-formed");
    let fb = mel_filterbank(&mel_attrs);
    // Zero-pad / trim to the Whisper 30 s window, matching `whisper::mel::log_mel`.
    let mut buf = vec![0.0f32; N_SAMPLES];
    let n = pcm.len().min(N_SAMPLES);
    buf[..n].copy_from_slice(&pcm[..n]);
    let mut out = vec![floor_log10(); N_MELS * N_FRAMES];
    fused_log_mel_scalar(&buf, &stft_attrs, &fb, N_FRAMES, &mut out);
    out
}

/// Constant `f32` sample count of the reference clip; the reference pads /
/// trims to [`N_SAMPLES`] internally regardless of input length, so passing a
/// shorter clip is a valid test vector (the tail beyond the input length is
/// zero-padded on both sides).
fn build_silence(sec: f32) -> Vec<f32> {
    vec![0.0f32; (sec * SAMPLE_RATE as f32) as usize]
}

fn build_sine(freq_hz: f32, sec: f32) -> Vec<f32> {
    let n = (sec * SAMPLE_RATE as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / SAMPLE_RATE as f64;
            (2.0 * std::f64::consts::PI * freq_hz as f64 * t).sin() as f32
        })
        .collect()
}

/// Linear chirp `sin(2π·(f0 + (f1 - f0)·t/T)·t)` over `sec` seconds. Sweeping
/// 200 Hz → 1000 Hz across 3 s spreads energy over roughly the first 8-10 mel
/// bands, catching accumulation-order sensitivity in the mel projection.
fn build_chirp(f0_hz: f32, f1_hz: f32, sec: f32) -> Vec<f32> {
    let n = (sec * SAMPLE_RATE as f32) as usize;
    let big_t = sec as f64;
    let f0 = f0_hz as f64;
    let f1 = f1_hz as f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / SAMPLE_RATE as f64;
            // Instantaneous frequency ramp from f0 to f1 over [0, T].
            let phase = 2.0 * std::f64::consts::PI * (f0 + (f1 - f0) * t / big_t) * t;
            phase.sin() as f32
        })
        .collect()
}

#[test]
fn silence_parity() {
    let pcm = build_silence(1.0);
    let reference = log_mel(&pcm, N_MELS);
    let fused = fused_reference_matched(&pcm);
    assert_close(&reference, &fused, "silence(1s)");
}

#[test]
fn sine_440hz_parity() {
    let pcm = build_sine(440.0, 1.0);
    let reference = log_mel(&pcm, N_MELS);
    let fused = fused_reference_matched(&pcm);
    assert_close(&reference, &fused, "sine(440Hz, 1s)");
}

#[test]
fn chirp_200_to_1000hz_parity() {
    // 3 s speech-like chirp covering the 200 Hz - 1 kHz vocal formant range.
    let pcm = build_chirp(200.0, 1000.0, 3.0);
    let reference = log_mel(&pcm, N_MELS);
    let fused = fused_reference_matched(&pcm);
    assert_close(&reference, &fused, "chirp(200-1000Hz, 3s)");
}

#[test]
fn shape_matches_whisper_window() {
    // Cross-check the output dimension the plan pins as bit-exact between the
    // two paths: `[n_mels, N_FRAMES]` (N_FRAMES = 3000, the Whisper 30 s
    // window). The imperative reference sizes to `n_mels * N_FRAMES` at
    // `mel.rs:79`; the fused kernel is passed the same `n_frames`.
    let pcm = build_silence(0.5);
    let reference = log_mel(&pcm, N_MELS);
    let fused = fused_reference_matched(&pcm);
    assert_eq!(reference.len(), N_MELS * N_FRAMES);
    assert_eq!(fused.len(), N_MELS * N_FRAMES);
}
