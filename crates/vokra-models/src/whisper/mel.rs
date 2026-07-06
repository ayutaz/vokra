//! Whisper log-mel front-end (PCM → `[n_mels, 3000]` log-mel features).
//!
//! Reuses the M0-04 `vokra-ops` STFT and mel filter bank (FR-OP-01/03) and
//! reproduces the openai/whisper `log_mel_spectrogram` post-processing exactly
//! (HF `WhisperFeatureExtractor` matches it):
//!
//! 1. zero-pad (or trim) the mono PCM to exactly 30 s = [`N_SAMPLES`] samples;
//! 2. STFT with `n_fft = 400`, `hop = 160`, periodic Hann, `center = true`
//!    reflect padding, no FFT normalization (raw); take the power `|X|²` and
//!    **drop the last STFT frame**, leaving [`N_FRAMES`] = 3000 frames;
//! 3. project onto `n_mels` Slaney-scale / Slaney-norm mel bands;
//! 4. `log10(clamp(·, 1e-10))`, then dynamic-range compress to the global
//!    max minus 8, then `(· + 4) / 4`.
//!
//! Parameters here come from Whisper's fixed front-end (`n_fft`, `hop`,
//! `sample_rate`, Slaney mel) — the same values the converter writes into
//! `vokra.frontend.*`.
//!
//! # Data-driven front-end + bit-exact check (M1-03)
//!
//! The knobs are declared **once** in [`runtime_frontend_spec`] (as a
//! [`FrontendSpec`]); [`log_mel`] derives its `StftAttrs` / `MelAttrs` from that
//! spec through the `vokra-ops` translation rather than hard-coding
//! `400 / 160 / Slaney` a second time (NFR-QL-03). At model load,
//! [`check_frontend_spec`] compares the GGUF's declared `vokra.frontend.*`
//! chunk against that same runtime spec bit-for-bit and warns / fails on a
//! mismatch (FR-LD-03). STFT ≠ FFT — every knob is explicit, per the CLAUDE.md
//! pitfall.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use vokra_core::gguf::GgufFile;
use vokra_core::{FrontendPolicy, FrontendSpec, Result};
use vokra_ops::{
    fused_log_mel_scalar, mel_attrs_from_spec, mel_filterbank, stft, stft_attrs_from_spec,
};

/// Runtime toggle for the fused log-mel path (M2-04-T08).
///
/// Defaults to **on**. The `VOKRA_DISABLE_FUSION=1` environment variable
/// overrides this to off, but is read **exactly once** on the first call to
/// [`log_mel`] and then locked for the process lifetime — this keeps the toggle
/// state bit-stable across the 3001-frame run and avoids per-call env lookups
/// on the hot path.
///
/// When disabled, [`log_mel`] falls through to the physically-unchanged
/// imperative `stft` → `power` → `MelFilterbank::apply` → `log10` → transpose
/// path that predates M2-04. This preserves the bit-identical unfused
/// reference for parity (T07) and bench A/B (T11) — see the M2-04 fusion pass
/// ADR at `crates/vokra-core/src/ir/fusion/mod.rs` for the two-face design.
mod fusion {
    use super::*;

    /// Env-lock state machine: 0 = not yet read, 1 = enabled, 2 = disabled.
    /// Using `u8` (not `bool`) lets us distinguish "not yet resolved" from
    /// "resolved to enabled", which matters for the once-and-lock semantics
    /// (once resolved, later env mutation is ignored — see `is_enabled`).
    static ENV_STATE: AtomicU8 = AtomicU8::new(0);

    /// Optional test-only override: when `Some(bool)`, wins over the env
    /// resolution and is *not* locked by the env-once machinery. Tests that
    /// need to toggle both states within a single process must go through
    /// [`set_enabled_for_test`] rather than mutating `VOKRA_DISABLE_FUSION`.
    static TEST_OVERRIDE_SET: AtomicBool = AtomicBool::new(false);
    static TEST_OVERRIDE_VAL: AtomicBool = AtomicBool::new(true);

    /// Reads the fused-path toggle, resolving `VOKRA_DISABLE_FUSION` on the
    /// first uncached call and locking the result for the process lifetime
    /// (safe: env is only sampled once, so no race between reads and later
    /// `set_var`/`remove_var` in other threads).
    pub(super) fn is_enabled() -> bool {
        // Test override wins if set (used by the T08 snapshot test to exercise
        // both toggle states without racing the once-lock).
        if TEST_OVERRIDE_SET.load(Ordering::Acquire) {
            return TEST_OVERRIDE_VAL.load(Ordering::Acquire);
        }
        let state = ENV_STATE.load(Ordering::Acquire);
        if state != 0 {
            return state == 1;
        }
        // Resolve the env exactly once; the CAS makes the "first reader
        // decides" race benign (all readers converge on the same value).
        let disabled = std::env::var("VOKRA_DISABLE_FUSION")
            .map(|v| v == "1")
            .unwrap_or(false);
        let new_state: u8 = if disabled { 2 } else { 1 };
        // If another thread already resolved, its value stands.
        let _ = ENV_STATE.compare_exchange(0, new_state, Ordering::AcqRel, Ordering::Acquire);
        ENV_STATE.load(Ordering::Acquire) == 1
    }

    /// Test-only toggle override. Bypasses the once-lock so both branches of
    /// [`log_mel`] can be exercised within a single test process.
    #[cfg(test)]
    pub(super) fn set_enabled_for_test(enabled: bool) {
        TEST_OVERRIDE_VAL.store(enabled, Ordering::Release);
        TEST_OVERRIDE_SET.store(true, Ordering::Release);
    }

    /// Clears the test-only override (returns to env-driven resolution).
    #[cfg(test)]
    pub(super) fn clear_test_override() {
        TEST_OVERRIDE_SET.store(false, Ordering::Release);
    }
}

/// Model sample rate in Hz (Whisper is 16 kHz).
pub const SAMPLE_RATE: u32 = 16_000;
/// STFT window / FFT size.
pub const N_FFT: usize = 400;
/// STFT hop length.
pub const HOP: usize = 160;
/// Fixed input length: 30 s at 16 kHz.
pub const N_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
/// Number of log-mel frames after dropping the trailing STFT frame.
pub const N_FRAMES: usize = 3000;

/// Computes the `[n_mels, N_FRAMES]` (row-major) log-mel features of mono
/// `pcm`, assumed to already be at [`SAMPLE_RATE`].
///
/// The input is zero-padded or trimmed to 30 s, so the output frame count is
/// always [`N_FRAMES`] regardless of the input length.
pub fn log_mel(pcm: &[f32], n_mels: usize) -> Vec<f32> {
    // Data-driven front-end (M1-03): build the STFT / mel attributes from the
    // one runtime spec via the vokra-ops translation, instead of hard-coding
    // `400 / 160 / Slaney` a second time here (NFR-QL-03). `runtime_frontend_spec`
    // is well-formed by construction, so the translation cannot fail.
    let spec = runtime_frontend_spec(n_mels);
    let stft_attrs = stft_attrs_from_spec(&spec).expect("runtime frontend spec is well-formed");
    let mel_attrs = mel_attrs_from_spec(&spec).expect("runtime frontend spec is well-formed");

    // 1. Pad / trim to exactly 30 s. Both paths pad to `N_SAMPLES` so the
    //    STFT frame grid is fixed and independent of `pcm.len()`.
    let mut buf = vec![0.0f32; N_SAMPLES];
    let n = pcm.len().min(N_SAMPLES);
    buf[..n].copy_from_slice(&pcm[..n]);

    // M2-04-T08: route through the fused single-pass kernel when enabled
    // (default). Falls through to the pre-fusion imperative path when the
    // toggle is off — that branch is kept physically identical to the
    // pre-M2-04 code so it remains the bit-identical unfused reference for
    // parity (T07) and bench A/B (T11). See the M2-04 ADR at
    // `crates/vokra-core/src/ir/fusion/mod.rs` for the two-face design.
    if fusion::is_enabled() {
        let fb = mel_filterbank(&mel_attrs);
        let floor_log = 1e-10f32.log10();
        let mut out = vec![floor_log; n_mels * N_FRAMES];
        fused_log_mel_scalar(&buf, &stft_attrs, &fb, N_FRAMES, &mut out);
        return out;
    }

    // --- Unfused reference path (bit-identical to the pre-M2-04 code) --------

    // 2. STFT → power, drop the trailing frame.
    let stft_out = stft(&buf, &stft_attrs).expect("valid whisper STFT attrs");
    let bins = stft_out.bins; // n_fft/2 + 1 = 201
    let frames = stft_out.frames.min(N_FRAMES + 1);
    let kept = frames.min(N_FRAMES);
    let power = stft_out.power();

    // 3. Mel projection on the kept frames → [kept, n_mels].
    let fb = mel_filterbank(&mel_attrs);
    let mel = fb.apply(&power[..kept * bins], kept);

    // 4. log10 + dynamic-range compression + normalization, transposed to
    //    [n_mels, N_FRAMES]. Frames beyond `kept` (only possible for absurdly
    //    short inputs) stay at the log floor.
    let floor_log = 1e-10f32.log10();
    let mut out = vec![floor_log; n_mels * N_FRAMES];
    let mut gmax = f32::NEG_INFINITY;
    for t in 0..kept {
        for m in 0..n_mels {
            let l = mel[t * n_mels + m].max(1e-10).log10();
            out[m * N_FRAMES + t] = l;
            if l > gmax {
                gmax = l;
            }
        }
    }
    let dyn_floor = gmax - 8.0;
    for v in &mut out {
        *v = (v.max(dyn_floor) + 4.0) / 4.0;
    }
    out
}

/// The front-end parameters [`log_mel`] actually computes, as a
/// [`FrontendSpec`] (FR-LD-03, M1-03).
///
/// This is the **runtime side** of the bit-exact `vokra.frontend.*` check: it
/// must stay identical to what the offline converter writes into the GGUF
/// (`vokra-convert`'s `whisper::frontend_spec()`), and the two are cross-checked
/// at load by [`check_frontend_spec`]. Every value is Whisper's fixed front-end
/// (openai/whisper `whisper/audio.py`): `n_fft = 400`, `hop = 160`, periodic
/// Hann, reflect padding, Slaney-scale / Slaney-norm mel over `[0, sr/2]`, no DC
/// removal, no pre-emphasis, 16 kHz. Only `n_mels` varies with the model.
pub fn runtime_frontend_spec(n_mels: usize) -> FrontendSpec {
    FrontendSpec {
        n_fft: N_FFT as u32,
        hop: HOP as u32,
        win_length: N_FFT as u32,
        window_type: "hann".to_owned(),
        mel_norm: "slaney".to_owned(),
        htk_mode: false,
        fmin: 0.0,
        fmax: SAMPLE_RATE as f32 / 2.0,
        n_mels: n_mels as u32,
        pad_mode: "reflect".to_owned(),
        dc_offset_removal: false,
        pre_emphasis: 0.0,
        sample_rate: SAMPLE_RATE,
    }
}

/// Reads the model's `vokra.frontend.*` chunk and checks it against the runtime
/// Whisper front-end bit-for-bit under `policy` (FR-LD-03, M1-03).
///
/// Whisper owns an STFT / mel front-end, so its GGUF **always** declares the
/// chunk: a missing key is therefore a load error (surfaced from
/// [`FrontendSpec::from_gguf`]). Models whose front-end Vokra does not control
/// (Silero VAD, piper-plus) write no such chunk and their loaders never call
/// this function — the per-model gating is by *caller*, not a global pass.
///
/// # Errors
///
/// [`VokraError::ModelLoad`](vokra_core::VokraError) if the chunk is absent or
/// malformed; [`VokraError::FrontendMismatch`](vokra_core::VokraError) under
/// [`FrontendPolicy::Fail`] if any field differs from [`runtime_frontend_spec`].
pub fn check_frontend_spec(file: &GgufFile, n_mels: usize, policy: FrontendPolicy) -> Result<()> {
    let model_spec = FrontendSpec::from_gguf(file)?;
    model_spec.check_against(&runtime_frontend_spec(n_mels), policy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_shape_is_n_mels_by_n_frames() {
        let pcm = vec![0.0f32; SAMPLE_RATE as usize]; // 1 s of silence
        let out = log_mel(&pcm, 80);
        assert_eq!(out.len(), 80 * N_FRAMES);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_tone_produces_finite_bounded_features() {
        // 440 Hz tone; log-mel is normalized to a bounded range by construction.
        let pcm: Vec<f32> = (0..SAMPLE_RATE as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let out = log_mel(&pcm, 80);
        // After (x+4)/4 with x in [max-8, max] and Whisper's normalization the
        // features sit within roughly [-1, 1]; assert a generous finite bound.
        assert!(out.iter().all(|v| v.is_finite() && *v > -2.0 && *v < 2.0));
    }

    #[test]
    fn empty_input_pads_to_full_frame_grid() {
        // The pad branch with a zero-length slice: buf stays all-zero, so the
        // output is still the full [n_mels, N_FRAMES] grid with finite values.
        let out = log_mel(&[], 80);
        assert_eq!(out.len(), 80 * N_FRAMES);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn oversized_input_is_trimmed_to_full_frame_grid() {
        // Longer than 30 s exercises the trim branch (pcm.len() > N_SAMPLES); the
        // frame count is still fixed and every value stays finite.
        let pcm = vec![0.1f32; N_SAMPLES + 5000];
        let out = log_mel(&pcm, 80);
        assert_eq!(out.len(), 80 * N_FRAMES);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // --- M2-04-T08: fusion toggle behaviour ---------------------------------

    /// Serialises the fusion-toggle tests so they don't race the shared
    /// [`super::fusion`] state machine (both the env-once lock and the
    /// test-override cell live at module scope).
    static FUSION_TOGGLE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The `fusion::is_enabled()` toggle must return the value we asked for and
    /// stay stable across a run — otherwise the two-branch `log_mel` would race
    /// the once-lock and pick the wrong path mid-way.
    #[test]
    fn fusion_toggle_reflects_test_override() {
        let _guard = FUSION_TOGGLE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        fusion::set_enabled_for_test(true);
        assert!(fusion::is_enabled());
        fusion::set_enabled_for_test(false);
        assert!(!fusion::is_enabled());
        fusion::clear_test_override();
    }

    /// With the fused path disabled, `log_mel` must produce values that are
    /// element-wise bit-identical to the pre-M2-04 unfused implementation.
    /// The reference here is the same imperative chain (kept physically
    /// unchanged inside the `else` arm), so we recompute it locally with the
    /// same input and assert exact equality — this pins the "toggle off = no
    /// behaviour change" invariant that the T07 parity test and the T11 A/B
    /// bench baseline both depend on.
    #[test]
    fn toggle_off_is_bit_identical_to_pre_fusion_baseline() {
        let _guard = FUSION_TOGGLE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Small deterministic input: a 100 ms tone at 440 Hz. Short enough to
        // stay well below `N_SAMPLES` (so we exercise the pad branch, not the
        // trim branch) and non-silent so the mel bands see real energy.
        let n = SAMPLE_RATE as usize / 10;
        let pcm: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();

        // Reference: the physically-preserved unfused branch inside `log_mel`.
        fusion::set_enabled_for_test(false);
        let reference = log_mel(&pcm, 80);

        // Reproduce the pre-fusion baseline **out of band** (independent
        // recomputation) so a silent regression in the `else` arm can't hide
        // behind a comparison against itself.
        let spec = runtime_frontend_spec(80);
        let stft_attrs = stft_attrs_from_spec(&spec).unwrap();
        let mel_attrs = mel_attrs_from_spec(&spec).unwrap();
        let mut buf = vec![0.0f32; N_SAMPLES];
        buf[..pcm.len()].copy_from_slice(&pcm);
        let stft_out = stft(&buf, &stft_attrs).unwrap();
        let bins = stft_out.bins;
        let frames = stft_out.frames.min(N_FRAMES + 1);
        let kept = frames.min(N_FRAMES);
        let power = stft_out.power();
        let fb = mel_filterbank(&mel_attrs);
        let mel = fb.apply(&power[..kept * bins], kept);
        let floor_log = 1e-10f32.log10();
        let mut baseline = vec![floor_log; 80 * N_FRAMES];
        let mut gmax = f32::NEG_INFINITY;
        for t in 0..kept {
            for m in 0..80 {
                let l = mel[t * 80 + m].max(1e-10).log10();
                baseline[m * N_FRAMES + t] = l;
                if l > gmax {
                    gmax = l;
                }
            }
        }
        let dyn_floor = gmax - 8.0;
        for v in &mut baseline {
            *v = (v.max(dyn_floor) + 4.0) / 4.0;
        }

        assert_eq!(reference.len(), baseline.len());
        for (i, (a, b)) in reference.iter().zip(baseline.iter()).enumerate() {
            // Bit-identical: same input → same allocations → same float ops
            // → same rounding. If this ever drifts, the `else` arm has been
            // touched — that would break the T07 parity oracle.
            assert_eq!(a.to_bits(), b.to_bits(), "drift at index {i}: {a} vs {b}");
        }
        fusion::clear_test_override();
    }

    /// With the fused path enabled the shape / finiteness contract must still
    /// hold, and the values must remain bit-close to the unfused reference
    /// (within the T07 atol = 0.01). This is the T08 hookup smoke check; the
    /// full three-input parity fixture lives in `vokra-ops` (T07).
    #[test]
    fn fused_path_matches_unfused_within_atol() {
        let _guard = FUSION_TOGGLE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let n = SAMPLE_RATE as usize / 10;
        let pcm: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();

        fusion::set_enabled_for_test(false);
        let unfused = log_mel(&pcm, 80);
        fusion::set_enabled_for_test(true);
        let fused = log_mel(&pcm, 80);

        assert_eq!(fused.len(), 80 * N_FRAMES);
        assert!(fused.iter().all(|v| v.is_finite()));
        for (i, (f, u)) in fused.iter().zip(unfused.iter()).enumerate() {
            assert!(
                (f - u).abs() < 0.01,
                "fused vs unfused delta > 0.01 at index {i}: {f} vs {u}"
            );
        }
        fusion::clear_test_override();
    }

    // --- frontend_spec check (M1-03) -----------------------------------------

    use vokra_core::VokraError;
    use vokra_core::gguf::GgufBuilder;

    /// A GGUF carrying `spec` as its only metadata (no tensors needed for the
    /// front-end check).
    fn gguf_with_spec(spec: &FrontendSpec) -> GgufFile {
        let mut b = GgufBuilder::new();
        spec.write_into(&mut b);
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn runtime_spec_pins_the_whisper_front_end() {
        // Drift guard: these are exactly the values `vokra-convert`'s
        // `whisper::frontend_spec()` writes into the GGUF (kept in sync so the
        // load-time check can never spuriously fire on a correctly converted
        // model). n_mels = 80 for whisper-base.
        let s = runtime_frontend_spec(80);
        assert_eq!(s.n_fft, 400);
        assert_eq!(s.hop, 160);
        assert_eq!(s.win_length, 400);
        assert_eq!(s.window_type, "hann");
        assert_eq!(s.mel_norm, "slaney");
        assert!(!s.htk_mode);
        assert_eq!(s.fmin, 0.0);
        assert_eq!(s.fmax, 8000.0);
        assert_eq!(s.n_mels, 80);
        assert_eq!(s.pad_mode, "reflect");
        assert!(!s.dc_offset_removal);
        assert_eq!(s.pre_emphasis, 0.0);
        assert_eq!(s.sample_rate, 16_000);
    }

    #[test]
    fn matching_chunk_passes_the_check() {
        let file = gguf_with_spec(&runtime_frontend_spec(80));
        assert!(check_frontend_spec(&file, 80, FrontendPolicy::Fail).is_ok());
        assert!(check_frontend_spec(&file, 80, FrontendPolicy::Warn).is_ok());
    }

    #[test]
    fn mismatched_chunk_fails_under_fail_and_warns_under_warn() {
        // A GGUF whose front-end declares HTK where the runtime computes Slaney.
        let mut declared = runtime_frontend_spec(80);
        declared.htk_mode = true;
        let file = gguf_with_spec(&declared);

        match check_frontend_spec(&file, 80, FrontendPolicy::Fail) {
            Err(VokraError::FrontendMismatch(msg)) => assert!(msg.contains("htk_mode"), "{msg}"),
            other => panic!("expected FrontendMismatch, got {other:?}"),
        }
        // Warn tolerates it (report goes to stderr).
        assert!(check_frontend_spec(&file, 80, FrontendPolicy::Warn).is_ok());
    }

    #[test]
    fn n_mels_mismatch_between_config_and_chunk_is_caught() {
        // The chunk was written for 80 mels but the config says 128: the check
        // is parameterised on the config's n_mels, so this is a real mismatch.
        let file = gguf_with_spec(&runtime_frontend_spec(80));
        assert!(matches!(
            check_frontend_spec(&file, 128, FrontendPolicy::Fail),
            Err(VokraError::FrontendMismatch(_))
        ));
    }

    #[test]
    fn whisper_requires_the_chunk_missing_is_a_load_error() {
        // Per-model conditional: Whisper owns a front-end, so a GGUF with no
        // `vokra.frontend.*` keys is rejected at the check (surfaced as a
        // ModelLoad from FrontendSpec::from_gguf). VAD / piper-plus never reach
        // here because their loaders do not call check_frontend_spec at all.
        let empty = GgufFile::parse(GgufBuilder::new().to_bytes().unwrap()).unwrap();
        assert!(matches!(
            check_frontend_spec(&empty, 80, FrontendPolicy::Fail),
            Err(VokraError::ModelLoad(_))
        ));
    }
}
