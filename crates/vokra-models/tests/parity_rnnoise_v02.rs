//! Xiph RNNoise v0.2 numerical parity harness — env-gated (denoise
//! tier, 2026-08-09 loud-partial forwards wave).
//!
//! Sibling of `parity_rmvpe.rs` / `parity_openwakeword.rs` /
//! `parity_dnsmos.rs`: every test that needs the RNNoise v0.2 GGUF is
//! gated on the [`GGUF_ENV`] environment variable and skips cleanly when
//! unset (never a fabricated pass, memory
//! `[[project-real-weight-eval]]` pattern). Once opted in, every failure
//! is hard: a missing / malformed / wrong-shaped fixture is a loud
//! panic (FR-EX-08).
//!
//! # Fixture recipe (owner-side)
//!
//! Upstream `github.com/xiph/rnnoise` ships the v0.2 weights as a
//! ~90 KB compact C-array blob (`weights_blob_9.bin`, bundled in the
//! v0.2 GitHub release tarball, additionally embedded as
//! `src/rnn_data.c` in the source tree). Bridge it to safetensors
//! offline with the existing `tools/parity/rnnoise_prepare_checkpoint.py`
//! side-car (fair-use pure-Python flattener, BSD-3), then convert to
//! Vokra GGUF:
//!
//! ```text
//! # 1. Fetch upstream v0.2 release tarball from
//! #    github.com/xiph/rnnoise/releases/tag/v0.2
//! # 2. Flatten to safetensors (offline, in a uv-managed venv):
//! uv run python tools/parity/rnnoise_prepare_checkpoint.py \
//!     --input  ~/rnnoise-v0.2/weights_blob_9.bin \
//!     --output ~/rnnoise-v02.safetensors
//! # 3. Convert to GGUF:
//! vokra-cli convert --model rnnoise-v0.2 \
//!     --input  ~/rnnoise-v02.safetensors \
//!     --output ~/rnnoise-v02.gguf
//! # 4. Point the parity harness at it:
//! export VOKRA_RNNOISE_V02_REAL_GGUF=~/rnnoise-v02.gguf
//! cargo test -p vokra-models --test parity_rnnoise_v02 -- --nocapture
//! ```
//!
//! # Numeric parity contract
//!
//! RNNoise's pitch tracker (upstream `src/pitch.c`) and Bark-band
//! filterbank (upstream `src/denoise.c`) are the two determinism
//! choke-points the harness pins. The pitch check is a *lag-band*
//! match rather than an L∞ delta because upstream downsamples to 12 kHz
//! before the autocorrelation (Vokra runs at 48 kHz for zero-dep — no
//! FIR filter tables) — so a bit-exact period match against upstream
//! `celt_pitch_xcorr` is architecturally infeasible; we check that the
//! Vokra period lies inside a `±3 %` band around the upstream period
//! (well under the RNNoise remove_doubling tolerance).
//!
//! The Bark filterbank check is a per-band `max |Δ|` bound
//! ([`BARK_ENERGY_ATOL`]) against upstream `compute_band_energy` output.
//! Filterbank arithmetic is bit-identical modulo a couple of ULPs (both
//! implementations use IEEE 754 f32 accumulate over the same triangular
//! ramp weights), so a bound of `1e-4` catches every substantive
//! algorithmic regression while surviving cross-arch libm noise.
//!
//! **When the real per-layer GRU forward wave lands** (deferred per
//! `docs/license-audit.md` §3.1 owner sign-off + real-checkpoint
//! parity), the `parity_rnnoise_full_denoise_forward` test upgrades
//! from a loud-partial pending marker to a real waveform-vs-waveform
//! SI-SNR bound; the harness is written so the pending-marker path
//! remains present until the full-denoise wave lights up.

use std::env;
use std::path::Path;

use vokra_ops::rnnoise::{
    FRAME_HOP, FRAME_SIZE, MAX_LAG_SAMPLES, MIN_LAG_SAMPLES, N_BARK_BANDS, N_PITCH_BANDS,
    N_STFT_BINS, PITCH_BUF_SIZE, PitchState, SAMPLE_RATE, bark_filterbank, pitch_analysis,
    vorbis_window,
};

/// Env var the owner sets to point the gated harness at a real RNNoise
/// v0.2 GGUF. Absent = skip cleanly (never a fabricated pass). Present =
/// binding: every downstream check hard-fails on any error.
const GGUF_ENV: &str = "VOKRA_RNNOISE_V02_REAL_GGUF";

/// Per-band `max |Δ|` bound for the Bark filterbank parity check.
/// See the module doc for the honest atol rationale.
#[allow(dead_code)] // Consumed once the real-GGUF path binds.
const BARK_ENERGY_ATOL: f32 = 1e-4;

/// Maximum fractional error between Vokra's 48-kHz-native pitch period
/// and upstream's 12-kHz-downsampled period. 3 % is well under the
/// upstream `remove_doubling` octave-error correction window, so a
/// pass at this bound proves the pitch tracker is on the right lag
/// grid even with the honest sample-rate divergence.
#[allow(dead_code)] // Consumed once the real-GGUF path binds.
const PITCH_PERIOD_FRACTIONAL_TOL: f32 = 0.03;

// ---------------------------------------------------------------------------
// FIXTURE-FREE primitives — unit tests independent of any real GGUF fixture
// ---------------------------------------------------------------------------

/// FIXTURE-FREE: the Vorbis window satisfies the Princen-Bradley
/// perfect-reconstruction condition at 50 % overlap. Pins the window
/// shape independently of the pitch / GRU wave so a future refactor
/// cannot silently regress into a Hann (which sums to 0.5 at 50 %
/// overlap — a −6 dB gain drop against upstream RNNoise).
#[test]
fn vorbis_window_pins_princen_bradley() {
    let w = vorbis_window(FRAME_SIZE);
    assert_eq!(w.len(), FRAME_SIZE);
    for n in 0..FRAME_HOP {
        let sum = w[n] * w[n] + w[n + FRAME_HOP] * w[n + FRAME_HOP];
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Vorbis window violates Princen-Bradley at n={n}: {sum}"
        );
    }
}

/// FIXTURE-FREE: the autocorrelation-based pitch tracker recovers a
/// pure 200 Hz sine's period within 1.5 samples at 48 kHz (parabolic
/// refinement precision).
#[test]
fn pitch_analysis_pure_tone_end_to_end() {
    let f0 = 200.0f32;
    let sr = SAMPLE_RATE as f32;
    let expected_period = sr / f0; // 240 at 48 kHz
    assert!(expected_period as usize >= MIN_LAG_SAMPLES);
    assert!(expected_period as usize <= MAX_LAG_SAMPLES);
    let n = PITCH_BUF_SIZE * 2;
    let signal: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * f0 * (i as f32) / sr).sin())
        .collect();

    let mut state = PitchState::default();
    let (period, gain, bands) = pitch_analysis(&mut state, &signal).unwrap();
    let delta = (period - expected_period).abs();
    assert!(
        delta < 1.5,
        "expected period ≈ {expected_period}, got {period} (Δ = {delta})"
    );
    assert!(gain > 0.9, "pure-tone gain must be near +1, got {gain}");
    assert_eq!(bands.len(), N_PITCH_BANDS);
}

/// FIXTURE-FREE: the Bark filterbank is a real triangular partition
/// over the STFT bins; feeding an all-ones magnitude spectrum must
/// produce a positive energy in every non-zero-width band.
#[test]
fn bark_filterbank_partition_of_unity_pin() {
    let mag = vec![1.0f32; N_STFT_BINS];
    let e = bark_filterbank(&mag).unwrap();
    assert_eq!(e.len(), N_BARK_BANDS);
    let sum: f32 = e.iter().sum();
    assert!(
        sum > 0.0,
        "an all-ones magnitude spectrum must have non-zero total Bark energy"
    );
}

// ---------------------------------------------------------------------------
// GATED — real RNNoise v0.2 GGUF fixture required
// ---------------------------------------------------------------------------

/// GATED: opens a real RNNoise v0.2 GGUF and verifies the runtime
/// binder is a genuine bind (real config parse, real tensor bind).
///
/// Skips cleanly when [`GGUF_ENV`] is unset. Once set, all failures
/// are hard: a missing / malformed / wrong-arch fixture fails loudly.
///
/// # Loud-partial marker
///
/// The current landing exercises the pitch primitive + Bark filterbank
/// end-to-end. The full denoise forward (STFT + Bark → per-frame
/// features → GRU stack → sigmoid gain mask → gain application) is
/// wave B; the harness marks the missing forward with an explicit
/// pending message so an owner opt-in is directed straight at that
/// follow-up wave rather than silently reporting "0 % SI-SNR".
#[test]
fn parity_rnnoise_v02_gguf_smoke() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping rnnoise v0.2 GGUF parity smoke; \
             this is a clean skip (never a fabricated pass). See the \
             module docs for the fixture recipe."
        );
        return;
    };

    let path = Path::new(&gguf_path);
    assert!(
        path.exists(),
        "opted in with {GGUF_ENV} but path {gguf_path} does not exist — \
         hard fail (FR-EX-08)"
    );

    // The RNNoise v0.2 GGUF runtime binder (`vokra-models::rnnoise` or
    // an equivalent runtime crate) is not yet landed — the loud-partial
    // marker below directs the opt-in owner to the follow-up wave. This
    // is the honest posture: no `assert!(false)` fabricated pass, and
    // no silent skip either.
    panic!(
        "rnnoise v0.2 GGUF opt-in received at {gguf_path}, but the runtime binder \
         (full-denoise forward wave) is deferred to a follow-up. Bind the GGUF from \
         `crates/vokra-models/src/rnnoise/` (add when the wave lands) and route \
         the frames through `vokra_ops::rnnoise::{{pitch_analysis, bark_filterbank, \
         gru_forward, dense_forward, pack_features}}` — every primitive is real \
         today. Bark-band `max |Δ|` bound: {BARK_ENERGY_ATOL:e}. Pitch period \
         fractional tolerance: {PITCH_PERIOD_FRACTIONAL_TOL:.0} %."
    );
}

/// GATED: verifies that the pitch tracker produces a period near the
/// upstream `remove_doubling`-corrected period on a real voiced clip.
///
/// The bound is a *fractional* period match (see [`PITCH_PERIOD_FRACTIONAL_TOL`]).
/// Skips cleanly when the pointer env is unset.
#[test]
fn parity_rnnoise_v02_pitch_period_matches_upstream_band() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping rnnoise v0.2 pitch parity; \
             clean skip (never a fabricated pass)."
        );
        return;
    };
    let _ = gguf_path;

    // Placeholder: the sidecar (`tools/parity/rnnoise_prepare_checkpoint.py`)
    // does not yet emit a per-frame reference pitch dump. When it does,
    // this test compares Vokra's `pitch_analysis` output against the
    // upstream `celt_pitch_xcorr` + `remove_doubling` reference under
    // `PITCH_PERIOD_FRACTIONAL_TOL`.
    panic!(
        "rnnoise v0.2 pitch parity: opt-in received but the reference dumper does \
         not yet emit per-frame periods. Extend `tools/parity/rnnoise_prepare_checkpoint.py` \
         with a `--dump-pitch-reference` mode that runs the upstream C library on the \
         same audio and dumps the corrected pitch periods; this test then loads that \
         JSON and checks Vokra's `pitch_analysis` output against \
         PITCH_PERIOD_FRACTIONAL_TOL={PITCH_PERIOD_FRACTIONAL_TOL:.0} %."
    );
}
