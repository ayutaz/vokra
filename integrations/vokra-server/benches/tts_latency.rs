//! # T18 — Server TTS latency measurement harness (NFR-PF-05).
//!
//! Non-Criterion, non-async, `harness = false` bench that measures the
//! two latency boundaries the requirement spec calls out:
//!
//! 1. **End-to-end HTTP boundary** — the moment the `POST /api/tts`
//!    request is fully parsed to the moment the final byte of the WAV
//!    response body has been built and would be handed off to the
//!    socket writer. In-process here because the T05 axum wire layer
//!    is a thin adapter over [`vokra_server::api::piper_http::dispatch_tts`]
//!    (the "response bytes ready" instant is the return of `dispatch_tts`
//!    — after that only a `Bytes::from` + `write_all` remain, both O(N)
//!    on a pre-allocated buffer). Measuring at the schema layer removes
//!    tokio scheduler jitter and TCP loopback noise from the number, so
//!    the artifact is comparable across CI runners.
//! 2. **Time-to-first-audio (TTFA)** — the streaming equivalent used by
//!    the Wyoming path (T16). `dispatch_tts` synthesises the full utterance
//!    up-front, so the first audio byte is available at the same instant
//!    as the last — meaning TTFA ≡ end-to-end for the current (non-
//!    streaming) engine. The number is still captured so the artifact
//!    schema is stable when a real streaming vocoder lands (M2-05
//!    `istft_streaming` is already wired at the op level; the server-side
//!    generator is on the M3+ roadmap).
//!
//! ## Why `harness = false` + no external deps
//!
//! Adding `criterion` (or any test-runner crate) would pull ~40 crates
//! into the excluded-workspace `Cargo.lock` for a five-number harness.
//! `std::time::Instant::now()` has ~ns resolution on macOS/Linux/Windows
//! and is precisely what the NFR-PF-05 90 ms budget cares about
//! (millisecond-scale). We iterate N times with a warm-up prefix to
//! amortise CPU caches and expose median / p50 / p95 / max — enough to
//! catch a regression at the 90 ms threshold.
//!
//! ## GPU pre-warm
//!
//! NFR-PF-05 assumes the GPU (M2-01 Metal / M2-03 CUDA) is already warm.
//! In production this is done at [`vokra_server::run_with_config`] boot
//! (T04 `InferenceService::build` loads and touches every GGUF). This
//! harness uses [`FakeSynth`] — a deterministic in-memory PCM generator
//! — so the number reflects the *server-side* latency only (schema
//! decode → dispatch → WAV encode), independent of engine backend or
//! GGUF availability. When run against a real engine (see the "real
//! service" instructions in the T23 docs) the same [`measure_dispatch`]
//! kernel is reused; the pre-warm is caller-side and off the hot path.
//!
//! ## Reference hardware & CI artifact
//!
//! The bench prints `[[reference-hardware]]` doc-strings (M1 iMac +
//! reference desktop) plus a `[[measurement]]` JSON blob to stdout so
//! CI can capture it as a build artifact (T22 required-checks). The
//! JSON schema is intentionally minimal and stable:
//!
//! ```text
//! {"boundary":"http_end_to_end","short_utterance":"...",
//!  "iterations":100,"warmup":10,
//!  "median_us":..,"p50_us":..,"p95_us":..,"max_us":..,"budget_ms":90}
//! ```
//!
//! `budget_ms` is the NFR-PF-05 threshold echoed for downstream tooling
//! that decides whether to raise a regression issue (NFR-PF-13).
//!
//! ## What this harness does NOT do
//!
//! * It does **not** bind a network port. The measurement boundary is
//!   defined at the schema layer (see rationale above); adding a
//!   loopback TCP round-trip only injects tokio scheduler noise into a
//!   number the requirement already brackets at 90 ms.
//! * It does **not** decide pass/fail. NFR-PF-05 is a reference-hardware
//!   target; CI publishes the artifact and the four-quarter Go/No-go
//!   review (NFR-MT-05) reads it. The bench exits `0` in every non-panic
//!   case so hosted runners (which may be slower than the reference
//!   desktop) don't spuriously block main PRs.
//! * It does **not** attempt to measure the real Metal / CUDA path. That
//!   number is only meaningful on the actual reference iMac (Metal) or
//!   an RTX 4090 (CUDA); the M1 harness is expected to be run in that
//!   environment offline. The [`FakeSynth`] path is the *floor* — a
//!   real engine cannot beat it.

// SAFETY (crate-level): this bench uses only safe `std::time::Instant`
// and safe serde; no `unsafe` blocks appear below.

#![forbid(unsafe_code)]

use std::time::{Duration, Instant};

use vokra_core::{SynthesisRequest, SynthesizedAudio};
use vokra_server::api::piper_http::{TtsOutcome, TtsRequest, VoiceDefaults, dispatch_tts};
use vokra_server::service::{ServiceError, SynthesizeService, model_names};

/// NFR-PF-05 reference budget (90 ms) echoed into the JSON artifact so
/// downstream tooling reads it from one place.
const BUDGET_MS: u64 = 90;

/// Short reference utterance (matches the "reference utterance" the
/// spec talks about — kept short so wall time is dominated by request
/// decode + dispatch overhead rather than sample count).
///
/// English kept intentionally short to be representative of a chat / HA
/// query response ("Hello world"). Change with care: the artifact's
/// `short_utterance` field is used by CI comparators.
const SHORT_UTTERANCE: &str = "Hello world.";

/// Number of measurement iterations. Enough samples to make p95 stable
/// on a warm CPU but small enough to keep the bench under a second on
/// hosted CI.
const ITERATIONS: usize = 100;

/// Warm-up iterations discarded before the measurement window. Amortises
/// cold caches and any first-touch allocator work.
const WARMUP: usize = 10;

/// Deterministic in-process synth double used by both measurement
/// boundaries. Emits a small mono PCM buffer at 22 050 Hz (the piper
/// voice default) so the WAV encoder in the return path is exercised
/// on realistic-shaped data.
///
/// This is intentionally NOT a real Metal / CUDA engine: the harness is
/// measuring server overhead, not model inference (see module-doc
/// "GPU pre-warm"). When a real engine is wired in offline runs the
/// same [`measure_dispatch`] loop is reused with a live
/// `InferenceService`.
struct FakeSynth {
    /// Sample rate baked into the fake audio. Matches
    /// `vokra.piper.sample_rate` for the M0-07 reference voice.
    sample_rate: u32,
    /// Length of the emitted PCM buffer. Approximates ~250 ms of audio
    /// at 22 050 Hz — representative of "Hello world." at piper's
    /// default `length_scale`. See module-doc "short reference
    /// utterance".
    num_samples: usize,
}

impl FakeSynth {
    /// Sample rate + sample count representative of a short piper voice
    /// utterance. Kept in-code (not a config) so the artifact stays
    /// reproducible run-to-run.
    fn short_reference() -> Self {
        Self {
            sample_rate: 22_050,
            // ~250 ms of audio; big enough that the WAV encoder does real
            // work, small enough that the raw dispatch cost dominates.
            num_samples: 22_050 / 4,
        }
    }
}

impl SynthesizeService for FakeSynth {
    fn synthesize(
        &self,
        model: &str,
        _request: &SynthesisRequest,
    ) -> Result<SynthesizedAudio, ServiceError> {
        // Only `piper-plus` is on the hot path for this harness; every
        // other model surfaces `UnknownModel` verbatim so a caller that
        // asks for `kokoro` here (mis-configuration) does NOT silently
        // succeed (FR-EX-08 preserved even in the bench harness).
        if model != model_names::PIPER_PLUS {
            return Err(ServiceError::UnknownModel(model.to_owned()));
        }
        // Deterministic zero PCM: the numeric content of the buffer is
        // immaterial to latency, but a constant buffer means the WAV
        // encoder's saturating scale runs exactly as it would on real
        // audio.
        let samples = vec![0.0_f32; self.num_samples];
        Ok(SynthesizedAudio::new(samples, self.sample_rate))
    }
}

/// Single-voice defaults double: 22 050 Hz native sample rate + the
/// canonical piper `length_scale` = 1.1 / `noise_scale` = 0.667 so a
/// request that omits both overrides never trips the T11
/// `PerRequestOverrideNotImplemented` gate.
struct FakeVoices;
impl VoiceDefaults for FakeVoices {
    fn defaults_for(&self, voice: &str) -> Option<(f32, f32)> {
        if voice == "en_US-lessac-medium" {
            Some((1.1, 0.667))
        } else {
            None
        }
    }
}

/// Builds the canonical short-utterance [`TtsRequest`] the harness
/// measures. Kept in one place so both measurement boundaries feed the
/// same body through the schema layer.
fn short_request() -> TtsRequest {
    TtsRequest {
        text: SHORT_UTTERANCE.to_owned(),
        voice: "en_US-lessac-medium".to_owned(),
        model: None, // defaults to piper-plus
        length_scale: None,
        noise_scale: None,
        // cc-18: `None` keeps the voice's own language detection, so the
        // measured path is unchanged from the pre-cc-18 baseline.
        language: None,
    }
}

/// Measures one call of [`dispatch_tts`]. The returned `Duration` is
/// the wall time between the start of dispatch and the moment the WAV
/// byte buffer is ready to be handed off to the axum body writer — the
/// "final response byte sent" boundary as defined at the top of this
/// file (the byte-write cost is O(N) memcpy on a fully-formed buffer
/// and does not participate in the NFR-PF-05 budget).
fn measure_dispatch(
    service: &dyn SynthesizeService,
    voices: &dyn VoiceDefaults,
    req: &TtsRequest,
) -> Result<Duration, String> {
    let start = Instant::now();
    let outcome = dispatch_tts(service, voices, req).map_err(|e| e.to_string())?;
    let elapsed = start.elapsed();
    // Defensively touch the outcome so LLVM cannot dead-code-eliminate
    // the WAV encoding step out of the measured window.
    let bytes_len = match &outcome {
        TtsOutcome::Wav(b) => b.len(),
    };
    // Sanity: a 44-byte header + N * 2 sample bytes MUST be non-zero.
    if bytes_len < 44 {
        return Err(format!("WAV response too small: {bytes_len} bytes"));
    }
    Ok(elapsed)
}

/// Result of one boundary's measurement window.
#[derive(Debug)]
struct Stats {
    /// Human-readable boundary tag written into the JSON artifact.
    boundary: &'static str,
    /// Number of samples in the measurement window (`ITERATIONS`).
    iterations: usize,
    /// Warm-up iterations that preceded the window and were discarded.
    warmup: usize,
    /// Median of the measurement window.
    median: Duration,
    /// Same as `median` for iid samples; kept as a separate field for
    /// future streaming-vs-batch differentiation.
    p50: Duration,
    /// 95th percentile — a NFR-PF-13 regression detector should watch
    /// this rather than the median (tail latency is where the 90 ms
    /// budget is violated first).
    p95: Duration,
    /// Worst-case sample; captured for diagnostics but noisy on hosted
    /// CI, so downstream tooling should NOT gate on it.
    max: Duration,
}

/// Runs `WARMUP` throwaway iterations, then `ITERATIONS` measured ones,
/// and returns the tail statistics.
fn run_measurement(
    boundary: &'static str,
    service: &dyn SynthesizeService,
    voices: &dyn VoiceDefaults,
    req: &TtsRequest,
) -> Result<Stats, String> {
    // Warm-up: cache lines + allocator pool + branch predictors.
    for _ in 0..WARMUP {
        let _ = measure_dispatch(service, voices, req)?;
    }
    let mut samples: Vec<Duration> = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        samples.push(measure_dispatch(service, voices, req)?);
    }
    samples.sort_unstable();
    // Percentile picker matching numpy's `interpolation='lower'` (safe
    // choice for small N — always returns a real sample, never an
    // interpolated value).
    fn percentile(sorted: &[Duration], p: f64) -> Duration {
        assert!(!sorted.is_empty());
        let idx = ((sorted.len() as f64) * p).floor() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    let median = percentile(&samples, 0.50);
    let p50 = median;
    let p95 = percentile(&samples, 0.95);
    let max = *samples.last().expect("ITERATIONS > 0");
    Ok(Stats {
        boundary,
        iterations: ITERATIONS,
        warmup: WARMUP,
        median,
        p50,
        p95,
        max,
    })
}

/// Serialises [`Stats`] as a single-line JSON blob CI can grep on.
///
/// Deliberately hand-written (no `serde_json` at bench time — the whole
/// point of the excluded workspace is to keep dep-count honest). The
/// three known fields never contain characters that need JSON escaping
/// (`boundary` is a `&'static str` literal, the utterance is ASCII, the
/// numbers are integers).
fn print_measurement_json(stats: &Stats, short_utterance: &str) {
    fn us(d: Duration) -> u128 {
        d.as_micros()
    }
    // Prefixed with a stable marker so CI can `grep '^\[\[measurement\]\]'`
    // without pulling in a JSON parser.
    println!(
        "[[measurement]] {{\"boundary\":\"{boundary}\",\"short_utterance\":\"{utt}\",\
         \"iterations\":{iterations},\"warmup\":{warmup},\
         \"median_us\":{median},\"p50_us\":{p50},\"p95_us\":{p95},\"max_us\":{max},\
         \"budget_ms\":{budget}}}",
        boundary = stats.boundary,
        utt = short_utterance,
        iterations = stats.iterations,
        warmup = stats.warmup,
        median = us(stats.median),
        p50 = us(stats.p50),
        p95 = us(stats.p95),
        max = us(stats.max),
        budget = BUDGET_MS,
    );
}

/// Doc-strings the CI artifact carries alongside the measurement so a
/// human reading the log knows what runner produced it.
fn print_reference_hardware() {
    println!("[[reference-hardware]] M1 iMac (Apple Silicon, Metal backend, macOS)");
    println!("[[reference-hardware]] Reference desktop (x86_64 + RTX 4090, CUDA backend, Linux)");
    println!(
        "[[reference-hardware-note]] NFR-PF-05 budget = {BUDGET_MS} ms; this harness runs the \
         schema+WAV layer only and uses a deterministic FakeSynth (see module doc for the \
         GPU pre-warm assumption)."
    );
}

fn main() {
    print_reference_hardware();

    let service = FakeSynth::short_reference();
    let voices = FakeVoices;
    let req = short_request();

    // Boundary 1: HTTP end-to-end (request-receipt-complete →
    // final-response-byte-sent). See module-doc rationale for measuring
    // at the schema layer.
    let http = run_measurement("http_end_to_end", &service, &voices, &req)
        .expect("dispatch must not fail with FakeSynth");
    print_measurement_json(&http, SHORT_UTTERANCE);

    // Boundary 2: time-to-first-audio (streaming alt). Identical to
    // Boundary 1 today because the current engine synthesises the full
    // utterance up-front; captured for schema stability once a real
    // streaming vocoder lands (M2-05 op is done; server-side generator
    // is M3+). Reusing the same fixture keeps the two numbers directly
    // comparable — regression tooling wants (ttfa - http_end_to_end) as
    // a signal that streaming has been wired.
    let ttfa = run_measurement("time_to_first_audio", &service, &voices, &req)
        .expect("dispatch must not fail with FakeSynth");
    print_measurement_json(&ttfa, SHORT_UTTERANCE);
}
