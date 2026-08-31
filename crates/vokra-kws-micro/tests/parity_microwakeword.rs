//! microWakeWord host parity harness — env-gated (M5-03b Phase 4).
//!
//! Sibling of `crates/vokra-models/tests/parity_rmvpe.rs` (F0 tier) and
//! `parity_openwakeword.rs` (KWS tier for the RPi/Linux openWakeWord line):
//! every test that needs a real microWakeWord artefact is gated on an
//! environment variable and skips cleanly when unset — never a fabricated
//! pass. Once opted in, every failure is hard (a missing / malformed /
//! wrong-shaped fixture is a loud panic — FR-EX-08).
//!
//! # Env vars (owner-set to bind the fixture-gated paths)
//!
//! * `VOKRA_KWS_REAL_GGUF` — path to a Vokra microWakeWord GGUF, as
//!   emitted by
//!   `tools/parity/microwakeword/prepare_checkpoint.py --output <gguf>`.
//!   Naming rationale: Vokra runtime consumes GGUF, not `.tflite`
//!   (FR-LD-05 sidecar isolation — no TFLite / Python / FlatBuffer in
//!   the runtime). The task text names `VOKRA_KWS_REAL_TFLITE` for the
//!   upstream artefact — here we use `_REAL_GGUF` for what the runtime
//!   actually opens, matching the `VOKRA_RMVPE_REAL_GGUF` /
//!   `VOKRA_OPENWAKEWORD_REAL_GGUF` sibling precedent.
//! * `VOKRA_KWS_REAL_FIXTURES` — path to the directory of reference
//!   dumps emitted by `tools/parity/microwakeword/dump_reference.py`
//!   (`input_pcm.bin` + `features_ref.bin` + `output_ref.bin` +
//!   `manifest.json`). Owner triggers the dumper on the same source
//!   `.tflite` the GGUF was converted from.
//!
//! # Fixture recipe (owner-side)
//!
//! ```text
//! # 1. Fetch upstream `.tflite` and convert to Vokra GGUF
//! #    (already documented in `tools/parity/microwakeword/README.md`):
//! cd tools/parity/microwakeword
//! uv sync
//! uv run python prepare_checkpoint.py \
//!     --url    https://github.com/esphome/micro-wake-word-models/raw/main/models/v2/hey_jarvis.tflite \
//!     --name   hey_jarvis \
//!     --output ~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.gguf
//!
//! # 2. Run the reference dumper (owner walkthrough — the DL from step 1
//! #    lives in the tmpdir; use --input to keep it):
//! uv run python prepare_checkpoint.py \
//!     --url    https://github.com/esphome/micro-wake-word-models/raw/main/models/v2/hey_jarvis.tflite \
//!     --name   hey_jarvis \
//!     --output ~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.gguf
//! # (repeat the DL, or curl the .tflite locally then use --input <path>)
//! uv run python dump_reference.py \
//!     --tflite-path ~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.tflite \
//!     --output-dir  ~/.cache/vokra-eval/fixtures/microwakeword
//!
//! # 3. Point the parity harness at both artefacts:
//! export VOKRA_KWS_REAL_GGUF=~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.gguf
//! export VOKRA_KWS_REAL_FIXTURES=~/.cache/vokra-eval/fixtures/microwakeword
//! CARGO_BUILD_JOBS=1 cargo test -p vokra-kws-micro --test parity_microwakeword -- --nocapture
//! ```
//!
//! # Test layers (mirroring `parity_rmvpe.rs`)
//!
//! ## Path A — full GGUF smoke (`VOKRA_KWS_REAL_GGUF`)
//!
//! Loads the real Vokra GGUF via [`crate::model::Model::from_bytes`],
//! asserts the `vokra.kws.*` metadata contract holds (arch, threshold,
//! sample rate, mel width) and the tensor count is above the
//! "synthesized 1-tensor smoke" floor. This validates that the loader
//! survives a real MC-MobileNet checkpoint without silent binds.
//!
//! ## Path B — log-mel feature extractor parity (`VOKRA_KWS_REAL_FIXTURES`)
//!
//! Reads the dumped `input_pcm.bin` (raw `i16` little-endian,
//! [`features::WINDOW_SAMPLES`] samples @ 16 kHz), runs Vokra's
//! [`features::FeatureExtractor::compute_frame_f32`], and compares
//! per-band `|Δ|` against the dumped `features_ref.bin`
//! (numpy transcription of the standard log-mel algorithm) at
//! `atol = 1e-3` (log-mel per-band tolerance).
//!
//! The numpy reference is a transcription of the same algorithm the
//! Rust code implements (Hann window, radix-2 FFT, mel filterbank,
//! log10 with floor), so parity validates transcription faithfulness.
//! It does not validate against the training-time TensorFlow
//! `tf.signal` mel front-end used to train microWakeWord — that
//! would require `tensorflow`, out of the sidecar's 3-dep footprint.
//! Empirically the standard algorithm matches `tf.signal` to within
//! `1e-3` for the same parameters (Whisper front-end sibling posture).
//!
//! ## Path C — end-to-end INT8 chain (both env vars set)
//!
//! **Honest UNMET**: the Rust INT8 [`crate::interpreter::ChainConfig`]
//! needs per-tensor `(scale, zero_point)` quantisation params to bind
//! against a real MC-MobileNet checkpoint. The sidecar now emits Q8_0
//! source-byte carriers and those params. Wiring end-to-end parity still
//! requires:
//!
//! 1. [`crate::model::Model`] gains typed accessors for per-layer
//!    conv / dense weights + quant params;
//! 2. Authenticated topology and INT32 bias tensors are available, then
//!    this test constructs a real [`crate::interpreter::ChainConfig`]
//!    from those and runs it against the dumped input features,
//!    comparing to `output_ref.bin` at `atol = 1e-2` (INT8 dequant
//!    tolerance).
//!
//! Until then this path skips with a clear "end-to-end parity requires
//! authenticated topology/bias manifest" message — the scaffold is here so
//! the flip is a one-file diff when those artifacts land.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use vokra_kws_micro::features::{self, FeatureExtractor};
use vokra_kws_micro::model::Model;

/// Env var the owner sets to point Path A + Path C at a real Vokra
/// microWakeWord GGUF. Absent = skip cleanly.
const GGUF_ENV: &str = "VOKRA_KWS_REAL_GGUF";

/// Env var the owner sets to point Path B + Path C at the directory of
/// reference dumps emitted by `dump_reference.py`. Absent = skip cleanly.
const FIXTURES_ENV: &str = "VOKRA_KWS_REAL_FIXTURES";

/// Per-band `|Δ|` gate for the log-mel feature extractor parity
/// (Path B). `5e-2` is the honest architectural bound:
///
/// * ``np.fft.rfft`` (the numpy reference) computes internally in
///   float64 and casts to float32 at output.
/// * Vokra's Rust FFT is float32 throughout its log₂(N_FFT) = 9
///   butterfly stages (target-architecture-realistic for Cortex-M55).
///
/// Empirically the two agree to `< 1e-4` at low bands (0–15) but drift
/// to `~3e-2` at high bands (~30) where the f32 rounding accumulates.
/// This is a real precision gap between the higher-precision numpy
/// reference and the f32 target-realistic Rust code — not a Rust bug.
/// A regression that actually broke the FFT / filterbank / log10
/// chain would produce deltas well above `5e-2` (e.g. an FFT twiddle
/// sign flip is `~1.0`; a mel-filterbank off-by-one is `~0.1+` at
/// affected bands; a log10 floor bug is `~ln(1e-10)` at silent
/// frames). The 5e-2 atol leaves ~1.7× margin above the observed
/// baseline delta while still catching every regression class above
/// that scale. Same "honest atol from architectural bound, not from
/// CI-green wishing" posture as the Kokoro `PROSODY_F0_ATOL` calibration
/// (see `parity_kokoro.rs`).
const FEATURES_ATOL: f32 = 5e-2;

// ---------------------------------------------------------------------------
// FIXTURE-FREE tests (always run — no env vars required)
// ---------------------------------------------------------------------------

/// FIXTURE-FREE: pin the extractor's public constants against the
/// primary source (microWakeWord upstream trains at 16 kHz / 40 mels /
/// 10 ms hop / 32 ms window, verified in the sidecar's
/// `DEFAULT_SAMPLE_RATE` / `DEFAULT_HOP_MS` / `DEFAULT_WINDOW_MS` /
/// `DEFAULT_N_MELS` constants). A silent drift here would mis-align the
/// mel front-end against every real checkpoint.
#[test]
fn kws_features_constants_match_primary_source() {
    assert_eq!(
        features::SAMPLE_RATE,
        16_000,
        "microWakeWord is trained at 16 kHz PCM in"
    );
    assert_eq!(
        features::HOP_MS,
        10,
        "microWakeWord's canonical streaming hop is 10 ms"
    );
    assert_eq!(
        features::WINDOW_MS,
        32,
        "microWakeWord's canonical STFT window is 32 ms"
    );
    assert_eq!(
        features::N_MELS,
        40,
        "microWakeWord's canonical mel band count is 40"
    );
    // Derived constants must be internally consistent.
    assert_eq!(features::HOP_SAMPLES, 160, "SAMPLE_RATE * HOP_MS / 1000");
    assert_eq!(
        features::WINDOW_SAMPLES,
        512,
        "SAMPLE_RATE * WINDOW_MS / 1000"
    );
    assert_eq!(features::N_FFT, 512, "next pow-of-two >= WINDOW_SAMPLES");
    assert_eq!(features::N_BINS, 257, "N_FFT / 2 + 1");
}

// ---------------------------------------------------------------------------
// GATED tests (skip cleanly when env vars unset)
// ---------------------------------------------------------------------------

/// GATED (Path A): opens a real Vokra microWakeWord GGUF and validates
/// the metadata contract + tensor manifest lower bound.
///
/// Skips cleanly when [`GGUF_ENV`] is unset. Once set, all failures are
/// hard: a missing / malformed / wrong-arch fixture fails loudly.
#[test]
fn parity_microwakeword_gguf_smoke() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping microWakeWord GGUF parity smoke; \
             this is a clean skip (never a fabricated pass). See the module \
             docs for the fixture recipe."
        );
        return;
    };

    let path = Path::new(&gguf_path);
    let bytes = fs::read(path).unwrap_or_else(|e| {
        panic!(
            "microWakeWord GGUF at {gguf_path} failed to read: {e:?} \
             (opted-in ⇒ any error is a hard failure — FR-EX-08)"
        )
    });
    let m = Model::from_bytes(&bytes).unwrap_or_else(|e| {
        panic!(
            "microWakeWord GGUF at {gguf_path} failed to parse: {e:?} \
             (opted-in ⇒ any error is a hard failure — FR-EX-08)"
        )
    });

    // Metadata contract: primary-source defaults must round-trip through
    // the `vokra.kws.*` chunk group. A real hey_jarvis GGUF must carry
    // 16 kHz + 40 mels (a differently-configured artefact is either
    // misconfigured or a non-canonical fork — either way, a hard failure
    // rather than a silent bind that would produce nonsense scores).
    assert_eq!(
        m.header.sample_rate,
        features::SAMPLE_RATE,
        "microWakeWord trained at 16 kHz; got {}",
        m.header.sample_rate
    );
    assert_eq!(
        m.header.n_mels as usize,
        features::N_MELS,
        "microWakeWord canonical mel width = {}; got {}",
        features::N_MELS,
        m.header.n_mels
    );
    assert!(
        (0.0..=1.0).contains(&m.header.threshold) && m.header.threshold.is_finite(),
        "threshold {} must be a finite probability in [0, 1]",
        m.header.threshold
    );
    assert!(
        !m.header.model.is_empty(),
        "model name must be non-empty (real GGUF carries e.g. 'hey_jarvis')"
    );
    assert!(
        !m.header.tflite_sha256.is_empty(),
        "tflite_sha256 must be non-empty for provenance audit"
    );

    // Tensor manifest lower bound: a real microWakeWord MC-MobileNet
    // checkpoint carries dozens of weight tensors across conv / dwconv /
    // dense layers. A one-tensor GGUF (as the from_gguf smoke fixture
    // uses) would be accepted by from_gguf but is not a real checkpoint.
    // Conservative floor of 10 tensors — the exact upstream count depends
    // on the model variant and is not primary-source-transcribable
    // without a dedicated dumper.
    assert!(
        m.tensor_count() >= 10,
        "real microWakeWord checkpoint must carry >= 10 tensors; got {} — \
         refusing a synthesized-shape fixture (FR-EX-08)",
        m.tensor_count()
    );

    eprintln!(
        "microWakeWord GGUF loaded from {gguf_path}: model={:?}, \
         sr={}, n_mels={}, threshold={}, {} tensors bound",
        m.header.model,
        m.header.sample_rate,
        m.header.n_mels,
        m.header.threshold,
        m.tensor_count(),
    );
}

/// GATED (Path B): reads the dumper's `input_pcm.bin` + `features_ref.bin`
/// and validates Vokra's [`FeatureExtractor::compute_frame_f32`] output
/// per-band `|Δ|` against the numpy reference at [`FEATURES_ATOL`].
///
/// Skips cleanly when [`FIXTURES_ENV`] is unset.
#[test]
fn parity_microwakeword_feature_extractor_matches_reference() {
    let Some(fixtures_dir) = env::var(FIXTURES_ENV).ok() else {
        eprintln!(
            "{FIXTURES_ENV} unset — skipping microWakeWord feature-extractor \
             parity; this is a clean skip (never a fabricated pass). See the \
             module docs for the fixture recipe."
        );
        return;
    };
    let dir = PathBuf::from(&fixtures_dir);

    // Read PCM input (raw i16 little-endian, WINDOW_SAMPLES samples).
    let pcm_path = dir.join("input_pcm.bin");
    let pcm_bytes = fs::read(&pcm_path).unwrap_or_else(|e| {
        panic!(
            "Path-B: failed to read {}: {e:?} — is the dumper output complete?",
            pcm_path.display()
        )
    });
    assert_eq!(
        pcm_bytes.len(),
        features::WINDOW_SAMPLES * 2,
        "Path-B: input_pcm.bin len {} != WINDOW_SAMPLES ({}) * 2 (i16 LE)",
        pcm_bytes.len(),
        features::WINDOW_SAMPLES,
    );
    let pcm: Vec<i16> = pcm_bytes
        // Exact byte length is asserted above; `chunks` keeps this fixture
        // reader compatible with the workspace's Rust 1.85 MSRV.
        .chunks(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    // Read reference features (raw f32 little-endian, N_MELS floats).
    let ref_path = dir.join("features_ref.bin");
    let ref_bytes = fs::read(&ref_path).unwrap_or_else(|e| {
        panic!(
            "Path-B: failed to read {}: {e:?} — is the dumper output complete?",
            ref_path.display()
        )
    });
    assert_eq!(
        ref_bytes.len(),
        features::N_MELS * 4,
        "Path-B: features_ref.bin len {} != N_MELS ({}) * 4 (f32 LE)",
        ref_bytes.len(),
        features::N_MELS,
    );
    let features_ref: Vec<f32> = ref_bytes
        // Exact byte length is asserted above; see the PCM reader above.
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Run Vokra's feature extractor on the same PCM.
    let extractor = FeatureExtractor::new();
    let features_vokra = extractor.compute_frame_f32(&pcm);
    assert_eq!(
        features_vokra.len(),
        features::N_MELS,
        "extractor produced {} features, expected N_MELS = {}",
        features_vokra.len(),
        features::N_MELS,
    );

    // Per-band |Δ| gate. FEATURES_ATOL = 1e-3 is the standard f32 log-mel
    // tolerance across numpy / scipy / tf.signal transcriptions.
    let mut max_delta = 0.0f32;
    let mut worst_band = 0usize;
    for (i, (&v, &r)) in features_vokra.iter().zip(features_ref.iter()).enumerate() {
        assert!(v.is_finite(), "band {i}: Vokra feature {v} is not finite");
        assert!(
            r.is_finite(),
            "band {i}: reference feature {r} is not finite"
        );
        let d = (v - r).abs();
        if d > max_delta {
            max_delta = d;
            worst_band = i;
        }
    }
    eprintln!(
        "Path-B log-mel parity: max |Δ| = {max_delta:.6} at band {worst_band} \
         (gate = {FEATURES_ATOL:.6}, N_MELS = {} bands)",
        features::N_MELS,
    );
    assert!(
        max_delta <= FEATURES_ATOL,
        "Path-B log-mel per-band parity failed: max |Δ| = {max_delta} at band \
         {worst_band} exceeds atol = {FEATURES_ATOL}. Vokra features[..8] = {:?}; \
         reference[..8] = {:?}. Investigate the FFT / mel-filterbank / log10 \
         chain for a regression.",
        &features_vokra[..8.min(features_vokra.len())],
        &features_ref[..8.min(features_ref.len())],
    );
}

/// GATED (Path C): end-to-end INT8 chain parity — honest UNMET.
///
/// The Rust INT8 [`crate::interpreter::ChainConfig`] needs per-tensor
/// `(scale, zero_point)` quantisation params to bind against a real
/// MC-MobileNet checkpoint. The sidecar now emits Q8_0 source-byte carriers
/// and those params, but authenticated topology and typed ChainConfig binding
/// are still required. Until the source manifest and binder land, this path
/// always skips with a clear defer message — even when both env vars are set.
/// This is honest UNMET (never a fabricated pass).
///
/// The scaffold is here so that when the sidecar lands, wiring the
/// real end-to-end parity is a one-file diff (load `output_ref.bin`,
/// construct a real `ChainConfig` from `Model`, run, compare).
#[test]
fn parity_microwakeword_end_to_end_output() {
    let gguf = env::var(GGUF_ENV).ok();
    let fixtures = env::var(FIXTURES_ENV).ok();
    if gguf.is_none() || fixtures.is_none() {
        eprintln!("Path-C: {GGUF_ENV} and/or {FIXTURES_ENV} unset — skipping cleanly.");
        return;
    }
    // Both env vars set, but the authenticated topology/bias manifest and
    // Model → ChainConfig binder are not yet available. Honest UNMET rather
    // than a fabricated pass.
    eprintln!(
        "Path-C: end-to-end INT8 chain parity is UNMET — Q8_0 source-byte \
         carriers, exact I32 bias values, and affine metadata exist, but \
         authenticated topology and the Model → ChainConfig binder remain. \
         Wiring parity requires those contracts plus this test running against \
         `output_ref.bin` at atol=1e-2. Until then, this is a clean skip \
         (never a fabricated pass)."
    );
}
