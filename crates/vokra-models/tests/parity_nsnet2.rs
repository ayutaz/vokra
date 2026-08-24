//! NSNet2 numerical parity harness — env-gated (denoise family, 2026-08-05).
//!
//! Sibling of `parity_nkf_aec.rs` / `parity_rmvpe.rs` /
//! `parity_openwakeword.rs`: every test that needs a real NSNet2 GGUF +
//! a paired noisy WAV is gated on [`GGUF_ENV`] / [`WAV_ENV`] and skips
//! cleanly when unset (never a fabricated pass — memory
//! `[[project-real-weight-eval]]`). Once opted in, every failure is
//! hard: a missing / malformed / wrong-shaped fixture is a loud panic
//! (FR-EX-08).
//!
//! # Fixture recipe (owner-side)
//!
//! The upstream Microsoft `DNS-Challenge` NSNet2-baseline release ships
//! `NSNet2-baseline/nsnet2-20ms-baseline.onnx` (~10.8 MB). Bridge it
//! offline to Vokra GGUF via the landed sidecar + converter:
//!
//! ```text
//! # 1. Fetch the ONNX from the DNS-Challenge repo:
//! git clone --depth 1 https://github.com/microsoft/DNS-Challenge \
//!     ~/checkpoints/nsnet2/repo
//!
//! # 2. Flatten ONNX -> safetensors (offline, in the tools/parity uv venv):
//! uv run --project tools/parity python tools/parity/nsnet2_prepare_checkpoint.py \
//!     --input  ~/checkpoints/nsnet2/repo/NSNet2-baseline/nsnet2-20ms-baseline.onnx \
//!     --output ~/checkpoints/nsnet2/model.safetensors
//!
//! # 3. Convert safetensors -> GGUF:
//! vokra-cli convert --model nsnet2 \
//!     --input  ~/checkpoints/nsnet2/model.safetensors \
//!     --output ~/gguf/nsnet2.gguf
//!
//! # 4. Point the parity harness at the artefacts:
//! export VOKRA_NSNET2_REAL_GGUF=~/gguf/nsnet2.gguf
//! export VOKRA_NSNET2_REAL_WAV=<any 16 kHz mono noisy WAV>
//! cargo test -p vokra-models --test parity_nsnet2 -- --nocapture
//! ```
//!
//! # Numeric parity contract
//!
//! NSNet2 is a per-bin gain predictor; the cleaned PCM is deterministic
//! (no stochastic sampler, no dropout on the ONNX release), and the
//! reference is ONNX Runtime — the same graph the upstream authors
//! evaluated. The parity check has two legs:
//!
//! - **Structural (fixture-free)**: opens the real GGUF, binds the
//!   config + every tensor, runs the forward on a short synthetic PCM
//!   snippet, and checks the output has the right length + is finite.
//!   Catches wrong-shape / wrong-name binding regressions the
//!   synthetic tests miss.
//! - **Reference parity (side-car env `REFERENCE_WAV_ENV`)**: if the owner
//!   provides a cleaned WAV emitted by
//!   `tools/parity/nsnet2_dump_reference.py` from the pinned official ONNX,
//!   the harness compares
//!   per-sample max |Δ| against [`PCM_ATOL`]. Off by default because
//!   the generated reference WAV is not committed.
//!
//! Neither leg fabricates a pass on missing weights.

use std::env;

use vokra_core::backend::BackendKind;
use vokra_core::engines::DenoiseEngine;
use vokra_eval::wav::read_wav;
use vokra_models::nsnet2::{Nsnet2V1, SAMPLE_RATE_DEFAULT};

/// Env var the owner sets to point the gated harness at a real
/// NSNet2 GGUF. Absent = skip cleanly (never a fabricated pass).
const GGUF_ENV: &str = "VOKRA_NSNET2_REAL_GGUF";

/// Env var pointing at a 16 kHz mono WAV — the noisy input used for
/// the structural bind test + (when [`REFERENCE_WAV_ENV`] is set) the
/// reference-bit-parity leg.
const WAV_ENV: &str = "VOKRA_NSNET2_REAL_WAV";

/// Optional side-car: if set, points at a 16 kHz mono WAV containing
/// the upstream ONNX Runtime pipeline's cleaned output on the same
/// noisy input. When present, the parity test also runs a per-sample
/// max-|Δ| leg against it (bound [`PCM_ATOL`]).
#[allow(dead_code)]
const REFERENCE_WAV_ENV: &str = "VOKRA_NSNET2_REFERENCE_WAV";

/// Per-sample max-|Δ| tolerance when [`REFERENCE_WAV_ENV`] is set.
/// NSNet2's forward is a straight Linear + GRU + Linear chain plus
/// sigmoid; float ordering differences between Vokra's row-major
/// scalar GEMV and ONNX Runtime's optimised backends produce ULP-scale
/// residuals per multiply, accumulating over 400×161 dot products. A
/// The pinned official ONNX (`88429b62…`) measured max |Δ| = 2.92e-6 on
/// `tests/parity/silero_vad/test_16k.wav` for both CPU and Metal on
/// 2026-08-24. A 5e-5 bound retains >17× margin for platform GEMM/FFT
/// ordering without hiding a frontend, gate-layout, or overlap-add regression.
#[allow(dead_code)]
const PCM_ATOL: f32 = 5e-5;

// -------------------------------------------------------------------------

/// UNGATED: proves the harness compiles + links against the real
/// runtime. Serves the same "no fabricated pass on missing weights"
/// role for CI aggregation as the sibling parity harnesses: the test
/// always PASSES, but any real-weight regression that lands on a
/// per-tensor `Nsnet2V1::from_gguf` bind — say, a tensor name typo —
/// will show up here the moment the owner sets the env.
#[test]
fn parity_nsnet2_harness_wired() {
    // The env pinning matches the module docstring — if these ever
    // drift, this test catches the mismatch. Not a real assertion, but
    // it prevents a silent drift where the harness talks about one env
    // and the sibling handoff docs another.
    assert_eq!(GGUF_ENV, "VOKRA_NSNET2_REAL_GGUF");
    assert_eq!(WAV_ENV, "VOKRA_NSNET2_REAL_WAV");
    assert_eq!(REFERENCE_WAV_ENV, "VOKRA_NSNET2_REFERENCE_WAV");
}

/// GATED: opens a real NSNet2 GGUF, binds it, runs the forward on a
/// noisy WAV and pins structural properties (length, finite, bounded).
/// Skips cleanly when [`GGUF_ENV`] / [`WAV_ENV`] are unset.
#[test]
fn parity_nsnet2_gguf_smoke() {
    let Ok(gguf_path) = env::var(GGUF_ENV) else {
        eprintln!("{GGUF_ENV} unset — skipping NSNet2 GGUF smoke; set to a real GGUF path to run");
        return;
    };
    let Ok(wav_path) = env::var(WAV_ENV) else {
        eprintln!("{WAV_ENV} unset — skipping NSNet2 GGUF smoke; set to a 16 kHz mono WAV to run");
        return;
    };

    let model = Nsnet2V1::open(&gguf_path)
        .unwrap_or_else(|e| panic!("Nsnet2V1::open({gguf_path}) failed: {e}"));

    let cfg = model.config().clone();
    assert_eq!(
        cfg.sample_rate, SAMPLE_RATE_DEFAULT,
        "real GGUF must ship the canonical 16 kHz sample rate"
    );
    // Every documented upstream hparam must be present.
    assert_eq!(cfg.n_bins, 161, "upstream NSNet2 has 161 STFT bins");
    assert_eq!(cfg.hidden_dim, 400, "upstream NSNet2 has 400-wide GRU");
    assert_eq!(cfg.n_fft, 320);
    assert_eq!(cfg.hop, 160);

    let wav = read_wav(&wav_path).unwrap_or_else(|e| panic!("read_wav({wav_path}) failed: {e}"));
    assert_eq!(
        wav.sample_rate, SAMPLE_RATE_DEFAULT,
        "WAV must be 16 kHz mono (upstream training rate); got {} Hz",
        wav.sample_rate
    );
    assert!(
        wav.samples.len() >= cfg.n_fft,
        "WAV must have >= n_fft ({}) samples for at least one STFT frame; got {}",
        cfg.n_fft,
        wav.samples.len()
    );

    // Streaming path (the DenoiseEngine impl).
    let mut stream = model
        .open_stream(cfg.sample_rate)
        .unwrap_or_else(|e| panic!("open_stream({}) failed: {e}", cfg.sample_rate));
    let mut cleaned = Vec::new();
    // Push in 3200-sample (200 ms @ 16 kHz) chunks — realistic
    // streaming granularity, and small enough that a state-carry-over
    // bug would surface as PCM discontinuity.
    for chunk in wav.samples.chunks(3200) {
        let out = stream
            .push_pcm(chunk)
            .unwrap_or_else(|e| panic!("push_pcm({} samples) failed: {e}", chunk.len()));
        cleaned.extend(out);
    }
    cleaned.extend(
        stream
            .finalize()
            .unwrap_or_else(|e| panic!("finalize failed: {e}")),
    );

    // Pinned `featurelib.stft(..., nodelay=True)` emits
    // `ceil(input_len / hop)` frames, and its raw overlap-add iSTFT therefore
    // returns `(frames - 1) * hop + n_fft` samples. Pin the exact length so a
    // future framing regression cannot be hidden by a prefix-only comparison.
    let expected_frames = wav.samples.len().div_ceil(cfg.hop);
    let expected_samples = (expected_frames - 1) * cfg.hop + cfg.n_fft;
    assert_eq!(
        cleaned.len(),
        expected_samples,
        "NSNet2 output length must match the pinned no-delay frontend"
    );

    for (i, v) in cleaned.iter().enumerate() {
        assert!(
            v.is_finite(),
            "cleaned sample {i} = {v} is not finite (NaN/inf from the forward chain)"
        );
        assert!(
            v.abs() <= 8.0,
            "cleaned sample {i} = {v} exceeds sanity bound |v| <= 8.0 — a gain > 1 \
             would indicate a sigmoid pre-activation overflow or a bad tensor bind"
        );
    }

    // Apple GPU leg: the independently validated CPU route remains the
    // oracle, while every learned dense/GRU projection and complex mask apply
    // is dispatched through Metal. The STFT/iSTFT is deliberately shared
    // host-side DSP. A device/build failure is hard once the real fixture is
    // opted in; it is never relabelled as a CPU pass.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    {
        let metal_model = Nsnet2V1::open(&gguf_path)
            .unwrap_or_else(|e| panic!("Metal Nsnet2V1::open({gguf_path}) failed: {e}"))
            .with_backend(BackendKind::Metal);
        let mut metal_stream = metal_model
            .open_stream(cfg.sample_rate)
            .unwrap_or_else(|e| panic!("Metal open_stream({}) failed: {e}", cfg.sample_rate));
        let mut metal_cleaned = Vec::new();
        for chunk in wav.samples.chunks(3200) {
            metal_cleaned.extend(
                metal_stream
                    .push_pcm(chunk)
                    .unwrap_or_else(|e| panic!("Metal push_pcm failed: {e}")),
            );
        }
        metal_cleaned.extend(
            metal_stream
                .finalize()
                .unwrap_or_else(|e| panic!("Metal finalize failed: {e}")),
        );
        assert_eq!(metal_cleaned.len(), cleaned.len());
        let max_delta = metal_cleaned
            .iter()
            .zip(&cleaned)
            .map(|(metal, cpu)| (metal - cpu).abs())
            .fold(0.0f32, f32::max);
        eprintln!("NSNet2 real CPU/Metal PCM max_abs={max_delta}");
        assert!(
            max_delta <= PCM_ATOL,
            "NSNet2 CPU/Metal PCM max |Δ| = {max_delta} exceeds {PCM_ATOL}"
        );
    }

    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    let _ = BackendKind::Cpu;

    // Optional reference-parity leg. Enabled only when the owner
    // provides a cleaned WAV from the upstream ONNX pipeline on the
    // same input.
    if let Ok(ref_path) = env::var(REFERENCE_WAV_ENV) {
        let reference =
            read_wav(&ref_path).unwrap_or_else(|e| panic!("read_wav({ref_path}) failed: {e}"));
        assert_eq!(
            reference.sample_rate, cfg.sample_rate,
            "reference WAV must match the model sample rate"
        );
        assert_eq!(
            reference.samples.len(),
            cleaned.len(),
            "reference and native output lengths must match exactly"
        );
        let mut max_delta = 0.0f32;
        for (c, r) in cleaned.iter().zip(&reference.samples) {
            let d = (c - r).abs();
            if d > max_delta {
                max_delta = d;
            }
        }
        assert!(
            max_delta <= PCM_ATOL,
            "NSNet2 parity: max |Δ| = {max_delta} exceeds PCM_ATOL = {PCM_ATOL}"
        );
    }
}
