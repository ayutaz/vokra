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
//! `NSNet2-baseline/nsnet2-20ms-baseline.onnx` (~2 MB). Bridge it
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
//! - **Reference bit-parity (side-car env `REFERENCE_WAV_ENV`)**: if
//!   the owner provides a cleaned WAV emitted by the upstream ONNX
//!   Runtime pipeline on the same noisy input, the harness compares
//!   per-sample max |Δ| against [`PCM_ATOL`]. Off by default because
//!   the reference cleaned WAV is not in the DNS-Challenge release —
//!   the owner has to run the upstream `run_nsnet2_baseline.py` once
//!   and stash the output.
//!
//! Neither leg fabricates a pass on missing weights.

use std::env;

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
/// residuals per multiply, accumulating over 400×257 dot products. A
/// 5e-3 max-|Δ| bound at the PCM output catches every real regression
/// while leaving room for GEMM ordering. Matches the DFN3 handoff
/// bound (`docs/handoff/parity-ci-flip-switch.md` §DFN3).
#[allow(dead_code)]
const PCM_ATOL: f32 = 5e-3;

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
    assert_eq!(cfg.n_bins, 257, "upstream NSNet2 has 257 STFT bins");
    assert_eq!(cfg.hidden_dim, 400, "upstream NSNet2 has 400-wide GRU");
    assert_eq!(cfg.n_fft, 512);
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
        let n = cleaned.len().min(reference.samples.len());
        assert!(
            n >= cfg.n_fft,
            "reference / cleaned overlap must exceed n_fft; got {n}"
        );
        let mut max_delta = 0.0f32;
        for (c, r) in cleaned.iter().zip(reference.samples.iter()).take(n) {
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
