//! DNSMOS P.808 / P.835 numerical parity harness — env-gated (2026-08-05).
//!
//! Sibling of `parity_openwakeword.rs` / `parity_rmvpe.rs`: every test
//! that needs a real DNSMOS GGUF is gated on the [`GGUF_ENV`]
//! environment variable and skips cleanly when unset (never a
//! fabricated pass — memory `[[project-real-weight-eval]]`). Once
//! opted in, every failure is hard: a missing / malformed / wrong-
//! shaped fixture is a loud panic (FR-EX-08).
//!
//! # Fixture recipe (owner-side)
//!
//! `microsoft/DNS-Challenge/DNSMOS/` (MIT) ships two ONNX checkpoints
//! (`model_v8.onnx` = P.808, `sig_bak_ovr.onnx` = P.835). Bridge them
//! offline into a merged safetensors + reference JSONL:
//!
//! ```text
//! # 1. Fetch upstream ONNX from github.com/microsoft/DNS-Challenge
//! # 2. Prepare merged safetensors (uv / Python 3.12):
//! uv run python tools/parity/dnsmos_prepare_checkpoint.py \
//!     --p808     ~/DNSMOS/model_v8.onnx \
//!     --p835     ~/DNSMOS/sig_bak_ovr.onnx \
//!     --output-st ~/dnsmos.safetensors
//! # 3. Emit the reference JSONL (uses the upstream `dnsmos_local.py`
//! #    under `onnxruntime` — offline reference tool only, never enters
//! #    the runtime dependency graph):
//! uv run python tools/parity/dnsmos_score_reference.py \
//!     --p808      ~/DNSMOS/model_v8.onnx \
//!     --p835      ~/DNSMOS/sig_bak_ovr.onnx \
//!     --input-wav ~/test-clean.wav \
//!     --input-wav ~/test-noisy.wav \
//!     --output-jsonl ~/dnsmos_reference.jsonl
//! # 4. Convert safetensors → GGUF:
//! vokra-cli convert --model dnsmos-p808-p835 \
//!     --input  ~/dnsmos.safetensors \
//!     --output ~/dnsmos.gguf
//! # 5. Point the parity harness at every artefact:
//! export VOKRA_DNSMOS_REAL_GGUF=~/dnsmos.gguf
//! export VOKRA_DNSMOS_REAL_WAVS=~/test-clean.wav:~/test-noisy.wav
//! export VOKRA_DNSMOS_REFERENCE_JSONL=~/dnsmos_reference.jsonl
//! cargo test -p vokra-models --test parity_dnsmos -- --nocapture
//! ```
//!
//! # Numeric parity contract
//!
//! DNSMOS emits per-chunk MOS scalars (`p808 ∈ [1,5]` and P.835
//! `(SIG, BAK, OVRL) ∈ [1,5]³`), averaged across all 9.01 s chunks.
//! [`MOS_ATOL`] bounds `max_v |Δ|` between Vokra's runtime and the
//! upstream `dnsmos_local.py` + `onnxruntime` reference — calibrated
//! after the first owner-run once real weights ship (mirror of
//! `parity_kokoro.rs`'s per-tensor scaffold, see
//! `[[feedback-honest-parity-atol]]`).
//!
//! **When the CNN backbone forward wires** (deferred per module docs
//! in `crates/vokra-models/src/dnsmos_p808_p835/mod.rs`), the
//! `parity_dnsmos_gated_scores` test upgrades from a loud-partial smoke
//! to a real MOS comparison; no code change beyond wiring
//! `cnn_forward` behind the `vokra.dnsmos.{p808,p835}.topology`
//! metadata is required here.

use std::env;
use std::path::Path;

use vokra_core::VokraError;
use vokra_core::engines::MosScorerEngine;
use vokra_models::dnsmos_p808_p835::{
    Dnsmos, DnsmosSubmodel, EXPECTED_SAMPLE_RATE, INPUT_LENGTH_SAMPLES,
};

/// Env var the owner sets to point the gated harness at a real DNSMOS
/// GGUF. Absent = skip cleanly (never a fabricated pass).
const GGUF_ENV: &str = "VOKRA_DNSMOS_REAL_GGUF";

/// Env var holding a colon-separated list of 16 kHz mono WAV paths the
/// reference scorer also consumed. Iterating in the same order as the
/// reference JSONL lets the parity harness key on file basename.
const WAVS_ENV: &str = "VOKRA_DNSMOS_REAL_WAVS";

/// Env var pointing at the reference JSONL side-car — one line per
/// WAV, each `{"wav": "<basename>", "p808": <f>, "sig": <f>, ...}`.
const REFERENCE_JSONL_ENV: &str = "VOKRA_DNSMOS_REFERENCE_JSONL";

/// Per-MOS-scalar |Δ| tolerance (consumed once the CNN forward wires).
/// Calibrated by the first owner-run against the upstream reference —
/// the initial value is a placeholder honest to typical
/// `onnxruntime`-vs-Rust f32 rounding on a small CNN + FC (mirror of
/// the openwakeword `PROB_ATOL = 1e-4`).
#[allow(dead_code)]
const MOS_ATOL: f32 = 1e-2;

/// FIXTURE-FREE: primary-source constants pin. `INPUT_LENGTH_SAMPLES`
/// = 144160 (9.01 s at 16 kHz) is transcribed verbatim from
/// `microsoft/DNS-Challenge/DNSMOS/dnsmos_local.py:INPUT_LENGTH` — a
/// silent drift in this constant would misalign the chunking window
/// against every reference dumper run.
#[test]
fn dnsmos_primary_source_constants_pin() {
    assert_eq!(
        EXPECTED_SAMPLE_RATE, 16_000,
        "DNSMOS is trained at 16 kHz PCM in (dnsmos_local.py::SAMPLING_RATE)"
    );
    assert_eq!(
        INPUT_LENGTH_SAMPLES, 144_160,
        "DNSMOS chunks input to 9.01 s windows (dnsmos_local.py::INPUT_LENGTH)"
    );
    // Sub-model short-name pin — the tensor prefix and metadata key
    // both derive from these strings.
    assert_eq!(DnsmosSubmodel::P808.short(), "p808");
    assert_eq!(DnsmosSubmodel::P835.short(), "p835");
    assert_eq!(DnsmosSubmodel::P808.tensor_prefix(), "p808.");
    assert_eq!(DnsmosSubmodel::P835.tensor_prefix(), "p835.");
}

/// FIXTURE-FREE: the loud-partial `Dnsmos::score_*` contract must
/// surface a `VokraError::UnsupportedOp` naming both the future
/// topology metadata chunk and the sidecar to extend so an owner
/// reading the error knows exactly where to flip the switch. Uses the
/// synthesized-bundle constructor which lets this test run without a
/// real GGUF fixture.
#[test]
fn dnsmos_score_paths_are_loud_partial() {
    let session = Dnsmos::synthesized();
    let pcm = vec![0.0f32; 16_000]; // 1 s zero — small enough to short-circuit.

    let err = session
        .score_p808(&pcm)
        .expect_err("loud-partial p808 must not return Ok");
    let msg = match err {
        VokraError::UnsupportedOp(m) => m,
        other => panic!("expected UnsupportedOp for p808, got {other:?}"),
    };
    assert!(
        msg.contains("vokra.dnsmos") && msg.contains("topology"),
        "loud-partial message must name the topology metadata: {msg}"
    );
    assert!(
        msg.contains("dnsmos_prepare_checkpoint"),
        "loud-partial message must name the sidecar to extend: {msg}"
    );

    // Same posture for the 3-scalar P.835 head.
    let err = session
        .score_p835(&pcm)
        .expect_err("loud-partial p835 must not return Ok");
    assert!(matches!(err, VokraError::UnsupportedOp(_)));
}

/// GATED: opens a real DNSMOS GGUF and verifies the load path is a
/// genuine bind (real config parse, real bundle-inventory walk, real
/// tensor-prefix presence check per variant).
///
/// Skips cleanly when [`GGUF_ENV`] is unset. Once set, all failures
/// are hard: a missing / malformed / wrong-arch fixture fails loudly.
#[test]
fn parity_dnsmos_gguf_smoke() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping DNSMOS GGUF parity smoke; \
             this is a clean skip (never a fabricated pass). See the \
             module docs for the fixture recipe."
        );
        return;
    };

    let path = Path::new(&gguf_path);
    let session = Dnsmos::from_path(path).unwrap_or_else(|e| {
        panic!(
            "dnsmos GGUF at {gguf_path} failed to load: {e:?} \
             (opted-in ⇒ any error is a hard failure — FR-EX-08)"
        )
    });

    let cfg = session.config();
    assert_eq!(
        cfg.sample_rate, 16_000,
        "DNSMOS is trained at 16 kHz PCM in; a differently-rated GGUF is \
         either mis-configured or a non-canonical fork (loud-fail)"
    );
    assert!(
        !cfg.bundle.is_empty(),
        "real DNSMOS GGUF must advertise at least one variant; bundle is empty"
    );
    for v in &cfg.bundle {
        assert!(
            v == "p808" || v == "p835",
            "bundle inventory carries unknown variant `{v}` (real DNSMOS ships \
             only `p808` / `p835`)"
        );
    }
    let variants = session.variants();
    for v in variants {
        assert!(cfg.bundle.iter().any(|b| b == v));
    }
    eprintln!(
        "dnsmos GGUF loaded from {gguf_path}: bundle={:?}, sample_rate={} Hz",
        cfg.bundle, cfg.sample_rate
    );
}

/// GATED: pushes real 16 kHz PCM through the session and compares MOS
/// scalars against the reference JSONL. Today this test enforces the
/// **loud-partial** posture: it opts in, loads the real GGUF, invokes
/// [`Dnsmos::score`] on a placeholder buffer, and asserts the call
/// errors with [`VokraError::UnsupportedOp`] — no fabricated `0.0 == 0.0`
/// "match". The moment the CNN backbone forward wires (see the module
/// docs), the assertion flips from "expect UnsupportedOp" to "match
/// reference MOS within [`MOS_ATOL`]".
#[test]
fn parity_dnsmos_gated_scores() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping DNSMOS MOS parity; clean skip. \
             See module docs for the fixture recipe."
        );
        return;
    };
    // WAV / JSONL side-cars are only consumed once the CNN forward
    // lights up; assert their presence for owner-side documentation so
    // the missing side-car is caught early.
    if env::var(WAVS_ENV).is_err() {
        eprintln!(
            "note: {WAVS_ENV} unset — the loud-partial gate still fires \
             without it; the side-car becomes required once the CNN \
             forward wires."
        );
    }
    if env::var(REFERENCE_JSONL_ENV).is_err() {
        eprintln!("note: {REFERENCE_JSONL_ENV} unset — same posture as {WAVS_ENV}.");
    }

    let session = Dnsmos::from_path(&gguf_path)
        .unwrap_or_else(|e| panic!("dnsmos GGUF at {gguf_path} failed to load: {e:?}"));

    // A 1 s zero buffer suffices to reach the loud-partial gate; it is
    // still under the reference 9.01 s window so the length-check runs.
    let pcm = vec![0.0f32; 16_000];
    let result = session.score(&pcm);

    match result {
        Err(VokraError::UnsupportedOp(msg)) => {
            assert!(
                msg.contains("vokra.dnsmos") && msg.contains("topology"),
                "loud-partial message must name the topology metadata: {msg}"
            );
            assert!(
                msg.contains("dnsmos_prepare_checkpoint"),
                "loud-partial message must name the sidecar to extend: {msg}"
            );
            eprintln!(
                "dnsmos MOS parity: loud-partial gate fired as expected — the \
                 CNN backbone forward is deferred to the topology-metadata \
                 sidecar extension. When that lands, this test flips to \
                 reference-JSONL MOS comparison (see the module docs)."
            );
        }
        Ok(_score) => {
            // Once the CNN forward lights up we land here. The follow-
            // up wave parses REFERENCE_JSONL_ENV with vokra_core::json
            // (zero-dep) and iterates WAVS_ENV.
            let _ = env::var(REFERENCE_JSONL_ENV).unwrap_or_else(|_| {
                panic!(
                    "the CNN forward is now real but {REFERENCE_JSONL_ENV} is \
                     unset — provide the reference JSONL side-car per \
                     tools/parity/dnsmos_score_reference.py"
                )
            });
            panic!(
                "reference-JSONL comparison harness is a follow-up wave; \
                 CNN forward lit up unexpectedly early — wire the comparison here"
            );
        }
        Err(other) => panic!("dnsmos score returned unexpected error: {other:?}"),
    }
}
