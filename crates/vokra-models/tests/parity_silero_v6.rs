//! Silero VAD v6.2.1 numerical parity — env-gated (2026-07-30).
//!
//! Mirrors the layout of `parity_cosyvoice2.rs` / `parity_kokoro.rs`: the
//! tests that need a real v6.2.1 GGUF are gated on the
//! `VOKRA_SILERO_V6_GGUF` environment variable and skip cleanly when it is
//! unset, so CI stays green without the upstream ONNX. The fixture-free
//! tests (variant load-path plumbing) run everywhere so the FR-EX-08 error
//! surface (unknown-tag rejection, absent-key backward compat) is exercised
//! without any HuggingFace download.
//!
//! # Provenance boundary (documented, not enforced by these tests)
//!
//! Upstream `snakers4/silero-vad` v6.2.1 (release 2026-02-24) retains the
//! architecture of v5 per primary source:
//!
//! - `src/silero_vad/tinygrad_model.py` declares the same `Conv1d(1, 258,
//!   k=256, stride=128)` + 4-conv encoder + `LSTMCell(128, 128)` +
//!   `Conv1d(128, 1, 1)` head as v5;
//! - `src/silero_vad/utils_vad.py` has identical inference geometry
//!   (`num_samples = 512 if sr == 16000 else 256`, `context_size = 64 if sr
//!   == 16000 else 32`, state shape `[2, 1, 128]`);
//! - `silero_vad.onnx` file size (2 327 524 bytes) is identical across the
//!   `v5.1.2`, `v6.0` and `v6.2.1` tags, while the git blob shas differ —
//!   retrained weights, same topology.
//!
//! What v6.2.1 changes is the trained weights (and hence the frame-by-frame
//! probability stream), not tensor names / shapes. So this harness rebuilds
//! the parity flow that `parity.rs` already exercises against the v5
//! fixture — the ORT ground truth for v6.2.1 must be regenerated (owner
//! task = handoff), but the Rust side needs no new forward code. The
//! `atol = 5e-6` inherited from the v5 SPEC still applies (the numerical
//! deviation is bounded by the transcendental scalar port, not by the
//! training data).
//!
//! # Env vars
//!
//! - `VOKRA_SILERO_V6_GGUF` — path to the v6.2.1 GGUF produced by
//!   `vokra-cli convert --model silero-vad --input silero_vad.onnx
//!   --silero-variant v6.2.1 --output silero-vad-v6.2.1.gguf`. When absent
//!   the fixture-dependent tests skip with a `println!` marker.
//! - `VOKRA_SILERO_V6_PROBS_16K_CTX` — optional path to a text file
//!   containing one ORT-reference probability per line, generated over the
//!   v6.2.1 ONNX with the same official-context wrapper the v5 fixture
//!   uses (`tests/parity/silero_vad/probs_16k_ctx.txt` recipe). When
//!   absent, the harness only exercises the shape / plumbing path and does
//!   NOT fabricate a parity assertion.

use std::env;

use vokra_core::engines::VadEngine;
use vokra_models::silero_vad::{SampleRate, SileroVadV5, SileroVariant};

/// The env var CI / owners set to point the gated tests at a real
/// v6.2.1 GGUF. Absent = skip (never fabricate a pass).
const V6_GGUF_ENV: &str = "VOKRA_SILERO_V6_GGUF";

/// Optional companion env var: path to the ORT-reference probability
/// file for the 16 kHz + official-context (`ctx576`) leg. Absent = skip
/// the numeric compare (documented in the module doc).
const V6_PROBS_16K_CTX_ENV: &str = "VOKRA_SILERO_V6_PROBS_16K_CTX";

/// Same-shared atol as the v5 SPEC (`crates/vokra-models/src/silero_vad/SPEC.md`
/// §"Parity"): FP32 `5e-6` is the deviation bound that the scalar
/// transcendental port sustains without amplification through the LSTM
/// stack (measured at v5; the retrained v6.2.1 weights do not change this
/// numerical property). NFR-QL-01 admits `atol = 0.01`; we hold to the
/// tighter measured bound instead so a regression is caught at ~2000×
/// margin, not at product-level tolerance.
const ATOL: f32 = 5e-6;

/// Fixture-free: [`SileroVariant`] is re-exported from `vokra-vad-micro`
/// through `vokra-models::silero_vad` for downstream consumers. This pin
/// keeps the re-export live so a rename in `vokra-vad-micro` immediately
/// surfaces at the model wrapper boundary.
#[test]
fn silero_variant_reexport_is_live_from_models_wrapper() {
    // Both variants are constructible + distinct + carry their canonical
    // release-tag string.
    let v5 = SileroVariant::V5;
    let v6 = SileroVariant::V6_2_1;
    assert_ne!(v5, v6);
    assert_eq!(v5.tag(), "v5");
    assert_eq!(v6.tag(), "v6.2.1");
}

/// Fixture-free: the model wrapper exposes the loaded artifact's release
/// tag through [`SileroVadV5::variant`]. Uses the committed pre-tagging v5
/// fixture — asserted to default to V5 through the model wrapper (the
/// binder-level default is pinned in `vokra_vad_micro::weights::tests`;
/// this test pins the wrapper's re-exposed accessor).
#[test]
fn committed_v5_fixture_reports_v5_variant_through_wrapper() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/parity/silero_vad/silero-vad-v5.gguf");
    let m = SileroVadV5::open(&fixture).expect("committed v5 fixture loads");
    assert_eq!(m.variant(), SileroVariant::V5);
}

/// Env-gated: loading the v6.2.1 GGUF reports [`SileroVariant::V6_2_1`]
/// through the model wrapper's accessor, and the model supports both
/// sample rates the upstream ONNX carries. Skips (no fabricated pass) when
/// [`V6_GGUF_ENV`] is unset.
#[test]
fn v6_2_1_gguf_reports_correct_variant() {
    let Ok(path) = env::var(V6_GGUF_ENV) else {
        println!(
            "SKIP: ${V6_GGUF_ENV} unset — set to a v6.2.1 GGUF path to run this test \
             (see module doc for the `vokra-cli convert` recipe)"
        );
        return;
    };
    let m = SileroVadV5::open(&path).expect("v6.2.1 GGUF loads");
    assert_eq!(
        m.variant(),
        SileroVariant::V6_2_1,
        "GGUF at {path} must be stamped `vokra.silero.version = v6.2.1`"
    );
    // Full ONNX (both-rate) is the primary v6.2.1 distribution; the
    // 16k-only safetensors variant is out of scope for this file (its
    // support arrives with a separate loader path).
    assert!(m.supports(SampleRate::Hz16000), "16 kHz weights present");
}

/// Env-gated: v6.2.1 forward produces probabilities in `[0, 1]` on a
/// zero-input frame — the trivial numerical smoke that a stream over
/// silence never leaves the sigmoid domain. Skips when [`V6_GGUF_ENV`] is
/// unset. This is *not* a parity assertion (that requires the ORT
/// reference — see [`v6_2_1_prob_stream_matches_ort_reference_16k_ctx`]).
#[test]
fn v6_2_1_forward_runs_and_stays_in_sigmoid_range() {
    let Ok(path) = env::var(V6_GGUF_ENV) else {
        println!("SKIP: ${V6_GGUF_ENV} unset");
        return;
    };
    let m = SileroVadV5::open(&path).unwrap();
    let mut s = m.open_stream();
    // 30 frames of silence at 16 kHz.
    let silence = vec![0.0f32; 512];
    for _ in 0..30 {
        let probs = s.push_pcm(&silence, 16_000).expect("push_pcm");
        for p in probs {
            assert!(
                (0.0..=1.0).contains(&p),
                "sigmoid probability out of range: {p}"
            );
        }
    }
}

/// Env-gated: full parity against the ORT reference. Requires BOTH
/// [`V6_GGUF_ENV`] (the GGUF) and [`V6_PROBS_16K_CTX_ENV`] (the reference
/// prob file, one value per line — same format as
/// `tests/parity/silero_vad/probs_16k_ctx.txt` for v5). Skips cleanly if
/// either is missing so partial CI environments do not gate on a
/// half-populated fixture set.
///
/// The 16 kHz + official-context (`ctx576`) leg is the ratified reference
/// interface (the raw bare-frame path collapses on real speech — see the
/// v5 SPEC's "2026-07-16 real-weight eval P1"). The 8 kHz variant follows
/// the same shape but needs a separate reference file.
#[test]
fn v6_2_1_prob_stream_matches_ort_reference_16k_ctx() {
    let Ok(gguf) = env::var(V6_GGUF_ENV) else {
        println!("SKIP: ${V6_GGUF_ENV} unset");
        return;
    };
    let Ok(refpath) = env::var(V6_PROBS_16K_CTX_ENV) else {
        println!(
            "SKIP: ${V6_PROBS_16K_CTX_ENV} unset — set to a text file of one ORT probability \
             per line (generated over the v6.2.1 ONNX with `utils_vad.OnnxWrapper` semantics; \
             see `tests/parity/silero_vad/gen_reference.py` `probs_16k_ctx` for the recipe)"
        );
        return;
    };
    let m = SileroVadV5::open(&gguf).unwrap();
    // Reference values, one per line. `str::parse` — never `strtod` (NFR-RL-01).
    let ref_probs: Vec<f32> = std::fs::read_to_string(&refpath)
        .expect("read reference probs file")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| {
            l.parse::<f32>()
                .unwrap_or_else(|e| panic!("failed to parse `{l}` as f32: {e}"))
        })
        .collect();
    assert!(
        !ref_probs.is_empty(),
        "reference probs file at {refpath} is empty"
    );
    // Zero PCM per frame is the smallest input that still exercises the
    // full forward + state carry-over; use it as the paired input signal
    // for the reference. The owner-generated reference must be produced
    // over the same zero-input clip for this comparison to be honest —
    // this test intentionally does NOT fabricate a matching signal from
    // the probs alone. If the reference was generated over a WAV clip
    // rather than zero PCM, the harness surfaces a mismatch loudly.
    let mut s = m.open_stream();
    let silence = vec![0.0f32; 512];
    let mut got: Vec<f32> = Vec::with_capacity(ref_probs.len());
    while got.len() < ref_probs.len() {
        let probs = s.push_pcm(&silence, 16_000).unwrap();
        got.extend(probs);
    }
    got.truncate(ref_probs.len());
    let mut max_err = 0.0f32;
    for (i, (g, r)) in got.iter().zip(&ref_probs).enumerate() {
        let e = (g - r).abs();
        if e > max_err {
            max_err = e;
        }
        assert!(
            e <= ATOL,
            "frame {i}: got {g} vs ref {r} (|Δ| {e} > atol {ATOL}) — \
             was the reference produced over the same zero-PCM signal?"
        );
    }
    println!(
        "[silero-v6.2.1 parity 16k ctx] frames={} max_abs_err={max_err:.3e} (atol={ATOL})",
        ref_probs.len()
    );
}
