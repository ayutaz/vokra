//! M4-06 T27 — the quality-judgement wiring (mel_loss + UTMOS) with the
//! **M4-18 G2 defer branch** fixed in code (ADR M4-06 §D1-(f); the
//! csm_quality_gate.rs discipline verbatim).
//!
//! # Judgement state (honest — fabricated pass 禁止)
//!
//! - milestones §8 M4-06 行 carries **no** NFR-QL-01/02 in its完了条件
//!   (the M4-05 row does) while the M4-18 row says both M4-05/M4-06
//!   require the 5% gate — the spec flags this asymmetry as an **owner
//!   confirmation item**; this file implements the stricter reading.
//! - M4-18 = defer branch (weights deferred to a v1.0.x patch): the
//!   mel-loss half asserts, the UTMOS half surfaces as 未判定 (advisory).
//!   mel_loss alone is half of the Kill-switch-I red line — every report
//!   says so.
//! - The upstream-reference judgement additionally needs the **shared
//!   Mimi module's real weight binding** (`MimiEncoder::from_gguf` is the
//!   M4-05 T29-gated stub): Moshi's LM weights bind for real today, but
//!   its PCM ends ride the synthesized bridge, so a PCM-level quality
//!   comparison against upstream audio would be meaningless — the
//!   env-gated leg says exactly that instead of pretending.

use vokra_core::{DialogRequest, S2sEngine};
use vokra_eval::degradation::{MosDomain, check_degradation, check_degradation_with_utmos};
use vokra_eval::metrics::utmos::{ConvActivation, HeadPool, TransformerNorm, Utmos, UtmosConfig};
use vokra_models::csm::EchoPath;
use vokra_models::moshi::MoshiEngine;

/// NFR-QL-02: 劣化 5% 未満.
const THRESHOLD: f64 = 0.05;

fn fixture_output() -> (Vec<f32>, u32) {
    let engine = MoshiEngine::synthesized_fixture(55)
        .expect("fixture engine")
        .with_echo_path(EchoPath::BypassRecordedInput);
    let hop = engine.mimi_config().frame_hop_samples().expect("hop");
    let input: Vec<f32> = (0..hop * 12)
        .map(|i| ((i as f32) * 0.03).sin() * 0.25)
        .collect();
    let turn = engine
        .dialog(
            &DialogRequest::new("")
                .with_input_audio(input)
                .deterministic(),
        )
        .expect("dialog");
    let audio = turn.audio.expect("audio");
    (audio.samples, audio.sample_rate)
}

#[test]
fn gate_plumbing_mel_half_runs_and_surfaces_the_defer_caveat() {
    let (pcm, sr) = fixture_output();
    let report = check_degradation(&pcm, &pcm, sr, THRESHOLD).expect("gate runs");
    assert!(report.passes_5pct_gate, "self-reference delta is 0");
    assert!(
        report.mel_loss_only,
        "the defer branch must surface that UTMOS did not run"
    );
    assert!(report.utmos.is_none());
    println!(
        "moshi quality gate (G2 defer branch): mel_loss half ran \
         (relative_delta = {:.6}); UTMOS = 未判定 (advisory, pending the M4-18 \
         weight arrival / v1.0.x patch). mel_loss alone is half of the \
         Kill-switch-I red line; the upstream-reference judgement is the T29 \
         flip-the-switch (weights + the shared Mimi binding).",
        report.relative_delta
    );
}

#[test]
fn gate_go_branch_code_path_runs_with_a_synthesized_scorer_as_advisory() {
    // The G2 GO branch exercised with a synthesized scorer in the
    // advisory domain (Moshi output = Mimi-codec streaming audio →
    // MosDomain::CodecStreaming, advisory-only until the owner-side
    // correlation study — the M4-18 taxonomy).
    let (pcm, sr) = fixture_output();
    let config = UtmosConfig {
        sample_rate: sr,
        conv_channels: vec![4, 6],
        conv_kernels: vec![5, 3],
        conv_strides: vec![3, 2],
        conv_activation: ConvActivation::Gelu,
        n_layer: 1,
        n_head: 2,
        hidden_dim: 6,
        ffn_dim: 12,
        norm: TransformerNorm::Post,
        ln_eps: 1e-5,
        head_dims: vec![4, 1],
        head_pool: HeadPool::MeanAfter,
        head_scale: 1.0,
        head_offset: 3.0,
    };
    let scorer = Utmos::synthesized(config, 5).expect("synthesized scorer");
    let report = check_degradation_with_utmos(
        &pcm,
        &pcm,
        sr,
        THRESHOLD,
        &scorer,
        MosDomain::CodecStreaming,
    )
    .expect("gate runs");
    assert!(!report.mel_loss_only, "the UTMOS half ran");
    let mos = report.utmos.expect("assessment present");
    assert!(mos.advisory_only, "synthesized scorer must never gate");
    assert!(
        report.passes_5pct_gate,
        "advisory assessment must not flip the mel verdict"
    );
    println!(
        "moshi quality gate (G2 GO-branch plumbing): synthesized-UTMOS advisory \
         assessment ran (score_ref = {:.4}); gating awaits the real M4-18 weights.",
        mos.score_ref
    );
}

#[test]
fn upstream_reference_quality_gate_is_env_gated_until_mimi_binding() {
    let Some(dir) = std::env::var_os("VOKRA_MOSHI_PARITY_DIR") else {
        println!(
            "skip: VOKRA_MOSHI_PARITY_DIR not set — the NFR-QL-02 judgement against \
             upstream reference audio needs the T29 owner dump + the shared Mimi \
             module's real weight binding (clean gated skip, not a pass)"
        );
        return;
    };
    // Token-level parity CAN run with the fixtures (tests/parity_moshi.rs
    // — the Moshi LM binds for real); the PCM-level quality judgement
    // cannot until the shared Mimi ends bind real weights.
    panic!(
        "VOKRA_MOSHI_PARITY_DIR = {dir:?} is set. Token-level staged parity runs \
         in tests/parity_moshi.rs (real LM binding landed with the T02 manifest), \
         but the PCM-level NFR-QL-02 comparison additionally needs the shared \
         Mimi module's real weight binding (`MimiEncoder::from_gguf` /\
         `MimiNeuralDecoder::from_gguf` are the M4-05 T29-gated stubs). Land that \
         binding, then replace this panic with: decode the reference PCM, run the \
         Vokra turn with the same input/seed, and assert \
         check_degradation(vokra, reference) relative_delta < {THRESHOLD} (+ UTMOS \
         per the M4-18 gate state). Refusing to report a pass that did not run."
    );
}
