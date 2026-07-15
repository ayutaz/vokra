//! M4-05 T25 — the "MEL loss / UTMOS 劣化 5% 未満" quality gate
//! (NFR-QL-02) with the **G2 branch** fixed in code (ADR M4-05 §D1-(d)).
//!
//! # G2 state at landing (milestones §8 M4-18 行, 2026-07-14 judgement)
//!
//! M4-18 landed the weight-independent UTMOS harness + `AudioMosMetric`
//! wiring, but the real UTMOS weights are **deferred** (kickoff-week gate
//! → v1.0.x patch auto-defer). This WP therefore runs the **defer
//! branch**: the mel-loss half asserts; the UTMOS half is **未判定
//! (advisory)** and is surfaced as such — never folded into a pass.
//! mel-loss alone is *half* of the Kill-switch-I red line, and every
//! report below carries that caveat (`mel_loss_only` /
//! `advisory_only` — fabricated pass 禁止).
//!
//! # What can honestly run today
//!
//! The NFR-QL-02 judgement is "Vokra output vs the *upstream reference*
//! output, same input / same seed" — which needs the T29 real weights on
//! both sides. Until then:
//!
//! - `gate_plumbing_*` legs exercise the full gate machinery
//!   (mel half + the G2 GO-branch code path with a **synthesized** UTMOS
//!   scorer in the advisory domain) over the deterministic fixture
//!   engine's output — proving the wiring, claiming **no** upstream
//!   quality;
//! - the reference leg is env-gated exactly like `parity_csm.rs` and
//!   fails loudly (naming T29) if fixtures are supplied before the weight
//!   binding exists.

use vokra_core::{DialogRequest, S2sEngine};
use vokra_eval::degradation::{MosDomain, check_degradation, check_degradation_with_utmos};
use vokra_eval::metrics::utmos::{ConvActivation, HeadPool, TransformerNorm, Utmos, UtmosConfig};
use vokra_models::csm::{CsmEngine, EchoPath};

/// NFR-QL-02: 劣化 5% 未満.
const THRESHOLD: f64 = 0.05;

fn fixture_output() -> (Vec<f32>, u32) {
    let engine = CsmEngine::synthesized_fixture(55)
        .expect("fixture engine")
        .with_echo_path(EchoPath::BypassRecordedInput);
    let turn = engine
        .dialog(
            &DialogRequest::new("quality gate plumbing")
                .deterministic()
                .with_max_frames(24),
        )
        .expect("dialog");
    let audio = turn.audio.expect("audio");
    (audio.samples, audio.sample_rate)
}

#[test]
fn gate_plumbing_mel_half_runs_and_surfaces_the_defer_caveat() {
    // Defer branch (G2): mel-loss half only. Self-reference is a plumbing
    // exercise (delta = 0 by construction), NOT an upstream quality claim
    // — the honest signal here is that `mel_loss_only` is surfaced and the
    // caveat is printed for the CI summary.
    let (pcm, sr) = fixture_output();
    let report = check_degradation(&pcm, &pcm, sr, THRESHOLD).expect("gate runs");
    assert!(report.passes_5pct_gate, "self-reference delta is 0");
    assert!(
        report.mel_loss_only,
        "the defer branch must surface that UTMOS did not run"
    );
    assert!(report.utmos.is_none());
    println!(
        "csm quality gate (G2 defer branch): mel_loss half ran \
         (relative_delta = {:.6}); UTMOS = 未判定 (advisory, pending the M4-18 \
         weight arrival / v1.0.x patch — milestones §8 M4-18 行 2026-07-14 判定). \
         mel_loss alone is half of the Kill-switch-I red line; the upstream-\
         reference judgement is the T29 flip-the-switch.",
        report.relative_delta
    );
}

#[test]
fn gate_go_branch_code_path_runs_with_a_synthesized_scorer_as_advisory() {
    // The G2 GO branch (M4-18 weights arrive → UTMOS gates): the code path
    // is exercised today with the **synthesized** UTMOS scorer in the
    // **advisory** domain — the assessment is computed and surfaced but
    // never gates (synthesized weights carry no upstream MOS semantics;
    // NFR-QL-04 out-of-distribution posture). Flipping to the gating
    // branch = real `vokra.utmos.*` GGUF + `MosDomain::TtsSynthesis`.
    let (pcm, sr) = fixture_output();
    // A tiny UTMOS skeleton at the fixture rate, its affine shifted into a
    // MOS-like positive band (the M4-18 degradation-test recipe).
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
    // CSM output is Mimi-codec streaming audio — the M4-18 domain taxonomy
    // marks it advisory-only until the owner-side correlation study
    // (MosDomain::CodecStreaming; exactly this WP's case).
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
        "csm quality gate (G2 GO-branch plumbing): synthesized-UTMOS advisory \
         assessment ran (score_ref = {:.4}); gating awaits the real M4-18 weights.",
        mos.score_ref
    );
}

#[test]
fn upstream_reference_quality_gate_is_env_gated_until_t29() {
    let Some(dir) = std::env::var_os("VOKRA_CSM_PARITY_DIR") else {
        println!(
            "skip: VOKRA_CSM_PARITY_DIR not set — the NFR-QL-02 judgement against \
             the upstream reference audio needs the T29 owner dump + weight \
             binding (clean gated skip, not a pass)"
        );
        return;
    };
    panic!(
        "VOKRA_CSM_PARITY_DIR = {dir:?} is set, but the real-weight GGUF binding \
         is still the T29 stub — the upstream-reference NFR-QL-02 comparison \
         cannot run yet. Land the T29 binding, then replace this panic with: \
         decode the reference decode_pcm.f32, generate the Vokra turn with the \
         same context/seed, and assert check_degradation(vokra, reference) \
         relative_delta < {THRESHOLD} (+ UTMOS per the M4-18 gate state). \
         Refusing to report a pass that did not run."
    );
}
