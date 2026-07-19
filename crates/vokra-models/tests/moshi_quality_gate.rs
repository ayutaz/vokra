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
//! - The upstream-reference judgement runs **for real** now (cc-16): the
//!   real-Mimi mapping landed with the standalone converter (`vokra-cli
//!   convert --model mimi`, campaign ebe1cc5 — encode 4384/4384 codes vs
//!   upstream, decode max |Δ| 3.67e-6) and the engine binds it through
//!   [`vokra_models::moshi::MoshiEngine::with_mimi_gguf`]. The env-gated
//!   legs below need BOTH `VOKRA_MOSHI_PARITY_DIR` (T29 dump +
//!   `model.gguf`) and `VOKRA_MIMI_GGUF` (the converted codec); a
//!   half-configured run still refuses loudly instead of pretending.

use std::path::{Path, PathBuf};

use vokra_core::{DialogRequest, S2sEngine};
use vokra_eval::degradation::{MosDomain, check_degradation, check_degradation_with_utmos};
use vokra_eval::metrics::utmos::{
    ArchVariant, ConvActivation, HeadPool, TransformerNorm, Utmos, UtmosConfig,
};
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
        // The M4-18 weight-independent skeleton: `V0` carries no upstream
        // UTMOS22 stack, so the `v1` spec is absent by construction.
        variant: ArchVariant::V0,
        v1: None,
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

// ---------------------------------------------------------------------------
// Env-gated real legs (owner weights): PCM-level NFR-QL-02 + duplex peak
// ---------------------------------------------------------------------------

fn read_u32s(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{}: not u32-aligned", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn json_usize(v: &vokra_core::json::JsonValue, key: &str) -> usize {
    match v {
        vokra_core::json::JsonValue::Object(map) => map
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| match v {
                vokra_core::json::JsonValue::Int(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or_else(|| panic!("context.json: missing numeric `{key}`")),
        _ => panic!("context.json: not an object"),
    }
}

/// Resolves both gated inputs, refusing loudly on a half-configured run
/// (`VOKRA_MOSHI_PARITY_DIR` without `VOKRA_MIMI_GGUF`): the binding
/// exists now, so the only honest states are "fully off" (clean skip)
/// and "fully on" (the judgement actually runs).
fn gated_inputs(test: &str) -> Option<(PathBuf, PathBuf)> {
    let Some(dir) = std::env::var_os("VOKRA_MOSHI_PARITY_DIR") else {
        println!(
            "skip ({test}): VOKRA_MOSHI_PARITY_DIR not set — the real judgement \
             needs the T29 owner dump + converted model.gguf (clean gated skip, \
             not a pass)"
        );
        return None;
    };
    let dir = PathBuf::from(dir);
    let Some(mimi) = std::env::var_os("VOKRA_MIMI_GGUF") else {
        panic!(
            "VOKRA_MOSHI_PARITY_DIR = {} is set but VOKRA_MIMI_GGUF is not. The \
             real-Mimi binding LANDED (MoshiEngine::with_mimi_gguf) — convert the \
             kyutai codec checkpoint (`vokra-cli convert --model mimi --input \
             tokenizer-e351c8d8-checkpoint125.safetensors --output mimi.gguf`) and \
             point VOKRA_MIMI_GGUF at it. Refusing to half-run the PCM judgement \
             on the synthesized bridge (fabricated pass 禁止).",
            dir.display()
        );
    };
    let gguf = dir.join("model.gguf");
    assert!(
        gguf.exists(),
        "{} is missing — convert the T29 checkpoint into the fixture dir",
        gguf.display()
    );
    Some((dir, PathBuf::from(mimi)))
}

/// Builds the real-Mimi decode surface from the converted side-car,
/// clipped to `dep_q` (the engine's own `bind_real_mimi` clip semantics,
/// exercised here through the public standalone surfaces the converter
/// contract pins: effective tables + projection-less decoder).
fn real_mimi_decode(mimi_path: &Path, dep_q: usize, codes: &[u32]) -> (Vec<f32>, u32) {
    use vokra_core::gguf::GgufFile;
    use vokra_models::codec::MimiCodecGguf;
    use vokra_models::mimi::{MimiNeuralConfig, MimiNeuralDecoder};
    use vokra_ops::{MimiRvqAttrs, mimi_rvq_decode};

    let file = GgufFile::open(mimi_path).expect("open VOKRA_MIMI_GGUF");
    let mut cfg = MimiNeuralConfig::from_gguf(&file).expect("vokra.mimi.* config");
    cfg.validate().expect("side-car config validates");
    assert!(
        cfg.quantizer.n_q >= dep_q,
        "side-car carries {} codebooks < dep_q {dep_q}",
        cfg.quantizer.n_q
    );
    cfg.quantizer.n_q = dep_q; // set_num_codebooks(dep_q) — get_mimi
    let dec = MimiNeuralDecoder::from_gguf(&file, &cfg).expect("decoder binds real weights");
    assert_eq!(
        dec.expected_feature_dim(),
        cfg.seanet.dimension,
        "standalone converter GGUFs decode on the effective-table path"
    );
    let codec = MimiCodecGguf::from_gguf(&file).expect("effective tables");
    let attrs = MimiRvqAttrs {
        n_codebooks: dep_q,
        codebook_size: cfg.quantizer.bins,
        d_model: cfg.seanet.dimension,
    };
    assert_eq!(codes.len() % dep_q, 0);
    let n_frames = codes.len() / dep_q;
    let features =
        mimi_rvq_decode(codes, n_frames, &codec.tables[..dep_q], &attrs).expect("codes → features");
    let pcm = dec.decode_all(&features).expect("features → PCM");
    (pcm, cfg.sample_rate)
}

/// The PCM-level NFR-QL-02 judgement, live (the panic this replaces said:
/// "decode the reference PCM, run the Vokra turn with the same input/seed,
/// and assert check_degradation(vokra, reference) relative_delta <
/// THRESHOLD (+ UTMOS per the M4-18 gate state)"):
///
/// - reference PCM = the T29 dump's upstream `frame_codes` decoded through
///   the REAL Mimi codec;
/// - Vokra PCM = the full greedy generation re-run over the dump's
///   `user_codes` (the same input the reference consumed — the
///   `parity_moshi.rs` Stage-B loop), decoded through the same real codec.
///
/// Any generation drift OR decode-chain inconsistency moves the mel loss;
/// the UTMOS half stays 未判定 (M4-18 G2 defer branch — advisory).
#[test]
fn upstream_reference_pcm_quality_gate_runs_with_real_mimi() {
    let Some((dir, mimi_path)) = gated_inputs("pcm-quality-gate") else {
        return;
    };
    use vokra_models::moshi::{
        MoshiConfig, MoshiFrameOut, MoshiGenerationState, MoshiModel, MoshiSamplerPair,
    };

    // Mapped-lazy LM load (bounded memory — the cc-06 path).
    let gguf_path = dir.join("model.gguf");
    let file = std::sync::Arc::new(vokra_mmap::open_gguf(&gguf_path).expect("mmap model.gguf"));
    let cfg = MoshiConfig::from_gguf(&file).expect("vokra.moshi.* chunk group");
    cfg.validate_for_forward().expect("hparams populated");
    let head =
        vokra_models::moshi::MoshiBackboneWeights::head_from_gguf(&file, &cfg).expect("head bind");
    let mapped =
        vokra_models::moshi::MappedTemporalBlocks::bind(std::sync::Arc::clone(&file), &cfg)
            .expect("mapped blocks");
    let backbone = vokra_models::moshi::MoshiBackbone::new_mapped(cfg.clone(), head, mapped)
        .expect("backbone");
    let depth_w =
        vokra_models::moshi::MoshiDepthWeights::from_gguf(&file, &cfg).expect("depth bind");
    let depth =
        vokra_models::moshi::MoshiDepthTransformer::new(cfg.clone(), depth_w).expect("depformer");
    let model = MoshiModel::from_parts(backbone, depth).expect("model");

    let ctx = vokra_core::json::parse(&std::fs::read(dir.join("context.json")).unwrap())
        .expect("context.json");
    let n_steps = json_usize(&ctx, "n_steps");
    let n_user = json_usize(&ctx, "n_user");
    let dep_q = json_usize(&ctx, "dep_q");
    assert_eq!(dep_q, cfg.dep_q, "fixture/config dep_q");
    let user = read_u32s(&dir.join("user_codes.u32"));
    let ref_codes = read_u32s(&dir.join("frame_codes.u32"));
    assert_eq!(user.len(), n_steps * n_user);

    // The Vokra turn: same input (user codes), same greedy contract.
    let mut state = MoshiGenerationState::new(&cfg).unwrap();
    let mut samplers = MoshiSamplerPair::greedy();
    let mut out = MoshiFrameOut::new(&cfg);
    let mut got_codes = Vec::new();
    for s in 0..n_steps {
        let row = &user[s * n_user..(s + 1) * n_user];
        if model
            .step_into(&mut state, row, &mut samplers, &mut out)
            .expect("full step")
        {
            got_codes.extend_from_slice(&out.audio);
        }
    }
    assert_eq!(
        got_codes.len(),
        ref_codes.len(),
        "emitted frame count drifted from the reference dump"
    );

    // Both code streams through the REAL codec → PCM; judge NFR-QL-02.
    let (vokra_pcm, sr) = real_mimi_decode(&mimi_path, dep_q, &got_codes);
    let (ref_pcm, _) = real_mimi_decode(&mimi_path, dep_q, &ref_codes);
    let report =
        check_degradation(&vokra_pcm, &ref_pcm, sr, THRESHOLD).expect("degradation gate runs");
    assert!(
        report.passes_5pct_gate,
        "NFR-QL-02: PCM degradation vs the upstream-reference codes exceeded \
         {THRESHOLD} (relative_delta = {})",
        report.relative_delta
    );
    assert!(
        report.mel_loss_only,
        "UTMOS stays 未判定 (M4-18 G2 defer branch)"
    );
    println!(
        "moshi PCM quality gate (REAL Mimi): {} frames, relative_delta = {:.6} \
         (< {THRESHOLD}), codes bit-exact = {}; UTMOS half = 未判定 (M4-18 \
         advisory, v1.0.x patch)",
        got_codes.len() / dep_q,
        report.relative_delta,
        got_codes == ref_codes
    );
}

/// Engine-level duplex re-measure with the REAL codec (the audit's
/// follow-up on the synthesized-bridge peak 1.412): truncated Moshi LM +
/// real Mimi side-car, deterministic 5-frame turn — finite, bounded PCM
/// with the peak/RMS printed for the record. The amplitude bound mirrors
/// the standalone roundtrip's sanity bound (|x| < 4.0): a truncated
/// 2-layer LM emits gibberish codes, so this leg pins *codec* health, not
/// speech quality (that is the full-7B owner leg).
#[test]
fn real_mimi_duplex_turn_is_finite_and_peak_is_recorded() {
    let Some((dir, mimi_path)) = gated_inputs("duplex-peak") else {
        return;
    };
    let engine = vokra_models::moshi::MoshiEngine::from_path(dir.join("model.gguf"))
        .expect("truncated Moshi loads")
        .with_mimi_gguf(&mimi_path)
        .expect("real Mimi side-car binds")
        .with_echo_path(EchoPath::BypassRecordedInput);
    assert!(!engine.mimi_is_synthesized(), "real codec is live");
    let hop = engine.mimi_config().frame_hop_samples().expect("hop");
    let sr = engine.mimi_config().sample_rate;
    let input: Vec<f32> = (0..hop * 5)
        .map(|i| {
            let t = i as f32 / sr as f32;
            0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
        })
        .collect();
    let turn = engine
        .dialog(
            &DialogRequest::new("")
                .with_input_audio(input)
                .deterministic(),
        )
        .expect("duplex dialog");
    let audio = turn.audio.expect("audio");
    assert_eq!(audio.sample_rate, sr);
    assert!(audio.samples.iter().all(|v| v.is_finite()));
    let peak = audio.samples.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let rms = (audio
        .samples
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / audio.samples.len().max(1) as f64)
        .sqrt();
    assert!(
        peak < 4.0,
        "decoded PCM amplitude sanity (mirrors real_mimi_roundtrip): peak = {peak}"
    );
    println!(
        "real-mimi duplex re-measure: {} samples @ {sr} Hz, peak = {peak:.4}, \
         rms = {rms:.4} (synthesized-bridge audit peak was 1.412; truncated-LM \
         codes are gibberish, so this pins codec health — full-7B speech quality \
         is the owner leg)",
        audio.samples.len()
    );
}
