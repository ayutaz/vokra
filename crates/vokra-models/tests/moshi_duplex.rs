//! M4-06 T16/T17/T18/T21 — the Moshi full-duplex session: push/pull
//! pipeline, AEC wiring with a synthetic echo path, barge-in semantics,
//! and the AEC-required posture (three FR-EX-08 negative bands).
//!
//! Everything here runs on the synthesized fixture (tiny dims, seed
//! deterministic — no real weights, honest about it); the real-checkpoint
//! behaviour is the T25/T29 flip-the-switch (`tests/parity_moshi.rs`).

use std::sync::Arc;

use vokra_core::{DuplexSessionConfig, S2sDuplexEngine, S2sDuplexHandle, S2sEngine, VokraError};
use vokra_models::moshi::MoshiEngine;
use vokra_ops::aec::AecAttrs;

fn aec_attrs(engine: &MoshiEngine) -> AecAttrs {
    AecAttrs {
        sample_rate: engine.mimi_config().sample_rate,
        frame_size: 8, // divides the tiny Mimi hop (8)
        filter_length: 32,
    }
}

fn engine_with_aec(seed: u64) -> Arc<MoshiEngine> {
    let engine = MoshiEngine::synthesized_fixture(seed).expect("fixture engine");
    let attrs = aec_attrs(&engine);
    Arc::new(engine.with_aec(&attrs, 16_000).expect("aec recipe"))
}

/// Deterministic pseudo-speech mic frame.
fn mic_frame(hop: usize, step: usize) -> Vec<f32> {
    (0..hop)
        .map(|i| {
            let t = (step * hop + i) as f32;
            (t * 0.11).sin() * 0.3 + (t * 0.031).sin() * 0.2
        })
        .collect()
}

// ---------------------------------------------------------------------------
// T16 — session API: continuous push/pull, determinism, facade wiring
// ---------------------------------------------------------------------------

#[test]
fn duplex_pipeline_streams_end_to_end_and_is_deterministic() {
    let run = || {
        let engine = engine_with_aec(11);
        let mut s = engine
            .open_duplex_session(&DuplexSessionConfig::new().deterministic())
            .expect("session");
        let hop = s.frame_hop();
        let mut pulled = Vec::new();
        for step in 0..8 {
            let report = s.push_mic_frame(&mic_frame(hop, step)).expect("push");
            assert!(report.aec_applied, "default posture runs the canceller");
            while let Some(frame) = s.pull_model_frame().expect("pull") {
                assert_eq!(frame.len(), hop, "model frames are hop-sized");
                assert!(frame.iter().all(|v| v.is_finite()));
                pulled.push(frame);
            }
        }
        (pulled, s.monologue_text().expect("monologue"))
    };
    let (a_frames, a_text) = run();
    let (b_frames, b_text) = run();
    // Warmup (max_delay = 1) + the first-frame double step: the first
    // push already emits, so 8 pushes → 8 frames.
    assert_eq!(
        a_frames.len(),
        8,
        "one frame per push after the double-step"
    );
    assert_eq!(a_frames, b_frames, "deterministic mode reproduces audio");
    assert_eq!(
        a_text, b_text,
        "deterministic mode reproduces the monologue"
    );
}

#[test]
fn facade_duplex_entry_reaches_the_engine() {
    // Session injection (vokra-core) → S2s::duplex() → a live handle.
    let engine = engine_with_aec(5);
    let dir = std::env::temp_dir().join(format!("vokra-moshi-facade-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("stub.gguf");
    let mut b = vokra_core::gguf::GgufBuilder::new();
    b.add_string("vokra.model.arch", "moshi");
    std::fs::write(&model_path, b.to_bytes().unwrap()).unwrap();
    let session = vokra_core::Session::from_file(&model_path)
        .build()
        .expect("session")
        .with_s2s_duplex_engine(engine.clone() as Arc<dyn S2sDuplexEngine>);
    let mut handle = session
        .s2s()
        .duplex_with(&DuplexSessionConfig::new().deterministic())
        .expect("facade opens a duplex handle");
    let hop = handle.frame_hop();
    assert_eq!(handle.sample_rate(), engine.mimi_config().sample_rate);
    let report = handle.push_mic_frame(&mic_frame(hop, 0)).expect("push");
    assert!(report.step_emitted, "first push emits (double-step warmup)");
    assert!(handle.pull_model_frame().expect("pull").is_some());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn monologue_text_accumulates_through_the_display_rule() {
    let engine = engine_with_aec(23);
    let mut s = engine
        .open_duplex_session(&DuplexSessionConfig::new().deterministic())
        .expect("session");
    let hop = s.frame_hop();
    for step in 0..6 {
        s.push_mic_frame(&mic_frame(hop, step)).expect("push");
    }
    let tokens = s.monologue_tokens().to_vec();
    assert_eq!(tokens.len(), 6, "one text token per emitted frame");
    let text = s.monologue_text().expect("decode");
    // The fixture tokenizer renders " t{id}"; pad ids (0 / 3) are hidden.
    let cfg = engine.config();
    let expect: String = tokens
        .iter()
        .filter(|&&t| t != cfg.text_pad_id && t != cfg.text_end_pad_id)
        .map(|t| format!(" t{t}"))
        .collect();
    assert_eq!(text, expect, "display rule = skip pads, concat pieces");
}

#[test]
fn wrong_frame_size_is_a_loud_error() {
    let engine = engine_with_aec(3);
    let mut s = engine
        .open_duplex_session(&DuplexSessionConfig::new())
        .expect("session");
    let hop = s.frame_hop();
    let err = s.push_mic_frame(&vec![0.0; hop + 1]).unwrap_err();
    assert!(matches!(err, VokraError::InvalidArgument(_)));
    assert!(
        err.to_string().contains("whole frames"),
        "actionable: {err}"
    );
}

// ---------------------------------------------------------------------------
// T17 — AEC wiring: synthetic echo attenuation + reference clock
// ---------------------------------------------------------------------------

#[test]
fn synthetic_echo_is_attenuated_in_the_erle_direction() {
    // Echo-only mic: each pushed frame is an attenuated copy of the model
    // frame pulled on the *previous* cycle — a one-frame acoustic loop
    // (pull → speaker → mic → next push). That playback latency is
    // exactly what `playback_offset_samples` compensates (T17): stamping
    // the reference one hop late aligns it with the mic clock where the
    // echo actually lands. After adaptation the cleaned residual must
    // drop well below the raw echo — the csm::aec_front direction assert
    // through the whole duplex session. Real acoustics stay T30 (owner).
    let mut cfg = vokra_models::moshi::MoshiConfig::tiny_for_tests();
    cfg.max_ctx = 1024; // 400 echo steps + the double-step, within budget
    cfg.context = 1024;
    let engine = MoshiEngine::synthesized_with_config(cfg, 7).expect("engine");
    let attrs = aec_attrs(&engine);
    let engine = Arc::new(engine.with_aec(&attrs, 16_000).expect("aec"));
    let hop = engine.mimi_config().frame_hop_samples().expect("hop");
    let mut s = engine
        .open_duplex_session(
            &DuplexSessionConfig::new()
                .deterministic()
                .with_playback_offset_samples(hop as u64),
        )
        .expect("session");
    let atten = 0.7f32;
    let mut echo: Vec<f32> = vec![0.0; hop];
    let mut raw_tail = 0.0f64;
    let mut cleaned_tail = 0.0f64;
    let total = 400usize;
    for step in 0..total {
        let report = s.push_mic_frame(&echo).expect("push");
        if step >= total * 3 / 4 {
            raw_tail += f64::from(report.raw_rms) * f64::from(report.raw_rms);
            cleaned_tail += f64::from(report.cleaned_rms) * f64::from(report.cleaned_rms);
        }
        if let Some(frame) = s.pull_model_frame().expect("pull") {
            for (e, v) in echo.iter_mut().zip(frame.iter()) {
                *e = v * atten;
            }
        }
    }
    assert!(
        raw_tail > 1e-12,
        "precondition: the synthesized model output must produce a non-silent \
         echo (raw tail energy {raw_tail:.3e}) — otherwise this test is vacuous"
    );
    assert!(
        cleaned_tail < raw_tail * 0.5,
        "the canceller must attenuate the pure echo (raw {raw_tail:.6e}, \
         cleaned {cleaned_tail:.6e})"
    );
}

#[test]
fn playback_offset_shifts_the_reference_clock_without_breaking_monotonicity() {
    // The owner-tunable latency compensation (T17 (提案)): a non-zero
    // offset must keep the far-end tags monotone (the writer rejects
    // regressions loudly) across many pull cycles.
    let engine = engine_with_aec(9);
    let mut s = engine
        .open_duplex_session(
            &DuplexSessionConfig::new()
                .deterministic()
                .with_playback_offset_samples(24),
        )
        .expect("session");
    let hop = s.frame_hop();
    for step in 0..12 {
        s.push_mic_frame(&mic_frame(hop, step)).expect("push");
        while s
            .pull_model_frame()
            .expect("monotone tags never error")
            .is_some()
        {}
    }
}

// ---------------------------------------------------------------------------
// T18 — barge-in: flush + reset ≡ fresh session, cross-thread handle
// ---------------------------------------------------------------------------

#[test]
fn interrupt_flushes_pending_output_and_resets_to_a_fresh_session() {
    // The bypass (explicitly opted-in) keeps the cleaned input identical
    // across the reset, so post-reset generation must be bit-identical to
    // a brand-new session (module-doc scoping: with AEC enabled the
    // canceller's adaptive state intentionally survives).
    let engine = engine_with_aec(13);
    let opts = DuplexSessionConfig::new()
        .deterministic()
        .with_aec_disabled_explicitly();
    let mut s = engine.open_duplex_session(&opts).expect("session");
    let hop = s.frame_hop();

    // Dirty the session, leaving frames pending (pushed, never pulled).
    for step in 0..4 {
        s.push_mic_frame(&mic_frame(hop, step + 40)).expect("push");
    }
    assert!(
        s.pending_frames() > 0,
        "output is pending before the interrupt"
    );
    assert!(!s.monologue_tokens().is_empty());

    // Cross-thread barge-in (the handle is Send + Clone).
    let handle = s.interrupt_handle();
    std::thread::spawn(move || handle.interrupt())
        .join()
        .expect("interrupt thread");

    // The next boundary acknowledges: flush + reset, mic intake continues.
    assert!(
        s.pull_model_frame().expect("pull").is_none(),
        "interrupt flushes the pending model frames"
    );
    assert_eq!(s.monologue_tokens().len(), 0, "monologue resets");

    // Same inputs from here ≡ a fresh session, bit for bit.
    let mut fresh = engine.open_duplex_session(&opts).expect("fresh session");
    let mut replay = Vec::new();
    let mut expect = Vec::new();
    for step in 0..5 {
        let frame = mic_frame(hop, step);
        s.push_mic_frame(&frame).expect("push");
        fresh.push_mic_frame(&frame).expect("push");
        while let Some(f) = s.pull_model_frame().expect("pull") {
            replay.push(f);
        }
        while let Some(f) = fresh.pull_model_frame().expect("pull") {
            expect.push(f);
        }
    }
    assert_eq!(replay, expect, "post-interrupt generation ≡ fresh session");
    assert_eq!(
        s.monologue_text().unwrap(),
        fresh.monologue_text().unwrap(),
        "monologue restarts identically"
    );
}

// ---------------------------------------------------------------------------
// T21 — AEC-required posture: three FR-EX-08 negative bands
// ---------------------------------------------------------------------------

#[test]
fn band_a_missing_aec_without_opt_in_is_a_loud_error() {
    let engine = Arc::new(MoshiEngine::synthesized_fixture(1).expect("engine"));
    let err = engine
        .open_duplex_session(&DuplexSessionConfig::new())
        .unwrap_err();
    assert!(matches!(err, VokraError::InvalidArgument(_)));
    let msg = err.to_string();
    assert!(msg.contains("with_aec"), "names the wiring fix: {msg}");
    assert!(
        msg.contains("with_aec_disabled_explicitly"),
        "names the explicit opt-in: {msg}"
    );
}

#[test]
fn band_b_explicit_opt_out_records_a_citable_warning() {
    let engine = Arc::new(MoshiEngine::synthesized_fixture(1).expect("engine"));
    let s = engine
        .open_duplex_session(&DuplexSessionConfig::new().with_aec_disabled_explicitly())
        .expect("opt-out session opens");
    let warnings = s.warnings();
    assert_eq!(warnings.len(), 1, "exactly one posture warning");
    assert!(
        warnings[0].contains("自己エコーで即崩壊"),
        "cites the collapse rationale (レビュアー C 指摘 #3): {}",
        warnings[0]
    );
}

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
#[test]
fn band_c_off_feature_metal_backend_is_backend_unavailable() {
    let err = vokra_models::moshi::gpu_backend_probe(vokra_core::BackendKind::Metal).unwrap_err();
    assert!(
        matches!(err, VokraError::BackendUnavailable(_)),
        "off-feature Metal must be BackendUnavailable, got {err:?}"
    );
}

#[cfg(not(all(feature = "cuda", any(unix, windows))))]
#[test]
fn band_c_off_feature_cuda_backend_is_backend_unavailable() {
    let err = vokra_models::moshi::gpu_backend_probe(vokra_core::BackendKind::Cuda).unwrap_err();
    assert!(
        matches!(err, VokraError::BackendUnavailable(_)),
        "off-feature CUDA must be BackendUnavailable, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// T19 — the batch S2sEngine face (dialog over the duplex pipeline)
// ---------------------------------------------------------------------------

#[test]
fn dialog_speaks_over_the_duplex_pipeline_and_returns_the_monologue() {
    let engine = engine_with_aec(17);
    let hop = engine.mimi_config().frame_hop_samples().unwrap();
    let input: Vec<f32> = (0..hop * 6)
        .map(|i| ((i as f32) * 0.07).sin() * 0.2)
        .collect();
    let request = vokra_core::DialogRequest::new("")
        .with_input_audio(input)
        .deterministic();
    let turn = engine.dialog(&request).expect("dialog");
    let audio = turn.audio.expect("audio");
    assert_eq!(audio.sample_rate, engine.mimi_config().sample_rate);
    assert_eq!(audio.samples.len() % hop, 0, "whole frames");
    assert!(!audio.samples.is_empty());
    // The reply text is the model's own transcript (may legitimately be
    // empty if every sampled token is a pad id — assert the *contract*,
    // not the content: it decoded without error and matches a rerun).
    let rerun = engine.dialog(&request).expect("dialog rerun");
    assert_eq!(turn.text, rerun.text, "deterministic monologue");
    assert_eq!(audio.samples, rerun.audio.unwrap().samples);
}

#[test]
fn dialog_rejects_caller_supplied_reply_text_and_missing_audio() {
    // The inverse of the CSM contract (ADR M4-06 §D5).
    let engine = engine_with_aec(19);
    let err = engine
        .dialog(&vokra_core::DialogRequest::new("scripted reply"))
        .unwrap_err();
    assert!(
        err.to_string().contains("GENERATES its own"),
        "contract: {err}"
    );
    let err = engine
        .dialog(&vokra_core::DialogRequest::new(""))
        .unwrap_err();
    assert!(
        err.to_string().contains("input_audio"),
        "audio required: {err}"
    );
}

#[test]
fn dialog_honors_the_engine_echo_path_bypass() {
    // BypassRecordedInput (engine-level, CSM-mirroring) maps onto the
    // per-session explicit opt-out — a recipe-less engine can then run
    // batch dialogs over recorded input.
    use vokra_models::csm::EchoPath;
    let engine = MoshiEngine::synthesized_fixture(29)
        .expect("engine")
        .with_echo_path(EchoPath::BypassRecordedInput);
    let hop = engine.mimi_config().frame_hop_samples().unwrap();
    let request = vokra_core::DialogRequest::new("")
        .with_input_audio(vec![0.1; hop * 2])
        .deterministic();
    let turn = engine.dialog(&request).expect("bypass dialog");
    assert!(turn.audio.is_some());
    // Without the bypass, the same recipe-less engine refuses loudly.
    let engine = MoshiEngine::synthesized_fixture(29).expect("engine");
    assert!(engine.dialog(&request).is_err());
}

// ---------------------------------------------------------------------------
// T23 — the engine attribution surface (fixture side; the GGUF-resolved
// path is covered by the converter round-trip in vokra-convert)
// ---------------------------------------------------------------------------

#[test]
fn attribution_surface_defaults_none_and_carries_injected_info() {
    let engine = MoshiEngine::synthesized_fixture(2).expect("engine");
    assert!(
        engine.attribution().is_none(),
        "synthesized fixtures carry no weight → no attribution claim"
    );
    let engine = engine.with_attribution(vokra_core::AttributionInfo {
        text: "Moshi weights (c) Kyutai — CC-BY 4.0".into(),
        license: "attribution-required".into(),
        source_url: Some("https://github.com/kyutai-labs/moshi".into()),
    });
    assert_eq!(
        engine.attribution().unwrap().text,
        "Moshi weights (c) Kyutai — CC-BY 4.0"
    );
}

#[test]
fn watermark_config_default_is_on_and_backend_deferred() {
    let engine = MoshiEngine::synthesized_fixture(2).expect("engine");
    assert!(
        !engine.watermark().audioseal_opted_out(),
        "default ON (config-only posture; embedding stays Deferred)"
    );
}
