//! openWakeWord converter → binder handshake (2026-08-15 repair).
//!
//! # Why this test exists on the models side too
//!
//! `vokra-convert`'s `openwakeword_op` converter and this crate's
//! `kws::openwakeword` binder were written for each other and could not
//! handshake: `OpenwakewordConfig::from_gguf` reads seven
//! `vokra.openwakeword.*` keys as required, and the converter stamped
//! none of them, so every GGUF it produced failed at the binder's first
//! load. The owner recipe in `parity_openwakeword.rs` dead-ended there.
//!
//! The gap survived because both halves were tested against a mock of
//! the other: this crate's unit tests hand-build their GGUF with
//! `GgufBuilder` rather than running the converter, and the parity
//! harness that would have exercised the real pipeline is env-gated and
//! skips without an owner fixture. Tensor names matched all along —
//! only the metadata group was missing.
//!
//! `crates/vokra-convert/tests/openwakeword_op_roundtrip.rs` asserts the
//! key list from the converter's side. This test is the other half: it
//! runs the real converter and feeds its output straight into the real
//! binder, so neither crate can drift without something going red. It is
//! fixture-free — the synthetic checkpoint is built inline, so it runs
//! in CI with no owner-provisioned weights (unlike the env-gated parity
//! harness, which stays the place real-weight numerics are checked).
//!
//! Mirror of the `codec_gguf_roundtrip` / `moshi_convert_e2e` pattern
//! that the M4-04 codec wave established for exactly this hazard.

use std::path::PathBuf;

use vokra_convert::convert_openwakeword_op_file_with_config;
use vokra_core::VokraError;
use vokra_core::engines::KwsEngine;
use vokra_models::kws::openwakeword::{OpenwakewordSession, classify_embedding};

/// A unique temp path for this test process (no external `tempfile`
/// dep — zero-dep NFR-DS-02).
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-openwakeword-convert-bind-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    ));
    p
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Two wake-words over a shared 3-wide embedding, laid out exactly as
/// `tools/parity/openwakeword_prepare_checkpoint.py` emits and as this
/// crate's binder reads back.
///
/// - wake-word 0: `hidden_dim = 2`
/// - wake-word 1: `hidden_dim = 1`
///
/// Differing hidden dims are deliberate: the binder derives `hidden_dim`
/// per classifier from `linear1.bias`, so a bug that assumed one shared
/// hidden width would surface here.
fn synthetic_checkpoint() -> Vec<u8> {
    let w0_l1w = f32_bytes(&[0.1, 0.2, -0.1, 0.05, -0.05, 0.1]); // [2, 3]
    let w0_l1b = f32_bytes(&[0.01, -0.02]); // [2]
    let w0_l2w = f32_bytes(&[0.5, -0.3]); // [1, 2]
    let w0_l2b = f32_bytes(&[0.02]); // [1]
    let w1_l1w = f32_bytes(&[0.3, -0.2, 0.15]); // [1, 3]
    let w1_l1b = f32_bytes(&[0.03]); // [1]
    let w1_l2w = f32_bytes(&[0.4]); // [1, 1]
    let w1_l2b = f32_bytes(&[-0.04]); // [1]

    let mut off = 0usize;
    let mut bump = |n: usize| {
        let start = off;
        off += n;
        (start, off)
    };
    let (a0, a1) = bump(w0_l1w.len());
    let (b0, b1) = bump(w0_l1b.len());
    let (c0, c1) = bump(w0_l2w.len());
    let (d0, d1) = bump(w0_l2b.len());
    let (e0, e1) = bump(w1_l1w.len());
    let (f0, f1) = bump(w1_l1b.len());
    let (g0, g1) = bump(w1_l2w.len());
    let (h0, h1) = bump(w1_l2b.len());

    let header = format!(
        r#"{{"openwakeword.classifier.0.linear1.weight":{{"dtype":"F32","shape":[2,3],"data_offsets":[{a0},{a1}]}},"openwakeword.classifier.0.linear1.bias":{{"dtype":"F32","shape":[2],"data_offsets":[{b0},{b1}]}},"openwakeword.classifier.0.linear2.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[{c0},{c1}]}},"openwakeword.classifier.0.linear2.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{d0},{d1}]}},"openwakeword.classifier.1.linear1.weight":{{"dtype":"F32","shape":[1,3],"data_offsets":[{e0},{e1}]}},"openwakeword.classifier.1.linear1.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{f0},{f1}]}},"openwakeword.classifier.1.linear2.weight":{{"dtype":"F32","shape":[1,1],"data_offsets":[{g0},{g1}]}},"openwakeword.classifier.1.linear2.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{h0},{h1}]}}}}"#
    );

    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    for chunk in [
        &w0_l1w, &w0_l1b, &w0_l2w, &w0_l2b, &w1_l1w, &w1_l1b, &w1_l2w, &w1_l2b,
    ] {
        buf.extend_from_slice(chunk);
    }
    buf
}

/// Converts a synthetic checkpoint with the real converter and binds the
/// result with the real binder.
///
/// Returns the bound session plus the paths to clean up.
fn convert_and_bind(
    tag: &str,
    config_json: &str,
) -> (OpenwakewordSession, PathBuf, PathBuf, PathBuf) {
    let input = tmp_path(&format!("{tag}-in"));
    let config = tmp_path(&format!("{tag}-cfg"));
    let output = tmp_path(&format!("{tag}-out"));
    std::fs::write(&input, synthetic_checkpoint()).expect("write input");
    std::fs::write(&config, config_json).expect("write config");

    convert_openwakeword_op_file_with_config(&input, &config, &output, None)
        .unwrap_or_else(|e| panic!("converter failed: {e}"));

    let session = OpenwakewordSession::open(&output).unwrap_or_else(|e| {
        panic!(
            "THE HANDSHAKE IS BROKEN: a GGUF produced by \
             `convert_openwakeword_op_file_with_config` failed to load in \
             `OpenwakewordSession::from_gguf`: {e:?}"
        )
    });
    (session, input, config, output)
}

fn cleanup(paths: [&PathBuf; 3]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

/// THE fence. A GGUF from the converter must load in the binder written
/// for it, and every config field must survive the trip with the value
/// the checkpoint / side-car implied.
///
/// Before 2026-08-15 this test would have panicked inside
/// `convert_and_bind` with a `ModelLoad` on the first missing key.
#[test]
fn converter_output_binds_in_the_runtime_session() {
    let (session, input, config, output) =
        convert_and_bind("bind", r#"{"wakeword_names":["alexa","hey_jarvis"]}"#);

    let cfg = session.config();
    // Derived from the tensors by the converter.
    assert_eq!(
        cfg.n_wakewords, 2,
        "two classifier groups in the checkpoint"
    );
    assert_eq!(
        cfg.embedding_dim, 3,
        "embedding_dim comes from dim 1 of classifier.0.linear1.weight"
    );
    // Taken from the side-car — the axis that cannot be derived and is
    // never invented.
    assert_eq!(
        cfg.wakeword_names,
        vec!["alexa".to_owned(), "hey_jarvis".to_owned()],
        "the labels the runtime reports to callers must be the ones the \
         operator supplied, not synthesised indices"
    );
    // Mirrored front-end defaults.
    assert_eq!(cfg.window_frames, 76);
    assert_eq!(cfg.mel_bins, 32);
    assert_eq!(cfg.sample_rate, 16_000);
    assert_eq!(cfg.hop_samples, 160);

    // The classifier weights really bound — per-wake-word, with the
    // per-classifier hidden dims the checkpoint declared (2 and 1, not
    // one shared width).
    let classifiers = session.classifiers();
    assert_eq!(classifiers.len(), 2);
    assert_eq!(classifiers[0].name, "alexa");
    assert_eq!(classifiers[0].weights.hidden_dim, 2);
    assert_eq!(classifiers[0].weights.embedding_dim, 3);
    assert_eq!(classifiers[1].name, "hey_jarvis");
    assert_eq!(classifiers[1].weights.hidden_dim, 1);
    assert_eq!(classifiers[1].weights.embedding_dim, 3);

    // `KwsEngine::wakeword_names` is what a caller actually sees.
    let engine_names: Vec<String> = session.wakeword_names().to_vec();
    assert_eq!(
        engine_names,
        vec!["alexa".to_owned(), "hey_jarvis".to_owned()]
    );

    cleanup([&input, &config, &output]);
}

/// The bound classifiers are usable: running the real classifier forward
/// over a hand-built embedding of the converter-derived width produces
/// one finite sigmoid probability per wake-word.
///
/// This exercises the half of the pipeline that is real today. The
/// embedding extractor above it is a loud-partial (see below), so this
/// is deliberately driven from a caller-supplied embedding rather than
/// from PCM.
#[test]
fn bound_classifiers_run_the_real_forward() {
    let (session, input, config, output) =
        convert_and_bind("forward", r#"{"wakeword_names":["alexa","hey_jarvis"]}"#);

    let embedding = vec![0.3f32, -0.2, 0.5];
    assert_eq!(embedding.len(), session.config().embedding_dim);
    let out = classify_embedding(session.classifiers(), &embedding)
        .unwrap_or_else(|e| panic!("classify_embedding on bound weights failed: {e:?}"));
    assert_eq!(out.len(), 2);
    for (name, prob) in &out {
        assert!(
            (0.0..=1.0).contains(prob) && prob.is_finite(),
            "wake-word `{name}` probability {prob} must be a finite sigmoid output"
        );
    }
    assert_eq!(out[0].0, "alexa");
    assert_eq!(out[1].0, "hey_jarvis");

    cleanup([&input, &config, &output]);
}

/// The load now works; the FORWARD is still a loud-partial, and this
/// test pins that boundary so the repair is not mistaken for more than
/// it is.
///
/// `push_pcm16k` must return `UnsupportedOp` naming the env-gate, never
/// a fabricated `0.0` probability or an empty result a caller could read
/// as "no wake-word yet" (FR-EX-08). The frozen Google
/// `speech_embedding` extractor remains untranscribed.
#[test]
fn forward_remains_a_loud_partial_after_a_successful_bind() {
    let (mut session, input, config, output) =
        convert_and_bind("partial", r#"{"wakeword_names":["alexa","hey_jarvis"]}"#);

    let pcm = vec![0.0f32; 1_280]; // 80 ms at 16 kHz
    let Err(e) = session.push_pcm16k(&pcm) else {
        panic!(
            "push_pcm16k returned Ok — the embedding extractor is a loud-partial and must \
             not produce probabilities (FR-EX-08). If the extractor has genuinely landed, \
             update this test and the parity harness together."
        );
    };
    let msg = match e {
        VokraError::UnsupportedOp(m) => m,
        other => panic!("expected UnsupportedOp from the loud-partial gate, got {other:?}"),
    };
    assert!(
        msg.contains("VOKRA_OPENWAKEWORD_REAL_GGUF"),
        "the loud-partial must name the env-gate so an owner knows where to flip it: {msg}"
    );

    cleanup([&input, &config, &output]);
}

/// A side-car that overrides the front-end axes lands those values in
/// the binder's config, proving the override reaches the runtime rather
/// than stopping at the GGUF.
#[test]
fn side_car_front_end_overrides_reach_the_binder() {
    let (session, input, config, output) = convert_and_bind(
        "override",
        r#"{"wakeword_names":["alexa","hey_jarvis"],"window_frames":40,"mel_bins":16,"hop_samples":80}"#,
    );

    let cfg = session.config();
    assert_eq!(cfg.window_frames, 40);
    assert_eq!(cfg.mel_bins, 16);
    assert_eq!(cfg.hop_samples, 80);
    // Not overridden — still the mirrored default.
    assert_eq!(cfg.sample_rate, 16_000);

    cleanup([&input, &config, &output]);
}

/// A non-16 kHz checkpoint binds, but `push_pcm16k` refuses it loudly.
///
/// This is the safety argument for mirroring `sample_rate` as a constant
/// rather than demanding it: a wrong value cannot run silently. Pinning
/// it here keeps that argument honest — if the guard were ever dropped,
/// the mirror would become an invention.
#[test]
fn a_non_16khz_binding_is_refused_at_push_not_silently_run() {
    let (mut session, input, config, output) = convert_and_bind(
        "rate",
        r#"{"wakeword_names":["alexa","hey_jarvis"],"sample_rate":8000}"#,
    );
    assert_eq!(session.config().sample_rate, 8_000);

    let Err(e) = session.push_pcm16k(&[0.0f32; 160]) else {
        panic!("an 8 kHz-bound session must refuse 16 kHz PCM");
    };
    let msg = match e {
        VokraError::InvalidArgument(m) => m,
        other => panic!("expected InvalidArgument for the sample-rate mismatch, got {other:?}"),
    };
    assert!(
        msg.contains("16 kHz"),
        "the refusal must name the expected rate: {msg}"
    );

    cleanup([&input, &config, &output]);
}
