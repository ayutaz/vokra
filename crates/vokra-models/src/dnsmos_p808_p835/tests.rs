//! DNSMOS runtime binder tests (structural / round-trip / FR-EX-08 /
//! loud-partial). Mirror of `crate::kws::openwakeword::tests` (RMVPE
//! precedent): config round-trip, bundle-inventory walk (full / p808-
//! only / p835-only), missing-metadata loud errors, dim-mismatch loud
//! errors, and the loud-partial `score_*` gate.

use super::*;
use vokra_core::VokraError;
use vokra_core::engines::MosScorerEngine;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType, chunks,
};

/// Builds an `Array<String>` metadata chunk for the bundle inventory.
fn string_array(vs: &[&str]) -> GgufMetadataValue {
    GgufMetadataValue::Array(GgufArray {
        element_type: GgufValueType::String,
        values: vs
            .iter()
            .map(|s| GgufMetadataValue::String((*s).to_owned()))
            .collect(),
    })
}

/// Adds a zero-filled F32 tensor of the given dims.
fn add_zero(b: &mut GgufBuilder, name: &str, dims: &[u64]) {
    let n: u64 = dims.iter().product();
    b.add_tensor(
        name,
        GgmlType::F32,
        dims.to_vec(),
        vec![0u8; (n * 4) as usize],
    )
    .expect("add tensor");
}

/// Builds a tiny DNSMOS-shaped GGUF advertising the caller's bundle
/// variants. Each variant contributes one nominal weight tensor so the
/// tensor-presence check has something to walk. The absolute tensor
/// names mirror what the converter emits (verbatim prefixed initializer
/// names).
fn build_tiny_gguf(variants: &[&str], sample_rate: u32) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_u32(KEY_DNSMOS_SAMPLE_RATE, sample_rate);
    b.add_metadata(KEY_DNSMOS_BUNDLE, string_array(variants));
    for v in variants {
        match *v {
            "p808" => {
                b.add_string(KEY_DNSMOS_P808_CKPT, "model_v8.onnx");
                add_zero(&mut b, "p808.conv1/kernel", &[3, 3, 1, 16]);
            }
            "p835" => {
                b.add_string(KEY_DNSMOS_P835_CKPT, "sig_bak_ovr.onnx");
                add_zero(&mut b, "p835.conv1/kernel", &[3, 3, 1, 16]);
            }
            other => panic!("unknown variant `{other}` in tiny-GGUF builder"),
        }
    }
    b.to_bytes().expect("serialise tiny DNSMOS GGUF")
}

/// Full-bundle round-trip: both variants advertised, config lands.
#[test]
fn from_gguf_round_trips_full_bundle() {
    let bytes = build_tiny_gguf(&["p808", "p835"], 16_000);
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = Dnsmos::from_gguf(&gguf).expect("from_gguf must succeed on full bundle");
    let cfg = session.config();
    assert_eq!(cfg.sample_rate, 16_000);
    assert_eq!(cfg.bundle, vec!["p808".to_owned(), "p835".to_owned()]);
    assert!(cfg.has_p808);
    assert!(cfg.has_p835);
    let variants = session.variants();
    assert_eq!(variants, &["p808", "p835"]);
}

/// P.808-only bundle: variants set + config reflects the truthful subset.
#[test]
fn from_gguf_round_trips_p808_only_bundle() {
    let bytes = build_tiny_gguf(&["p808"], 16_000);
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = Dnsmos::from_gguf(&gguf).expect("p808-only bundle must load");
    assert!(session.config().has_p808);
    assert!(!session.config().has_p835);
    assert_eq!(session.variants(), &["p808"]);
}

/// P.835-only bundle: same as above with the other variant.
#[test]
fn from_gguf_round_trips_p835_only_bundle() {
    let bytes = build_tiny_gguf(&["p835"], 16_000);
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = Dnsmos::from_gguf(&gguf).expect("p835-only bundle must load");
    assert!(!session.config().has_p808);
    assert!(session.config().has_p835);
    assert_eq!(session.variants(), &["p835"]);
}

/// Wrong-arch GGUF is rejected before the tensor walk.
#[test]
fn from_gguf_rejects_wrong_arch_tag() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, "utmos");
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = Dnsmos::from_gguf(&gguf).expect_err("wrong arch must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(msg.contains("utmos"), "must name seen arch: {msg}");
    assert!(msg.contains(ARCH), "must name expected arch: {msg}");
}

/// Missing `vokra.dnsmos.bundle` is a loud [`VokraError::ModelLoad`].
#[test]
fn from_gguf_rejects_missing_bundle_metadata() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_u32(KEY_DNSMOS_SAMPLE_RATE, 16_000);
    // Deliberately omit KEY_DNSMOS_BUNDLE.
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = Dnsmos::from_gguf(&gguf).expect_err("missing bundle metadata must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains(KEY_DNSMOS_BUNDLE),
        "error must name missing key: {msg}"
    );
}

/// A `vokra.dnsmos.sample_rate` other than 16 000 is a loud
/// [`VokraError::ModelLoad`] (upstream DNSMOS is 16 kHz only).
#[test]
fn from_gguf_rejects_non_16k_sample_rate() {
    let bytes = build_tiny_gguf(&["p808"], 22_050);
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = Dnsmos::from_gguf(&gguf).expect_err("non-16 kHz sample rate must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(msg.contains("22050"), "error must name seen rate: {msg}");
    assert!(
        msg.contains("16000"),
        "error must name expected rate: {msg}"
    );
}

/// A bundle metadata advertising a variant with no matching tensor
/// prefix is a loud [`VokraError::ModelLoad`] (silent-partial is
/// forbidden per FR-EX-08 — a downstream binder that trusts the bundle
/// list would otherwise score against zero weights).
#[test]
fn from_gguf_rejects_advertised_variant_with_no_tensors() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_u32(KEY_DNSMOS_SAMPLE_RATE, 16_000);
    // Bundle advertises both variants but only P.808 tensors present.
    b.add_metadata(KEY_DNSMOS_BUNDLE, string_array(&["p808", "p835"]));
    add_zero(&mut b, "p808.conv1/kernel", &[3, 3, 1, 16]);
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = Dnsmos::from_gguf(&gguf)
        .expect_err("advertised variant with no matching tensor prefix must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains("p835"),
        "error must name the missing variant: {msg}"
    );
}

/// An empty `vokra.dnsmos.bundle` array is a loud
/// [`VokraError::ModelLoad`] (the converter refuses to emit one; a
/// hand-crafted GGUF with an empty bundle would otherwise pass the
/// "no tensors advertised → no gate to fire" branch silently).
#[test]
fn from_gguf_rejects_empty_bundle() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_u32(KEY_DNSMOS_SAMPLE_RATE, 16_000);
    b.add_metadata(KEY_DNSMOS_BUNDLE, string_array(&[]));
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = Dnsmos::from_gguf(&gguf).expect_err("empty bundle must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains("empty"),
        "error must name the emptiness offense: {msg}"
    );
}

/// A bundle inventory whose element type is not String is a loud
/// [`VokraError::ModelLoad`] (defense against a hand-crafted GGUF that
/// emits a `[u32]` bundle by mistake — the runtime must not silently
/// coerce).
#[test]
fn from_gguf_rejects_bundle_wrong_element_type() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_u32(KEY_DNSMOS_SAMPLE_RATE, 16_000);
    b.add_metadata(
        KEY_DNSMOS_BUNDLE,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: vec![GgufMetadataValue::U32(0), GgufMetadataValue::U32(1)],
        }),
    );
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = Dnsmos::from_gguf(&gguf).expect_err("wrong bundle element type must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains("String"),
        "error must name expected type: {msg}"
    );
}

/// Loud-partial contract (RMVPE + openwakeword precedent): a `score_*`
/// call on a real-shaped GGUF returns [`VokraError::UnsupportedOp`] with
/// an owner-facing message pointing at the topology-extension recipe.
/// No silent `0.0` MOS, no fabricated pass — an unwitting caller cannot
/// see a "MOS = 0" masquerading as a real score.
#[test]
fn score_p808_returns_loud_partial_until_cnn_topology_lands() {
    let bytes = build_tiny_gguf(&["p808", "p835"], 16_000);
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = Dnsmos::from_gguf(&gguf).unwrap();
    let pcm = vec![0.0f32; 16_000]; // 1 s
    let err = session
        .score_p808(&pcm)
        .expect_err("loud-partial must fire until CNN topology metadata lands (FR-EX-08)");
    let msg = match err {
        VokraError::UnsupportedOp(m) => m,
        other => panic!("expected UnsupportedOp, got {other:?}"),
    };
    // The message names the env-gate + the sidecar to extend so an
    // owner (or future CC wave) knows exactly where to flip the switch.
    assert!(
        msg.contains("vokra.dnsmos") && msg.contains("topology"),
        "loud-partial message must name the topology metadata: {msg}"
    );
    assert!(
        msg.contains("dnsmos_prepare_checkpoint"),
        "loud-partial message must name the sidecar to extend: {msg}"
    );
}

/// The `MosScorerEngine::score` surface hits the same loud-partial.
#[test]
fn engine_score_returns_loud_partial() {
    let bytes = build_tiny_gguf(&["p808", "p835"], 16_000);
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = Dnsmos::from_gguf(&gguf).unwrap();
    let pcm = vec![0.0f32; 16_000];
    let err = session
        .score(&pcm)
        .expect_err("MosScorerEngine::score must gate on the same loud-partial");
    assert!(matches!(err, VokraError::UnsupportedOp(_)));
}

/// Scoring a variant a partial bundle does not advertise is a loud
/// [`VokraError::InvalidArgument`] (not a silent `None`). A P.835-only
/// bundle handed a `score_p808` call must reject rather than fabricate.
#[test]
fn score_absent_variant_is_loud() {
    let bytes = build_tiny_gguf(&["p835"], 16_000);
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = Dnsmos::from_gguf(&gguf).unwrap();
    let err = session
        .score_p808(&[0.0f32; 16_000])
        .expect_err("scoring absent variant must be loud");
    let msg = match err {
        VokraError::InvalidArgument(m) => m,
        other => panic!("expected InvalidArgument, got {other:?}"),
    };
    assert!(
        msg.contains("p808"),
        "error must name the absent variant: {msg}"
    );
    assert!(
        msg.contains("bundle"),
        "error must reference the bundle: {msg}"
    );
}

/// The synthesized construction path yields a scorer that variants-
/// reports both variants (used by unit tests that do not want to build
/// a whole GGUF).
#[test]
fn synthesized_reports_both_variants() {
    let s = Dnsmos::synthesized();
    assert_eq!(s.variants(), &["p808", "p835"]);
    assert!(s.config().has_p808);
    assert!(s.config().has_p835);
    // Synthesized still lands on the loud-partial for score_* — the CNN
    // topology is not fabricated even for the synth path.
    let err = s
        .score_p808(&[0.0f32; 16_000])
        .expect_err("synth still loud-partial");
    assert!(matches!(err, VokraError::UnsupportedOp(_)));
}

/// Scoring a clip whose length is under one 9.01 s window is still
/// gated by the loud-partial — the length-check runs, but the
/// UnsupportedOp fires first so callers cannot confuse "too short →
/// zero score" with "real too-short-clip refusal". This pins the gate
/// order (loud-partial before any zero-pad-and-pretend).
#[test]
fn score_short_clip_still_hits_loud_partial() {
    let bytes = build_tiny_gguf(&["p808"], 16_000);
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = Dnsmos::from_gguf(&gguf).unwrap();
    let err = session
        .score_p808(&[0.0f32; 100])
        .expect_err("short clip still hits loud-partial");
    assert!(matches!(err, VokraError::UnsupportedOp(_)));
}

/// Hygiene: the loud-partial message must NOT reference the removed
/// `docs/handoff/dnsmos-runtime.md` file (drift would send owners
/// chasing a stub that does not exist). Verify defect
/// `dnsmos_p808_p835/mod.rs:499` (2026-08-05).
#[test]
fn loud_partial_message_does_not_reference_missing_handoff_doc() {
    let err = cnn_forward_loud_partial(DnsmosSubmodel::P808);
    let msg = match err {
        VokraError::UnsupportedOp(m) => m,
        other => panic!("expected UnsupportedOp, got {other:?}"),
    };
    assert!(
        !msg.contains("dnsmos-runtime.md"),
        "loud-partial must not reference the non-existent handoff doc: {msg}"
    );
}

/// Hygiene: the loud-partial message must reference the topology-key
/// constants verbatim (so a `KEY_DNSMOS_*_TOPOLOGY` rename cannot
/// silently drift the owner-facing recipe). Verify defect
/// `dnsmos_p808_p835/mod.rs:114` (2026-08-05).
#[test]
fn loud_partial_message_references_topology_key_consts_verbatim() {
    let err_p808 = cnn_forward_loud_partial(DnsmosSubmodel::P808);
    let msg_p808 = match err_p808 {
        VokraError::UnsupportedOp(m) => m,
        other => panic!("expected UnsupportedOp, got {other:?}"),
    };
    assert!(
        msg_p808.contains(KEY_DNSMOS_P808_TOPOLOGY),
        "P.808 loud-partial must reference `{KEY_DNSMOS_P808_TOPOLOGY}` verbatim: {msg_p808}",
    );
    let err_p835 = cnn_forward_loud_partial(DnsmosSubmodel::P835);
    let msg_p835 = match err_p835 {
        VokraError::UnsupportedOp(m) => m,
        other => panic!("expected UnsupportedOp, got {other:?}"),
    };
    assert!(
        msg_p835.contains(KEY_DNSMOS_P835_TOPOLOGY),
        "P.835 loud-partial must reference `{KEY_DNSMOS_P835_TOPOLOGY}` verbatim: {msg_p835}",
    );
}
