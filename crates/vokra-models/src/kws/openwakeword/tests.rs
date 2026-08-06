//! openWakeWord runtime binder tests (structural / round-trip / FR-EX-08).
//!
//! Mirror of `crate::fsmn_vad::tests` scope: config round-trip, missing-
//! tensor loud errors, dim-mismatch loud errors, and the **loud-partial**
//! `push_pcm16k` gate (RMVPE precedent — the embedding extractor is
//! deferred to owner-provisioned Google `speech_embedding` bundle).

use super::*;

use vokra_core::VokraError;
use vokra_core::engines::KwsEngine;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType, chunks,
};

/// Builds an `Array<String>` metadata chunk for the wake-word names.
fn string_array_chunk(names: &[&str]) -> GgufMetadataValue {
    GgufMetadataValue::Array(GgufArray {
        element_type: GgufValueType::String,
        values: names
            .iter()
            .map(|s| GgufMetadataValue::String((*s).to_owned()))
            .collect(),
    })
}

/// Tiny fixture: embedding_dim=4, hidden_dim=3, 2 wake-words. Weights are
/// F32 zeros unless overridden — the model-level binding is the SUT, not
/// the numeric forward (already covered in `vokra-ops`).
fn build_tiny_gguf(embedding_dim: usize, hidden_dim: usize, wakewords: &[&str]) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, DEFAULT_NAME);
    b.add_u32(KEY_N_WAKEWORDS, wakewords.len() as u32);
    b.add_u32(KEY_EMBEDDING_DIM, embedding_dim as u32);
    b.add_u32(KEY_WINDOW_FRAMES, 4); // Tiny window, not upstream 76.
    b.add_u32(KEY_MEL_BINS, 4); // Tiny mel width, not upstream 32.
    b.add_u32(KEY_SAMPLE_RATE, 16_000);
    b.add_u32(KEY_HOP_SAMPLES, 160);
    b.add_metadata(KEY_WAKEWORD_NAMES, string_array_chunk(wakewords));

    let zeros = |n: usize| vec![0u8; n * 4];
    let add = |b: &mut GgufBuilder, name: &str, dims: &[u64]| {
        let n: u64 = dims.iter().product();
        b.add_tensor(name, GgmlType::F32, dims.to_vec(), zeros(n as usize))
            .expect("add tensor");
    };
    for i in 0..wakewords.len() {
        add(
            &mut b,
            &tensor_classifier_linear1_weight(i),
            &[hidden_dim as u64, embedding_dim as u64],
        );
        add(
            &mut b,
            &tensor_classifier_linear1_bias(i),
            &[hidden_dim as u64],
        );
        add(
            &mut b,
            &tensor_classifier_linear2_weight(i),
            &[1, hidden_dim as u64],
        );
        add(&mut b, &tensor_classifier_linear2_bias(i), &[1]);
    }
    b.to_bytes().expect("serialise tiny GGUF")
}

/// Round-trip through `from_gguf`: config + classifier bind end-to-end.
#[test]
fn from_gguf_round_trips_config_and_classifiers() {
    let bytes = build_tiny_gguf(4, 3, &["alexa", "hey_jarvis"]);
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = OpenwakewordSession::from_gguf(&gguf).expect("from_gguf must succeed");
    let cfg = session.config();
    assert_eq!(cfg.n_wakewords, 2);
    assert_eq!(cfg.embedding_dim, 4);
    assert_eq!(cfg.window_frames, 4);
    assert_eq!(cfg.mel_bins, 4);
    assert_eq!(cfg.sample_rate, 16_000);
    assert_eq!(cfg.hop_samples, 160);
    assert_eq!(cfg.wakeword_names, vec!["alexa", "hey_jarvis"]);
    assert_eq!(session.classifiers().len(), 2);
    assert_eq!(session.classifiers()[0].name, "alexa");
    assert_eq!(session.classifiers()[0].weights.embedding_dim, 4);
    assert_eq!(session.classifiers()[0].weights.hidden_dim, 3);
    assert_eq!(session.wakeword_names(), &["alexa", "hey_jarvis"]);
}

/// The classifier-only helper computes sigmoid probabilities for every
/// wake-word from a bound session's classifiers. Zero weights ⇒ every
/// probability is `sigmoid(0) = 0.5` (a real forward, not a stub).
#[test]
fn classify_embedding_returns_sigmoid_of_zero_for_zero_weights() {
    let bytes = build_tiny_gguf(4, 3, &["alexa", "hey_jarvis"]);
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = OpenwakewordSession::from_gguf(&gguf).unwrap();
    let embedding = vec![1.0f32; 4];
    let out = classify_embedding(session.classifiers(), &embedding).unwrap();
    assert_eq!(out.len(), 2);
    for (name, p) in &out {
        assert!(
            (p - 0.5).abs() < 1e-6,
            "zero-weight classifier `{name}` must return sigmoid(0)=0.5, got {p}"
        );
    }
}

/// A GGUF missing `vokra.openwakeword.n_wakewords` is a loud
/// [`VokraError::ModelLoad`] naming the offending key.
#[test]
fn from_gguf_rejects_missing_hparams_loudly() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, DEFAULT_NAME);
    // Deliberately omit KEY_N_WAKEWORDS.
    b.add_u32(KEY_EMBEDDING_DIM, 96);
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = OpenwakewordSession::from_gguf(&gguf).expect_err("missing hparam must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains(KEY_N_WAKEWORDS),
        "error must name the offending key `{KEY_N_WAKEWORDS}`: {msg}"
    );
}

/// A GGUF with the wrong arch tag is rejected before the load pipeline
/// even reads the `vokra.openwakeword.*` chunks — the caller gets a
/// clear "wrong arch" message instead of a downstream "missing tensor".
#[test]
fn from_gguf_rejects_wrong_arch_tag() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, "fsmn-vad");
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = OpenwakewordSession::from_gguf(&gguf).expect_err("wrong arch must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains("fsmn-vad"),
        "error must name the seen arch: {msg}"
    );
    assert!(
        msg.contains(ARCH),
        "error must name the expected arch: {msg}"
    );
}

/// A GGUF with `n_wakewords` = 3 but only 2 names is a loud
/// [`VokraError::ModelLoad`] (naming inconsistency).
#[test]
fn from_gguf_rejects_wakeword_name_count_mismatch() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_u32(KEY_N_WAKEWORDS, 3);
    b.add_u32(KEY_EMBEDDING_DIM, 4);
    b.add_u32(KEY_WINDOW_FRAMES, 4);
    b.add_u32(KEY_MEL_BINS, 4);
    b.add_u32(KEY_SAMPLE_RATE, 16_000);
    b.add_u32(KEY_HOP_SAMPLES, 160);
    b.add_metadata(
        KEY_WAKEWORD_NAMES,
        string_array_chunk(&["alexa", "hey_jarvis"]),
    );
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = OpenwakewordSession::from_gguf(&gguf)
        .expect_err("wakeword name count mismatch must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains("wakeword_names"),
        "error must name the offending field: {msg}"
    );
    assert!(
        msg.contains("2"),
        "error must mention actual count 2: {msg}"
    );
    assert!(
        msg.contains("3"),
        "error must mention expected count 3: {msg}"
    );
}

/// A GGUF with a missing per-wake-word tensor is a loud
/// [`VokraError::ModelLoad`] naming the offending tensor.
#[test]
fn from_gguf_rejects_missing_classifier_tensor() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_u32(KEY_N_WAKEWORDS, 1);
    b.add_u32(KEY_EMBEDDING_DIM, 4);
    b.add_u32(KEY_WINDOW_FRAMES, 4);
    b.add_u32(KEY_MEL_BINS, 4);
    b.add_u32(KEY_SAMPLE_RATE, 16_000);
    b.add_u32(KEY_HOP_SAMPLES, 160);
    b.add_metadata(KEY_WAKEWORD_NAMES, string_array_chunk(&["alexa"]));
    // Add linear1_weight but deliberately omit linear1_bias — the
    // loader must catch the missing bias.
    b.add_tensor(
        &tensor_classifier_linear1_weight(0),
        GgmlType::F32,
        vec![3, 4],
        vec![0u8; 3 * 4 * 4],
    )
    .unwrap();
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err =
        OpenwakewordSession::from_gguf(&gguf).expect_err("missing classifier tensor must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains(&tensor_classifier_linear1_bias(0)),
        "error must name the missing tensor: {msg}"
    );
}

/// A GGUF whose linear1_weight shape does not match the declared
/// embedding_dim is a loud [`VokraError::ModelLoad`].
#[test]
fn from_gguf_rejects_dim_mismatched_linear1_weight() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_u32(KEY_N_WAKEWORDS, 1);
    b.add_u32(KEY_EMBEDDING_DIM, 4);
    b.add_u32(KEY_WINDOW_FRAMES, 4);
    b.add_u32(KEY_MEL_BINS, 4);
    b.add_u32(KEY_SAMPLE_RATE, 16_000);
    b.add_u32(KEY_HOP_SAMPLES, 160);
    b.add_metadata(KEY_WAKEWORD_NAMES, string_array_chunk(&["alexa"]));

    // Declared embedding_dim = 4, but linear1_weight has [3, 5] = 15
    // elements (hidden_dim=3 * embedding_dim=5); the loader must catch
    // the mismatch and refuse loudly.
    b.add_tensor(
        &tensor_classifier_linear1_weight(0),
        GgmlType::F32,
        vec![3, 5],
        vec![0u8; 3 * 5 * 4],
    )
    .unwrap();
    b.add_tensor(
        &tensor_classifier_linear1_bias(0),
        GgmlType::F32,
        vec![3],
        vec![0u8; 3 * 4],
    )
    .unwrap();
    b.add_tensor(
        &tensor_classifier_linear2_weight(0),
        GgmlType::F32,
        vec![1, 3],
        vec![0u8; 3 * 4],
    )
    .unwrap();
    b.add_tensor(
        &tensor_classifier_linear2_bias(0),
        GgmlType::F32,
        vec![1],
        vec![0u8; 4],
    )
    .unwrap();
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = OpenwakewordSession::from_gguf(&gguf)
        .expect_err("dim-mismatched linear1_weight must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains(&tensor_classifier_linear1_weight(0)),
        "error must name the offending tensor: {msg}"
    );
}

/// A GGUF with a zero-length hidden layer (empty linear1_bias) is a
/// loud [`VokraError::ModelLoad`].
#[test]
fn from_gguf_rejects_empty_hidden_layer() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_u32(KEY_N_WAKEWORDS, 1);
    b.add_u32(KEY_EMBEDDING_DIM, 4);
    b.add_u32(KEY_WINDOW_FRAMES, 4);
    b.add_u32(KEY_MEL_BINS, 4);
    b.add_u32(KEY_SAMPLE_RATE, 16_000);
    b.add_u32(KEY_HOP_SAMPLES, 160);
    b.add_metadata(KEY_WAKEWORD_NAMES, string_array_chunk(&["alexa"]));
    b.add_tensor(
        &tensor_classifier_linear1_weight(0),
        GgmlType::F32,
        vec![0, 4],
        vec![],
    )
    .unwrap();
    b.add_tensor(
        &tensor_classifier_linear1_bias(0),
        GgmlType::F32,
        vec![0],
        vec![],
    )
    .unwrap();
    b.add_tensor(
        &tensor_classifier_linear2_weight(0),
        GgmlType::F32,
        vec![1, 0],
        vec![],
    )
    .unwrap();
    b.add_tensor(
        &tensor_classifier_linear2_bias(0),
        GgmlType::F32,
        vec![1],
        vec![0u8; 4],
    )
    .unwrap();
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = OpenwakewordSession::from_gguf(&gguf).expect_err("empty hidden layer must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains("hidden_dim") || msg.contains(&tensor_classifier_linear1_bias(0)),
        "error must name the offending layer: {msg}"
    );
}

/// **Loud-partial** contract (RMVPE precedent): the first `push_pcm16k`
/// call returns [`VokraError::UnsupportedOp`] with an owner-facing
/// message pointing at the env-gated parity harness. No silent buffering
/// of never-consumable data, no fabricated `Ok(vec![])`.
#[test]
fn push_pcm16k_returns_loud_partial_until_real_embedding_binds() {
    let bytes = build_tiny_gguf(4, 3, &["alexa"]);
    let gguf = GgufFile::parse(bytes).unwrap();
    let mut session = OpenwakewordSession::from_gguf(&gguf).unwrap();
    let err = session
        .push_pcm16k(&[0.0f32; 160])
        .expect_err("loud-partial must fire on the first push (FR-EX-08)");
    let msg = match err {
        VokraError::UnsupportedOp(m) => m,
        other => panic!("expected UnsupportedOp, got {other:?}"),
    };
    assert!(
        msg.contains("VOKRA_OPENWAKEWORD_REAL_GGUF"),
        "loud-partial message must name the env-gate for owner flip: {msg}"
    );
    assert!(
        msg.contains("parity_openwakeword"),
        "loud-partial message must name the parity script: {msg}"
    );
}

/// A GGUF whose `linear1_weight` dims are the transpose of the
/// documented row-major `[hidden_dim, embedding_dim]` layout is a loud
/// [`VokraError::ModelLoad`]. Without this dim-order check the product
/// (`hidden_dim * embedding_dim`) alone would let a Python bridge that
/// silently writes `[embedding_dim, hidden_dim]` through, then
/// misforward on every inference.
#[test]
fn from_gguf_rejects_transposed_linear1_weight_dims() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_u32(KEY_N_WAKEWORDS, 1);
    b.add_u32(KEY_EMBEDDING_DIM, 4);
    b.add_u32(KEY_WINDOW_FRAMES, 4);
    b.add_u32(KEY_MEL_BINS, 4);
    b.add_u32(KEY_SAMPLE_RATE, 16_000);
    b.add_u32(KEY_HOP_SAMPLES, 160);
    b.add_metadata(KEY_WAKEWORD_NAMES, string_array_chunk(&["alexa"]));
    // hidden_dim=3 (from bias), embedding_dim=4 (from config). Documented
    // dims = [3, 4]. Adversarial transpose = [4, 3]; product 12 matches
    // so the product-only check would silently accept.
    b.add_tensor(
        &tensor_classifier_linear1_weight(0),
        GgmlType::F32,
        vec![4, 3],
        vec![0u8; 3 * 4 * 4],
    )
    .unwrap();
    b.add_tensor(
        &tensor_classifier_linear1_bias(0),
        GgmlType::F32,
        vec![3],
        vec![0u8; 3 * 4],
    )
    .unwrap();
    b.add_tensor(
        &tensor_classifier_linear2_weight(0),
        GgmlType::F32,
        vec![1, 3],
        vec![0u8; 3 * 4],
    )
    .unwrap();
    b.add_tensor(
        &tensor_classifier_linear2_bias(0),
        GgmlType::F32,
        vec![1],
        vec![0u8; 4],
    )
    .unwrap();
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = OpenwakewordSession::from_gguf(&gguf)
        .expect_err("transposed linear1_weight dims must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains(&tensor_classifier_linear1_weight(0)),
        "error must name the offending tensor: {msg}"
    );
    assert!(msg.contains("dims"), "error must mention dims: {msg}");
    assert!(
        msg.contains("hidden_dim") && msg.contains("embedding_dim"),
        "error must name the expected layout: {msg}"
    );
}

/// A GGUF whose `linear2_weight` dims are the transpose of the
/// documented `[1, hidden_dim]` layout is a loud
/// [`VokraError::ModelLoad`] (same rationale as
/// [`from_gguf_rejects_transposed_linear1_weight_dims`]).
#[test]
fn from_gguf_rejects_transposed_linear2_weight_dims() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_u32(KEY_N_WAKEWORDS, 1);
    b.add_u32(KEY_EMBEDDING_DIM, 4);
    b.add_u32(KEY_WINDOW_FRAMES, 4);
    b.add_u32(KEY_MEL_BINS, 4);
    b.add_u32(KEY_SAMPLE_RATE, 16_000);
    b.add_u32(KEY_HOP_SAMPLES, 160);
    b.add_metadata(KEY_WAKEWORD_NAMES, string_array_chunk(&["alexa"]));
    b.add_tensor(
        &tensor_classifier_linear1_weight(0),
        GgmlType::F32,
        vec![3, 4],
        vec![0u8; 3 * 4 * 4],
    )
    .unwrap();
    b.add_tensor(
        &tensor_classifier_linear1_bias(0),
        GgmlType::F32,
        vec![3],
        vec![0u8; 3 * 4],
    )
    .unwrap();
    // Documented dims = [1, 3]. Adversarial transpose = [3, 1]; product 3
    // matches so the product-only check would silently accept.
    b.add_tensor(
        &tensor_classifier_linear2_weight(0),
        GgmlType::F32,
        vec![3, 1],
        vec![0u8; 3 * 4],
    )
    .unwrap();
    b.add_tensor(
        &tensor_classifier_linear2_bias(0),
        GgmlType::F32,
        vec![1],
        vec![0u8; 4],
    )
    .unwrap();
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = OpenwakewordSession::from_gguf(&gguf)
        .expect_err("transposed linear2_weight dims must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains(&tensor_classifier_linear2_weight(0)),
        "error must name the offending tensor: {msg}"
    );
    assert!(msg.contains("dims"), "error must mention dims: {msg}");
    assert!(
        msg.contains("hidden_dim"),
        "error must mention expected layout: {msg}"
    );
}

/// The retry-loop resistance contract: a caller that swallows the
/// loud-partial [`VokraError::UnsupportedOp`] and keeps pushing PCM must
/// NOT grow the internal `pending_pcm` buffer. Without this gate
/// (buffer extend *after* the loud-partial fires), a naive retry loop
/// would exhaust memory over time even though every push errors out.
#[test]
fn push_pcm16k_does_not_grow_pending_pcm_on_loud_partial() {
    let bytes = build_tiny_gguf(4, 3, &["alexa"]);
    let gguf = GgufFile::parse(bytes).unwrap();
    let mut session = OpenwakewordSession::from_gguf(&gguf).unwrap();
    for _ in 0..100 {
        // Every push errors — retry loop simulation. Buffer must stay
        // empty since the loud-partial gate runs before the extend.
        let _ = session.push_pcm16k(&[0.0f32; 1600]);
    }
    assert_eq!(
        session.pending_pcm.len(),
        0,
        "loud-partial must run before any buffer extend — retry loops \
         would otherwise exhaust memory"
    );
}

/// A session with `wakeword_names` in a specific order round-trips them
/// through the [`KwsEngine::wakeword_names`] method. Callers key on the
/// name → index correspondence downstream, so an implicit sort would be
/// a silent bug.
#[test]
fn wakeword_names_order_matches_gguf_order() {
    let bytes = build_tiny_gguf(4, 3, &["hey_jarvis", "alexa", "hey_mycroft"]);
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = OpenwakewordSession::from_gguf(&gguf).unwrap();
    assert_eq!(
        session.wakeword_names(),
        &["hey_jarvis", "alexa", "hey_mycroft"]
    );
    assert_eq!(session.classifiers()[0].name, "hey_jarvis");
    assert_eq!(session.classifiers()[1].name, "alexa");
    assert_eq!(session.classifiers()[2].name, "hey_mycroft");
}
