//! FSMN-VAD model-level tests (structural / round-trip / FR-EX-08).

use vokra_core::VokraError;
use vokra_core::engines::{VadEngine, VadStreamHandle};
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile, chunks};

use super::*;

/// Builds a minimal but architecturally-consistent GGUF for the tiny
/// test config (n_blocks=1, input=6, proj=2, hidden=2, lorder=2,
/// rorder=0, n_class=2, n_mels=3, lfr_m=2). Every tensor is F32 zeroed
/// unless overridden via the closure `override_tensor`.
///
/// Zero weights are the easiest input to reason about (the encoder
/// reduces to `logits = out_bias`), which is exactly what the
/// `from_gguf` round-trip tests here need to pin — the model-level
/// binding is the SUT, not the numeric forward (already covered by
/// vokra-ops).
fn build_tiny_gguf() -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, DEFAULT_NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    // Tiny-config hparams (still real, non-zero, non-degenerate).
    b.add_u32(KEY_N_BLOCKS, 1);
    b.add_u32(KEY_INPUT_DIM, 6); // == lfr_m * n_mels = 2 * 3
    b.add_u32(KEY_PROJ_DIM, 2);
    b.add_u32(KEY_HIDDEN_DIM, 2);
    b.add_u32(KEY_LORDER, 2);
    b.add_u32(KEY_RORDER, 0);
    b.add_u32(KEY_N_CLASS, 2);
    b.add_u32(KEY_N_MELS, 3);
    b.add_u32(KEY_LFR_M, 2);
    b.add_u32(KEY_LFR_N, 1);
    b.add_u32(KEY_SAMPLE_RATE, 16000);
    // Tensors — zeroed, correctly-shaped F32.
    let zeros = |n: usize| vec![0u8; n * 4];
    let add = |b: &mut GgufBuilder, name: &str, dims: &[u64]| {
        let n: u64 = dims.iter().product();
        b.add_tensor(name, GgmlType::F32, dims.to_vec(), zeros(n as usize))
            .expect("add tensor");
    };
    // Shapes match the [out, in] convention encoded in the module doc.
    add(&mut b, TENSOR_IN_PROJ_WEIGHT, &[2, 6]);
    add(&mut b, TENSOR_IN_PROJ_BIAS, &[2]);
    add(&mut b, &tensor_ffn1_weight(0), &[2, 2]);
    add(&mut b, &tensor_ffn1_bias(0), &[2]);
    add(&mut b, &tensor_ffn2_weight(0), &[2, 2]);
    add(&mut b, &tensor_ffn2_bias(0), &[2]);
    add(&mut b, &tensor_memory_weight(0), &[2, 3]); // proj_dim × memory_kernel(=3)
    add(&mut b, &tensor_memory_bias(0), &[2]);
    add(&mut b, TENSOR_OUT_WEIGHT, &[2, 2]); // n_class × proj_dim
    add(&mut b, TENSOR_OUT_BIAS, &[2]);
    b.to_bytes().expect("serialise tiny GGUF")
}

/// Same layout as `build_tiny_gguf` but with `out_bias` set to a
/// distinctive `[0.5, -0.25]` pattern — used to verify the round-trip
/// end-to-end.
fn build_tiny_gguf_with_out_bias() -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, DEFAULT_NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    b.add_u32(KEY_N_BLOCKS, 1);
    b.add_u32(KEY_INPUT_DIM, 6);
    b.add_u32(KEY_PROJ_DIM, 2);
    b.add_u32(KEY_HIDDEN_DIM, 2);
    b.add_u32(KEY_LORDER, 2);
    b.add_u32(KEY_RORDER, 0);
    b.add_u32(KEY_N_CLASS, 2);
    b.add_u32(KEY_N_MELS, 3);
    b.add_u32(KEY_LFR_M, 2);
    b.add_u32(KEY_LFR_N, 1);
    b.add_u32(KEY_SAMPLE_RATE, 16000);
    let zeros = |n: usize| vec![0u8; n * 4];
    let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let add = |b: &mut GgufBuilder, name: &str, dims: &[u64], data: Vec<u8>| {
        b.add_tensor(name, GgmlType::F32, dims.to_vec(), data)
            .expect("add tensor");
    };
    add(&mut b, TENSOR_IN_PROJ_WEIGHT, &[2, 6], zeros(12));
    add(&mut b, TENSOR_IN_PROJ_BIAS, &[2], zeros(2));
    add(&mut b, &tensor_ffn1_weight(0), &[2, 2], zeros(4));
    add(&mut b, &tensor_ffn1_bias(0), &[2], zeros(2));
    add(&mut b, &tensor_ffn2_weight(0), &[2, 2], zeros(4));
    add(&mut b, &tensor_ffn2_bias(0), &[2], zeros(2));
    add(&mut b, &tensor_memory_weight(0), &[2, 3], zeros(6));
    add(&mut b, &tensor_memory_bias(0), &[2], zeros(2));
    add(&mut b, TENSOR_OUT_WEIGHT, &[2, 2], zeros(4));
    // Distinctive out_bias so `forward_features` → softmax → speech-col
    // is non-trivial.
    add(&mut b, TENSOR_OUT_BIAS, &[2], f32_bytes(&[0.5, -0.25]));
    b.to_bytes().expect("serialise tiny GGUF")
}

#[test]
fn upstream_default_config_validates() {
    let c = FsmnVadConfig::upstream_default();
    c.validate().expect("upstream default must validate");
    // The primary axes cross-check the SPEC.md documented invariants.
    assert_eq!(c.n_mels, 80);
    assert_eq!(c.lfr_m, 5);
    assert_eq!(c.encoder.input_dim, 400);
    assert_eq!(c.encoder.memory_kernel(), 21); // lorder(20) + 1 + rorder(0)
}

#[test]
fn from_gguf_round_trips_tiny_config() {
    let bytes = build_tiny_gguf();
    let gguf = GgufFile::parse(bytes).expect("parse tiny gguf");
    let m = FsmnVadV1::from_gguf(&gguf).expect("load tiny fsmn-vad");
    assert_eq!(m.config().encoder.n_blocks, 1);
    assert_eq!(m.config().encoder.input_dim, 6);
    assert_eq!(m.config().encoder.proj_dim, 2);
    assert_eq!(m.config().n_mels, 3);
    assert_eq!(m.config().lfr_m, 2);
    assert_eq!(m.config().sample_rate, 16000);
}

#[test]
fn from_gguf_rejects_missing_arch_stamp() {
    // A GGUF with no arch stamp at all — the loader must refuse loudly
    // rather than trying to bind tensors under a mystery schema.
    let bytes = GgufBuilder::new().to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = FsmnVadV1::from_gguf(&gguf).unwrap_err();
    match err {
        VokraError::ModelLoad(m) => assert!(
            m.contains("vokra.model.arch"),
            "error should name the missing arch stamp, got: {m}"
        ),
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn from_gguf_rejects_wrong_arch_stamp() {
    // A GGUF stamped as silero-vad must not silently load into fsmn-vad
    // (silently sharing arch tags would misroute).
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, "silero-vad");
    let gguf = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
    let err = FsmnVadV1::from_gguf(&gguf).unwrap_err();
    match err {
        VokraError::ModelLoad(m) => {
            assert!(m.contains("silero-vad") && m.contains(ARCH), "got: {m}")
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn from_gguf_rejects_missing_hparam() {
    // Remove `KEY_N_BLOCKS` from the built GGUF: the loader must refuse
    // loudly rather than defaulting to some other block count.
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    // deliberately no KEY_N_BLOCKS
    b.add_u32(KEY_INPUT_DIM, 6);
    b.add_u32(KEY_PROJ_DIM, 2);
    b.add_u32(KEY_HIDDEN_DIM, 2);
    b.add_u32(KEY_LORDER, 2);
    b.add_u32(KEY_RORDER, 0);
    b.add_u32(KEY_N_CLASS, 2);
    b.add_u32(KEY_N_MELS, 3);
    b.add_u32(KEY_LFR_M, 2);
    b.add_u32(KEY_LFR_N, 1);
    b.add_u32(KEY_SAMPLE_RATE, 16000);
    let gguf = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
    let err = FsmnVadV1::from_gguf(&gguf).unwrap_err();
    match err {
        VokraError::ModelLoad(m) => assert!(
            m.contains(KEY_N_BLOCKS),
            "error should name the missing hparam, got: {m}"
        ),
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn from_gguf_rejects_lfr_mismatch() {
    // input_dim ≠ lfr_m × n_mels — the loader must refuse the config.
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_u32(KEY_N_BLOCKS, 1);
    b.add_u32(KEY_INPUT_DIM, 7); // ≠ 2 * 3 = 6
    b.add_u32(KEY_PROJ_DIM, 2);
    b.add_u32(KEY_HIDDEN_DIM, 2);
    b.add_u32(KEY_LORDER, 2);
    b.add_u32(KEY_RORDER, 0);
    b.add_u32(KEY_N_CLASS, 2);
    b.add_u32(KEY_N_MELS, 3);
    b.add_u32(KEY_LFR_M, 2);
    b.add_u32(KEY_LFR_N, 1);
    b.add_u32(KEY_SAMPLE_RATE, 16000);
    let gguf = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
    let err = FsmnVadV1::from_gguf(&gguf).unwrap_err();
    match err {
        VokraError::ModelLoad(m) => assert!(m.contains("input_dim"), "got: {m}"),
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn forward_features_yields_softmax_of_out_bias_when_weights_zero() {
    // With every weight zeroed, the encoder reduces to `logits =
    // out_bias`; the terminal softmax yields
    //   speech_col = exp(-0.25) / (exp(0.5) + exp(-0.25))
    //              ≈ 0.32082
    let bytes = build_tiny_gguf_with_out_bias();
    let gguf = GgufFile::parse(bytes).unwrap();
    let m = FsmnVadV1::from_gguf(&gguf).expect("load tiny fsmn-vad");
    let n_frames = 3;
    let features = vec![1.0f32; n_frames * m.config().encoder.input_dim];
    let probs = m
        .forward_features(&features)
        .expect("forward_features must succeed");
    assert_eq!(probs.len(), n_frames * m.config().encoder.n_class);
    // Compute expected softmax(0.5, -0.25) analytically.
    let e0 = 0.5f32.exp();
    let e1 = (-0.25f32).exp();
    let sum = e0 + e1;
    let want_silence = e0 / sum;
    let want_speech = e1 / sum;
    for f in 0..n_frames {
        let got_silence = probs[f * 2];
        let got_speech = probs[f * 2 + 1];
        assert!(
            (got_silence - want_silence).abs() < 1e-5,
            "frame {f} silence: got {got_silence} want {want_silence}",
        );
        assert!(
            (got_speech - want_speech).abs() < 1e-5,
            "frame {f} speech: got {got_speech} want {want_speech}",
        );
    }
}

#[test]
fn vad_engine_stream_reset_reproduces_initial_output() {
    let bytes = build_tiny_gguf_with_out_bias();
    let gguf = GgufFile::parse(bytes).unwrap();
    let m = FsmnVadV1::from_gguf(&gguf).unwrap();
    let features = vec![1.0f32; 4 * m.config().encoder.input_dim];

    // Downcast so the test can call `push_features` (VadStreamHandle
    // trait only exposes push_pcm today). Mirror of the silero_vad
    // pattern where the raw-frame path is a test-only entry.
    let mut stream = FsmnVadStream::new(m.config().clone(), Arc::clone(&m.weights));
    let a = stream.push_features(&features).unwrap();
    stream.reset();
    let b = stream.push_features(&features).unwrap();
    assert_eq!(a, b, "reset must reproduce the initial run bit-for-bit");
}

#[test]
fn vad_engine_push_pcm_is_loud_error_until_frontend_lands() {
    // FR-EX-08: push_pcm must NOT silently short-circuit to a features
    // path. Until the fbank + LFR + CMVN chain is wired, callers must
    // pre-compute features via a Python bridge.
    let bytes = build_tiny_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let m = FsmnVadV1::from_gguf(&gguf).unwrap();
    let mut s = m.open_stream();
    let err = s.push_pcm(&[0.0; 512], 16000).unwrap_err();
    match err {
        VokraError::UnsupportedOp(msg) => {
            assert!(
                msg.contains("push_features") && msg.contains("front-end"),
                "error should tell the caller to use push_features, got: {msg}"
            );
        }
        other => {
            panic!("push_pcm should return UnsupportedOp (loud FR-EX-08 error), got: {other:?}")
        }
    }
}

#[test]
fn push_features_carries_state_across_chunks() {
    // Split a 4-frame input into two 2-frame chunks; state must be
    // carried, so the concatenated result matches a single 4-frame
    // push. The numeric core already pins this
    // (`vokra_ops::fsmn_vad::tests::state_carry_matches_single_chunk`);
    // this test verifies the model-level wrapper is not accidentally
    // resetting state between calls.
    let bytes = build_tiny_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let m = FsmnVadV1::from_gguf(&gguf).unwrap();
    let input_dim = m.config().encoder.input_dim;
    let features: Vec<f32> = (0..4 * input_dim)
        .map(|i| ((i as f32) - 5.5) * 0.13)
        .collect();

    // Path A: one 4-frame call.
    let mut sa = FsmnVadStream::new(m.config().clone(), Arc::clone(&m.weights));
    let all = sa.push_features(&features).unwrap();

    // Path B: two 2-frame calls.
    let mut sb = FsmnVadStream::new(m.config().clone(), Arc::clone(&m.weights));
    let mut split: Vec<f32> = Vec::new();
    split.extend(sb.push_features(&features[..2 * input_dim]).unwrap());
    split.extend(sb.push_features(&features[2 * input_dim..]).unwrap());

    assert_eq!(all.len(), split.len());
    for i in 0..all.len() {
        let d = (all[i] - split[i]).abs();
        assert!(
            d < 1e-5,
            "streaming ⇔ batch diverged at idx {i}: batch={} stream={} diff={d}",
            all[i],
            split[i],
        );
    }
}
