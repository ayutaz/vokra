use std::sync::Arc;

use vokra_core::engines::{VadEngine, VadStreamHandle};
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType, chunks,
};

use super::*;

fn f32_array(values: &[f32]) -> GgufMetadataValue {
    GgufMetadataValue::Array(GgufArray {
        element_type: GgufValueType::F32,
        values: values.iter().copied().map(GgufMetadataValue::F32).collect(),
    })
}

fn add_tensor(builder: &mut GgufBuilder, name: &str, shape: &[u64], values: &[f32]) {
    let elements = shape.iter().product::<u64>() as usize;
    assert_eq!(values.len(), elements);
    builder
        .add_tensor(
            name,
            GgmlType::F32,
            shape.to_vec(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )
        .unwrap();
}

fn tiny_gguf(out_bias: [f32; 3]) -> Vec<u8> {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, DEFAULT_NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_MODELSCOPE, UPSTREAM_MODELSCOPE);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, UPSTREAM_REVISION);
    builder.add_string(KEY_CHECKPOINT_SHA256, MODEL_SHA256);
    builder.add_string(KEY_CMVN_SHA256, CMVN_SHA256);
    builder.add_string(KEY_CONFIG_SHA256, CONFIG_SHA256);
    for (key, value) in [
        (KEY_N_BLOCKS, 1),
        (KEY_INPUT_DIM, 6),
        (KEY_INPUT_AFFINE_DIM, 2),
        (KEY_LINEAR_DIM, 2),
        (KEY_PROJ_DIM, 2),
        (KEY_LORDER, 2),
        (KEY_RORDER, 0),
        (KEY_LSTRIDE, 1),
        (KEY_RSTRIDE, 0),
        (KEY_OUTPUT_AFFINE_DIM, 2),
        (KEY_OUTPUT_DIM, 3),
        (KEY_N_MELS, 3),
        (KEY_LFR_M, 2),
        (KEY_LFR_N, 1),
        (KEY_SAMPLE_RATE, 16_000),
    ] {
        builder.add_u32(key, value);
    }
    builder.add_metadata(KEY_CMVN_ADD_SHIFT, f32_array(&[1.0; 6]));
    builder.add_metadata(KEY_CMVN_RESCALE, f32_array(&[2.0; 6]));

    add_tensor(&mut builder, TENSOR_IN_LINEAR1_WEIGHT, &[2, 6], &[0.0; 12]);
    add_tensor(&mut builder, TENSOR_IN_LINEAR1_BIAS, &[2], &[0.0; 2]);
    add_tensor(&mut builder, TENSOR_IN_LINEAR2_WEIGHT, &[2, 2], &[0.0; 4]);
    add_tensor(&mut builder, TENSOR_IN_LINEAR2_BIAS, &[2], &[0.0; 2]);
    add_tensor(
        &mut builder,
        &tensor_block_linear_weight(0),
        &[2, 2],
        &[0.0; 4],
    );
    add_tensor(
        &mut builder,
        &tensor_block_memory_weight(0),
        &[2, 1, 2, 1],
        &[0.0; 4],
    );
    add_tensor(
        &mut builder,
        &tensor_block_affine_weight(0),
        &[2, 2],
        &[0.0; 4],
    );
    add_tensor(&mut builder, &tensor_block_affine_bias(0), &[2], &[0.0; 2]);
    add_tensor(&mut builder, TENSOR_OUT_LINEAR1_WEIGHT, &[2, 2], &[0.0; 4]);
    add_tensor(&mut builder, TENSOR_OUT_LINEAR1_BIAS, &[2], &[0.0; 2]);
    add_tensor(&mut builder, TENSOR_OUT_LINEAR2_WEIGHT, &[3, 2], &[0.0; 6]);
    add_tensor(&mut builder, TENSOR_OUT_LINEAR2_BIAS, &[3], &out_bias);
    builder.to_bytes().unwrap()
}

fn tiny_model() -> FsmnVadV1 {
    FsmnVadV1::from_gguf(&GgufFile::parse(tiny_gguf([0.5, -0.25, 0.125])).unwrap()).unwrap()
}

fn stream_from(model: &FsmnVadV1) -> FsmnVadStream {
    FsmnVadStream::new(
        model.cfg.clone(),
        Arc::clone(&model.weights),
        Arc::clone(&model.cmvn_add_shift),
        Arc::clone(&model.cmvn_rescale),
        model.backend,
    )
}

#[test]
fn upstream_default_matches_released_config() {
    let cfg = FsmnVadConfig::upstream_default();
    cfg.validate().unwrap();
    assert_eq!(cfg.encoder.input_dim, 400);
    assert_eq!(cfg.encoder.linear_dim, 250);
    assert_eq!(cfg.encoder.proj_dim, 128);
    assert_eq!(cfg.encoder.output_dim, 248);
    assert_eq!((cfg.lfr_m, cfg.lfr_n), (5, 1));
}

#[test]
fn loader_binds_exact_tensor_schema() {
    let model = tiny_model();
    assert_eq!(model.config().encoder.output_dim, 3);
    assert_eq!(model.config().encoder.left_history_frames(), 1);
    assert_eq!(model.backend(), vokra_core::backend::BackendKind::Cpu);
    assert_eq!(
        tiny_model()
            .with_backend(vokra_core::backend::BackendKind::Metal)
            .backend(),
        vokra_core::backend::BackendKind::Metal
    );
}

#[test]
fn loader_rejects_unpinned_identity() {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
    let error = FsmnVadV1::from_gguf(&file).unwrap_err().to_string();
    assert!(error.contains(KEY_PROVENANCE_UPSTREAM_HF), "{error}");
}

#[test]
fn zero_graph_reduces_to_terminal_bias_softmax() {
    let model = tiny_model();
    let probabilities = model.forward_features(&[1.0; 12]).unwrap();
    assert_eq!(probabilities.len(), 6);
    let logits = [0.5f32, -0.25, 0.125];
    let sum = logits.iter().map(|value| value.exp()).sum::<f32>();
    for row in probabilities.chunks_exact(3) {
        for index in 0..3 {
            assert!((row[index] - logits[index].exp() / sum).abs() < 1e-6);
        }
    }
}

#[test]
fn vad_score_is_one_minus_silence_pdf() {
    let model = tiny_model();
    let mut stream = stream_from(&model);
    let scores = stream.push_features(&[0.0; 6]).unwrap();
    let logits = [0.5f32, -0.25, 0.125];
    let sum = logits.iter().map(|value| value.exp()).sum::<f32>();
    assert!((scores[0] - (1.0 - logits[0].exp() / sum)).abs() < 1e-6);
}

#[test]
fn cmvn_is_upstream_add_shift_then_rescale() {
    let model = tiny_model();
    let stream = stream_from(&model);
    let mut features = [0.5f32; 6];
    stream.apply_cmvn(&mut features);
    assert_eq!(features, [3.0; 6]);
}

#[test]
fn canonical_lfr_adds_two_left_edge_frames_and_keeps_incomplete_tail() {
    let model = tiny_model();
    let mut stream = stream_from(&model);
    stream.cfg.lfr_m = 5;
    stream.cfg.encoder.input_dim = 15;
    stream
        .pending_frames
        .extend_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    for _ in 0..2 {
        let first = stream.pending_frames[..3].to_vec();
        stream.pending_frames.splice(0..0, first);
    }
    stream.lfr_initialized = true;
    let output = stream.drain_frames_into_lfr();
    assert_eq!(output.len(), 15);
    assert_eq!(&output[..9], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    assert_eq!(stream.pending_frames.len() / 3, 4);
}

#[test]
fn pcm_stream_buffers_short_input_and_rejects_rate_mismatch() {
    let model = tiny_model();
    let mut stream = model.open_stream();
    assert!(stream.push_pcm(&[0.0; 200], 16_000).unwrap().is_empty());
    assert!(
        stream
            .push_pcm(&[], 8_000)
            .unwrap_err()
            .to_string()
            .contains("sample rate")
    );
}

#[test]
fn reset_clears_frontend_and_network_state() {
    let model = tiny_model();
    let mut stream = stream_from(&model);
    stream.pending_pcm.push(1.0);
    stream.pending_frames.push(2.0);
    stream.lfr_initialized = true;
    stream.reset();
    assert!(stream.pending_pcm.is_empty());
    assert!(stream.pending_frames.is_empty());
    assert!(!stream.lfr_initialized);
    assert!(stream.state.is_zero());
}
