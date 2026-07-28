//! `SbV2TextEncoder` + `BertBridge` tests (Task 17).

use vokra_models::sbv2::{BertBridge, SbV2TextEncoder};

/// `SbV2TextEncoder::forward` returns a flat `[seq_len * d_model]` buffer
/// (empty `transformer_layers` — the tested no-op stack configuration).
#[test]
fn text_encoder_forward_shape() {
    let (n_vocab, n_tones, d_model) = (8, 3, 4);
    let enc = SbV2TextEncoder::from_weights(
        vec![0.1; n_vocab * d_model],
        vec![0.2; n_tones * d_model],
        vec![0.3; 2 * d_model],
        Vec::new(),
        d_model,
        n_vocab,
        n_tones,
    );
    let phoneme_ids: [u16; 5] = [0, 1, 2, 3, 4];
    let tones: [u8; 5] = [0, 1, 2, 0, 1];
    let word_boundaries = [true, false, true, false, false];

    let out = enc.forward(&phoneme_ids, &tones, &word_boundaries);

    assert_eq!(out.len(), phoneme_ids.len() * d_model);
}

/// `SbV2TextEncoder::forward` is deterministic: identical inputs produce
/// identical output vectors.
#[test]
fn text_encoder_forward_deterministic() {
    let (n_vocab, n_tones, d_model) = (8, 3, 4);
    let enc = SbV2TextEncoder::from_weights(
        vec![0.05; n_vocab * d_model],
        vec![-0.1; n_tones * d_model],
        vec![0.02; 2 * d_model],
        Vec::new(),
        d_model,
        n_vocab,
        n_tones,
    );
    let phoneme_ids: [u16; 5] = [7, 0, 4, 4, 2];
    let tones: [u8; 5] = [2, 0, 1, 1, 0];
    let word_boundaries = [true, false, false, true, false];

    let out1 = enc.forward(&phoneme_ids, &tones, &word_boundaries);
    let out2 = enc.forward(&phoneme_ids, &tones, &word_boundaries);

    assert_eq!(out1, out2, "identical inputs must produce identical output");
}

/// `BertBridge::forward` returns a flat `[text_seq_len * d_target]`
/// buffer.
#[test]
fn bert_bridge_forward_shape() {
    let (d_bert, d_target) = (6, 4);
    let (text_seq_len, bert_seq_len) = (5, 3);
    let bridge = BertBridge::from_conv(
        vec![0.1; d_target * d_bert],
        vec![0.01; d_target],
        d_bert,
        d_target,
    );
    let bert_hidden = vec![0.5_f32; bert_seq_len * d_bert];

    let out = bridge.forward(&bert_hidden, text_seq_len, bert_seq_len);

    assert_eq!(out.len(), text_seq_len * d_target);
}

/// Zero `conv_weight`/`conv_bias` makes `BertBridge::forward` return an
/// all-zero contribution — the additive identity, so a caller (e.g.
/// `SbV2Model`) can no-op the BERT contribution entirely by zeroing this
/// bridge's weights instead of special-casing "no BERT" at the call site.
#[test]
fn bert_bridge_zero_weights_returns_zeros() {
    let (d_bert, d_target) = (6, 4);
    let (text_seq_len, bert_seq_len) = (5, 3);
    let bridge = BertBridge::from_conv(
        vec![0.0; d_target * d_bert],
        vec![0.0; d_target],
        d_bert,
        d_target,
    );
    // Arbitrary, nonzero bert_hidden -- must not matter, since every
    // weight touching it is zero.
    let bert_hidden: Vec<f32> = (0..bert_seq_len * d_bert)
        .map(|i| i as f32 * 0.37 - 1.0)
        .collect();

    let out = bridge.forward(&bert_hidden, text_seq_len, bert_seq_len);

    assert!(
        out.iter().all(|&x| x == 0.0),
        "zero conv weights must give an all-zero contribution"
    );
}

/// Regression: smallest valid `bert_seq_len` (1) must not panic; verifies
/// the `debug_assert` boundary added for the empty-`bert_seq_len` defect
/// (Task 17 review) and confirms nearest-neighbor interpolation correctly
/// collapses every text position onto the single source row.
#[test]
fn bert_bridge_single_bert_position_no_panic() {
    let d_bert = 4;
    let d_target = 3;
    let text_seq_len = 5;
    let bert_seq_len = 1;
    let bridge = BertBridge::from_conv(
        vec![0.1; d_target * d_bert],
        vec![0.0; d_target],
        d_bert,
        d_target,
    );
    let bert_hidden = vec![1.0_f32; bert_seq_len * d_bert];

    let out = bridge.forward(&bert_hidden, text_seq_len, bert_seq_len);

    assert_eq!(out.len(), text_seq_len * d_target);
    // With bert_seq_len=1, every text position must map to source 0 -> identical d_target chunks.
    let first_chunk = &out[..d_target];
    for t in 1..text_seq_len {
        let chunk = &out[t * d_target..(t + 1) * d_target];
        assert_eq!(chunk, first_chunk, "single-source interp must broadcast");
    }
}
