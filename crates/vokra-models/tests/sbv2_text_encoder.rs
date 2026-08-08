//! `SbV2TextEncoder` + `BertBridge` tests (Task 17; M6 refactor 2026-08-06
//! replaced `word_boundaries: &[bool]` with `language_id: u8` — see
//! `SbV2TextEncoder`'s `language_embed` design-correction doc for the
//! primary-source reason. The tests below cover the new signature end to
//! end: shape invariant, determinism, and a per-language distinctness
//! check that pins the row-ordering convention).

use vokra_models::sbv2::{BertBridge, N_LANGUAGES, SbV2TextEncoder};

/// `SbV2TextEncoder::forward` returns a flat `[seq_len * d_model]` buffer
/// (empty `transformer_layers` — the tested no-op stack configuration).
#[test]
fn text_encoder_forward_shape() {
    let (n_vocab, n_tones, d_model) = (8, 3, 4);
    let enc = SbV2TextEncoder::from_weights(
        vec![0.1; n_vocab * d_model],
        vec![0.2; n_tones * d_model],
        vec![0.3; N_LANGUAGES * d_model],
        Vec::new(),
        d_model,
        n_vocab,
        n_tones,
    );
    let phoneme_ids: [u16; 5] = [0, 1, 2, 3, 4];
    let tones: [u8; 5] = [0, 1, 2, 0, 1];

    let out = enc.forward(&phoneme_ids, &tones, /*language_id=*/ 0);

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
        vec![0.02; N_LANGUAGES * d_model],
        Vec::new(),
        d_model,
        n_vocab,
        n_tones,
    );
    let phoneme_ids: [u16; 5] = [7, 0, 4, 4, 2];
    let tones: [u8; 5] = [2, 0, 1, 1, 0];

    let out1 = enc.forward(&phoneme_ids, &tones, /*language_id=*/ 1);
    let out2 = enc.forward(&phoneme_ids, &tones, /*language_id=*/ 1);

    assert_eq!(out1, out2, "identical inputs must produce identical output");
}

/// M6 refactor: different `language_id` values with otherwise-identical
/// inputs must select different rows of `language_embed`, changing the
/// broadcast additive contribution to every position. Uses a
/// `language_embed` whose rows have distinct constant fills so the
/// per-position delta between two language ids equals exactly `row_a -
/// row_b` on every element (a check that any per-position broadcast add
/// must satisfy).
#[test]
fn text_encoder_forward_language_id_switches_embedding_row() {
    let (n_vocab, n_tones, d_model) = (8, 3, 4);
    // Row 0 = 0.10s, row 1 = 0.20s, row 2 = 0.30s — three distinct
    // per-row constants so a swap between any two language ids produces
    // an easily-predictable per-element delta.
    let mut language_embed = Vec::with_capacity(N_LANGUAGES * d_model);
    for row in 0..N_LANGUAGES {
        let fill = 0.1 * (row + 1) as f32;
        language_embed.extend(std::iter::repeat_n(fill, d_model));
    }
    let enc = SbV2TextEncoder::from_weights(
        vec![0.0; n_vocab * d_model],
        vec![0.0; n_tones * d_model],
        language_embed,
        Vec::new(),
        d_model,
        n_vocab,
        n_tones,
    );
    let phoneme_ids: [u16; 3] = [0, 1, 2];
    let tones: [u8; 3] = [0, 0, 0];

    let out_ja = enc.forward(&phoneme_ids, &tones, /*language_id=*/ 0);
    let out_en = enc.forward(&phoneme_ids, &tones, /*language_id=*/ 1);
    let out_zh = enc.forward(&phoneme_ids, &tones, /*language_id=*/ 2);

    assert_ne!(out_ja, out_en, "language_id 0 vs 1 must differ");
    assert_ne!(out_en, out_zh, "language_id 1 vs 2 must differ");
    assert_ne!(out_ja, out_zh, "language_id 0 vs 2 must differ");

    // With phoneme_embed / tone_embed / transformer stack all zero, the
    // output equals `language_embed[language_id]` broadcast to every
    // position and then multiplied by `sqrt(d_model)` — which matches
    // upstream VITS's `TextEncoder.forward`'s `x = self.emb(x) *
    // math.sqrt(self.hidden_channels)` (the M6 refactor added this scale
    // to match the real relative-position transformer encoder's forward
    // pass; see `SbV2TextEncoder::forward`'s doc). For `d_model = 4`,
    // `sqrt(4) = 2` exactly, so row-0 (fill 0.10) → 0.20, row-1 (0.20)
    // → 0.40, row-2 (0.30) → 0.60.
    let scale = (d_model as f32).sqrt();
    assert!(
        out_ja.iter().all(|&v| (v - 0.10 * scale).abs() < 1e-6),
        "with zero phoneme/tone weights, out_ja must be all 0.10 * sqrt(d_model) (row 0 fill)"
    );
    assert!(
        out_en.iter().all(|&v| (v - 0.20 * scale).abs() < 1e-6),
        "with zero phoneme/tone weights, out_en must be all 0.20 * sqrt(d_model) (row 1 fill)"
    );
    assert!(
        out_zh.iter().all(|&v| (v - 0.30 * scale).abs() < 1e-6),
        "with zero phoneme/tone weights, out_zh must be all 0.30 * sqrt(d_model) (row 2 fill)"
    );
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

/// BERT-BRIDGE-LINEAR fix (2026-08-09) parity pin.
///
/// Pre-fix `BertBridge::forward` used nearest-neighbor floor
/// interpolation: `s = min((t * bert_seq_len) / text_seq_len,
/// bert_seq_len - 1)`. Python reference uses
/// `F.interpolate(mode='linear', align_corners=False)` which computes
/// `src_x = (dst_x + 0.5) * bert_seq_len / text_seq_len - 0.5` and
/// linearly interpolates between `floor(src_x)` and `floor(src_x)+1`
/// clamped to `[0, bert_seq_len-1]`.
///
/// This test picks a non-integer resample ratio (`text_seq_len=8`,
/// `bert_seq_len=3`) where the two schemes disagree at multiple
/// destination positions, and verifies the Rust output matches the
/// hand-computed align_corners=False formula.
#[test]
fn bert_bridge_uses_linear_align_corners_false_interpolation() {
    let d_bert = 2;
    let d_target = 2;
    let text_seq_len = 8;
    let bert_seq_len = 3;
    // Identity projection: `conv_weight = I_2`, `conv_bias = 0`. Then
    // `projected == bert_hidden` and we can reason about
    // interpolation in isolation from the linear projection.
    let bridge = BertBridge::from_conv(
        vec![1.0, 0.0, 0.0, 1.0], // [d_target=2, d_bert=2] row-major identity
        vec![0.0; d_target],
        d_bert,
        d_target,
    );
    // Deliberate row-distinctive values so linear vs nearest picks
    // observably different outputs.
    // Row 0 = [10.0, 20.0], Row 1 = [30.0, 40.0], Row 2 = [50.0, 60.0].
    let bert_hidden = vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];

    let out = bridge.forward(&bert_hidden, text_seq_len, bert_seq_len);
    assert_eq!(out.len(), text_seq_len * d_target);

    // Reference (align_corners=False linear):
    for t in 0..text_seq_len {
        let src_x = ((t as f32) + 0.5) * (bert_seq_len as f32) / (text_seq_len as f32) - 0.5;
        let low_f = src_x.floor();
        let low = (low_f as i64).clamp(0, (bert_seq_len as i64) - 1) as usize;
        let high = ((low_f as i64) + 1).clamp(0, (bert_seq_len as i64) - 1) as usize;
        let alpha = (src_x - low_f).clamp(0.0, 1.0);
        // If clamp pushed low_f out of range, alpha collapses so the
        // clamped index still dominates.
        for d in 0..d_target {
            let low_val = bert_hidden[low * d_bert + d];
            let high_val = bert_hidden[high * d_bert + d];
            let expected = (1.0 - alpha) * low_val + alpha * high_val;
            let got = out[t * d_target + d];
            let delta = (got - expected).abs();
            assert!(
                delta < 1e-4,
                "bert_bridge linear-interp mismatch at (t={t}, d={d}): expected {expected}, \
                 got {got} (Δ = {delta}). src_x={src_x}, low={low}, high={high}, alpha={alpha}. \
                 Pre-fix nearest-neighbor would produce very different values on this fixture; \
                 post-fix must match the align_corners=False linear formula."
            );
        }
    }
}

/// Regression: smallest valid `bert_seq_len` (1) must not panic; verifies
/// the `debug_assert` boundary added for the empty-`bert_seq_len` defect
/// (Task 17 review) and confirms interpolation correctly
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
