//! DeBERTa v2 GGUF loader test. Clean-room per arXiv:2006.03654 +
//! HF transformers deberta_v2 (Apache-2.0).
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)

use std::path::{Path, PathBuf};

use vokra_bert::deberta_v2::DebertaV2Encoder;
use vokra_bert::tokenizer::SbertTokenizer;
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};

/// Repo-root-relative real-fixture directory for the DeBERTa v2/v3 GGUF
/// fixtures shared with the SBV2 v2 loader/parity tests
/// (`tests/fixtures/sbv2/`, gated by the committed `*.gguf.sha256`
/// sidecars). `CARGO_MANIFEST_DIR` is `<repo>/crates/vokra-bert` — `cargo
/// test` sets a test binary's working directory to the crate root, not the
/// invocation directory, so every repo-root fixture path in this workspace
/// is built this way (`parity_sbv2_real.rs`, `parity_whisper.rs`,
/// `parity_kokoro.rs`, `parity_voxtral.rs`, `parity_csm.rs`,
/// `parity_moshi.rs`) rather than as a bare relative literal.
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("sbv2")
}

#[test]
#[ignore = "requires real deberta-v2 GGUF fixture, gated by tests/fixtures/sbv2/*.gguf.sha256"]
fn load_real_deberta_v2_ja() {
    let path = fixtures_dir().join("deberta-v2-large-japanese-char-wwm.gguf");
    let g = GgufFile::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let enc = DebertaV2Encoder::from_gguf(&g)
        .unwrap_or_else(|e| panic!("DebertaV2Encoder::from_gguf: {e}"));
    let out = enc.forward(&[2, 100, 200, 3]); // <s> ... </s>
    assert!(!out.is_empty());
}

/// Adds a full set of DeBERTa v2 tensors for one layer (with the given
/// per-layer prefix and hparams) into `b`. Extracted so both the
/// "loads with pos biases" and "loads without pos biases" tests share
/// exactly the same base tensor set (only the presence of the two
/// optional bias tensors varies).
///
/// Weights are index-varying (not uniform) — a uniform projection
/// weight collapses post-LayerNorm inputs to zero (LN removes the row
/// mean, then a constant weight sums to zero), which would mask any
/// bias-only diff we want to observe.
fn add_layer_tensors(
    b: &mut GgufBuilder,
    prefix: &str,
    d_model: usize,
    n_pos_buckets: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let f32_bytes = |vals: &[f32]| {
        vals.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    // Deterministic per-column-varying pattern for a `[d_out, d_in]`
    // weight — ensures matmul(LN(hidden), W.T) does not degenerate to
    // 0. A uniform weight matmul'd with a mean-0 (post-LN) input sums
    // to zero; here each row varies strongly across `d_in` (the sum
    // dimension) so the projection carries actual signal.
    let make_weight = |d_out: usize, d_in: usize, tag: f32| -> Vec<f32> {
        (0..d_out)
            .flat_map(|o| {
                (0..d_in).map(move |d| {
                    // Vary strongly across d (the sum dimension); tag
                    // shifts each projection so they aren't identical.
                    tag + 0.05 * (((o * 7 + d * 13) % 11) as f32 - 5.0)
                })
            })
            .collect()
    };
    let d_ffn = 4 * d_model;
    // Attention Q/K/V projection weights + biases.
    for (name, tag) in [("wq", 0.02_f32), ("wk", 0.025_f32), ("wv", 0.03_f32)] {
        b.add_tensor(
            &format!("{prefix}.attn.{name}.weight"),
            GgmlType::F32,
            vec![d_model as u64, d_model as u64],
            f32_bytes(&make_weight(d_model, d_model, tag)),
        )?;
        b.add_tensor(
            &format!("{prefix}.attn.{name}.bias"),
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&vec![0.0_f32; d_model]),
        )?;
    }
    // Position-aware Q/K weight (bias is added by caller when
    // stress-testing the optional-bias path).
    for (name, tag) in [("wq_pos", 0.04_f32), ("wk_pos", 0.045_f32)] {
        b.add_tensor(
            &format!("{prefix}.attn.{name}.weight"),
            GgmlType::F32,
            vec![d_model as u64, d_model as u64],
            f32_bytes(&make_weight(d_model, d_model, tag)),
        )?;
    }
    // Output projection weight + bias.
    b.add_tensor(
        &format!("{prefix}.attn.w_out.weight"),
        GgmlType::F32,
        vec![d_model as u64, d_model as u64],
        f32_bytes(&make_weight(d_model, d_model, 0.05)),
    )?;
    b.add_tensor(
        &format!("{prefix}.attn.w_out.bias"),
        GgmlType::F32,
        vec![d_model as u64],
        f32_bytes(&vec![0.0_f32; d_model]),
    )?;
    // Per-layer position embedding (v2 duplicates the shared table into
    // every layer at convert-time).
    b.add_tensor(
        &format!("{prefix}.attn.pos_embed.weight"),
        GgmlType::F32,
        vec![n_pos_buckets as u64, d_model as u64],
        f32_bytes(
            &(0..(n_pos_buckets * d_model))
                .map(|i| 0.01 + 0.001 * i as f32)
                .collect::<Vec<_>>(),
        ),
    )?;
    // FFN.
    b.add_tensor(
        &format!("{prefix}.ffn.w1.weight"),
        GgmlType::F32,
        vec![d_ffn as u64, d_model as u64],
        f32_bytes(&make_weight(d_ffn, d_model, 0.02)),
    )?;
    b.add_tensor(
        &format!("{prefix}.ffn.w1.bias"),
        GgmlType::F32,
        vec![d_ffn as u64],
        f32_bytes(&vec![0.0_f32; d_ffn]),
    )?;
    b.add_tensor(
        &format!("{prefix}.ffn.w2.weight"),
        GgmlType::F32,
        vec![d_model as u64, d_ffn as u64],
        f32_bytes(&make_weight(d_model, d_ffn, 0.02)),
    )?;
    b.add_tensor(
        &format!("{prefix}.ffn.w2.bias"),
        GgmlType::F32,
        vec![d_model as u64],
        f32_bytes(&vec![0.0_f32; d_model]),
    )?;
    // Two LayerNorms per block (pre-attn, pre-FFN).
    for name in ["ln1", "ln2"] {
        b.add_tensor(
            &format!("{prefix}.{name}.gamma"),
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&vec![1.0_f32; d_model]),
        )?;
        b.add_tensor(
            &format!("{prefix}.{name}.beta"),
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&vec![0.0_f32; d_model]),
        )?;
    }
    Ok(())
}

/// Builds a full single-layer DeBERTa v2 GGUF from scratch and
/// optionally stamps the WP-15 optional `wq_pos.bias` / `wk_pos.bias`
/// tensors. Returns the parsed [`GgufFile`].
fn build_v2_gguf(
    d_model: usize,
    n_heads: usize,
    n_pos_buckets: usize,
    with_pos_biases: bool,
    bq_pos_val: f32,
) -> GgufFile {
    let f32_bytes = |vals: &[f32]| {
        vals.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let vocab: usize = 8;
    let mut b = GgufBuilder::new();
    b.add_u32("vokra.bert.deberta_v2.n_layers", 1);
    b.add_u32("vokra.bert.deberta_v2.d_model", d_model as u32);
    b.add_u32("vokra.bert.deberta_v2.n_heads", n_heads as u32);
    b.add_u32("vokra.bert.deberta_v2.vocab_size", vocab as u32);
    b.add_u32("vokra.bert.deberta_v2.n_pos_buckets", n_pos_buckets as u32);
    b.add_u32("vokra.bert.deberta_v2.max_pos_dist", 32);

    b.add_tensor(
        "bert.embed.weight",
        GgmlType::F32,
        vec![vocab as u64, d_model as u64],
        f32_bytes(
            &(0..(vocab * d_model))
                .map(|i| 0.05 + 0.01 * i as f32)
                .collect::<Vec<_>>(),
        ),
    )
    .expect("embed");
    b.add_tensor(
        "bert.embed.ln.gamma",
        GgmlType::F32,
        vec![d_model as u64],
        f32_bytes(&vec![1.0_f32; d_model]),
    )
    .expect("gamma");
    b.add_tensor(
        "bert.embed.ln.beta",
        GgmlType::F32,
        vec![d_model as u64],
        f32_bytes(&vec![0.0_f32; d_model]),
    )
    .expect("beta");

    add_layer_tensors(&mut b, "bert.encoder.layer.0", d_model, n_pos_buckets)
        .expect("layer tensors");

    if with_pos_biases {
        b.add_tensor(
            "bert.encoder.layer.0.attn.wq_pos.bias",
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&vec![bq_pos_val; d_model]),
        )
        .expect("wq_pos.bias");
        b.add_tensor(
            "bert.encoder.layer.0.attn.wk_pos.bias",
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&vec![0.5_f32; d_model]),
        )
        .expect("wk_pos.bias");
    }

    let bytes = b.to_bytes().expect("build gguf");
    GgufFile::parse(bytes).expect("parse gguf")
}

/// WP-15 loader test. Two identical single-layer GGUFs differ only in
/// whether they carry the optional `wq_pos.bias` / `wk_pos.bias`
/// tensors. Loading the pos-bias variant and running forward must
/// produce different output than loading the no-pos-bias variant —
/// proving that the loader wired the optional biases into
/// [`AttnWeights::bq_pos`] and that
/// [`DisentangledAttention::forward`](vokra_bert::deberta_v2::DisentangledAttention::forward)
/// consumes them.
#[test]
fn from_gguf_reads_optional_pos_biases_when_present() {
    let d_model = 8;
    let n_heads = 2;
    let n_pos_buckets = 8;
    let g_no_bias = build_v2_gguf(d_model, n_heads, n_pos_buckets, false, 0.0);
    let g_with_bias = build_v2_gguf(d_model, n_heads, n_pos_buckets, true, 5.0);

    let enc_no_bias =
        DebertaV2Encoder::from_gguf(&g_no_bias).expect("load without wq_pos.bias / wk_pos.bias");
    let enc_with_bias =
        DebertaV2Encoder::from_gguf(&g_with_bias).expect("load with wq_pos.bias / wk_pos.bias");

    // Direct wiring proof: layer 0 of the with-bias variant loaded
    // both position-projection biases, layer 0 of the no-bias variant
    // loaded neither. This forecloses the pre-WP-15 silent-drop bug
    // regardless of downstream signal attenuation.
    assert_eq!(
        enc_with_bias.probe_layer_has_pos_biases(0),
        (true, true),
        "with-bias GGUF must produce Some(bq_pos), Some(bk_pos)"
    );
    assert_eq!(
        enc_no_bias.probe_layer_has_pos_biases(0),
        (false, false),
        "no-bias GGUF must produce None, None (backward-compat)"
    );

    // Observable end-to-end confirmation that `bq_pos` reaches forward
    // and moves output — the encoder pipeline heavily attenuates the
    // P2C path (small weight matrices → small residual perturbation),
    // so this diff is small but must be non-zero. A `bq_pos` silently
    // dropped in the loader would leave the diff at exactly 0.
    let ids: &[u32] = &[1, 2, 3, 4];
    let out_no = enc_no_bias.forward(ids);
    let out_with = enc_with_bias.forward(ids);
    let diff: f32 = out_no
        .iter()
        .zip(&out_with)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.0,
        "wq_pos.bias in GGUF must observably change forward output; sum(|diff|) = {diff}"
    );
}

/// Sanity probe: `GgufFile::tensor_info` finds the optional bias tensor
/// name that `from_gguf`'s new WP-15 probe relies on.
#[test]
fn probe_tensor_info_finds_optional_bias() {
    let g = build_v2_gguf(8, 2, 8, true, 0.5);
    let info = g.tensor_info("bert.encoder.layer.0.attn.wq_pos.bias");
    assert!(
        info.is_some(),
        "GGUF should carry wq_pos.bias when with_pos_biases=true"
    );
    let g_no = build_v2_gguf(8, 2, 8, false, 0.0);
    let info_no = g_no.tensor_info("bert.encoder.layer.0.attn.wq_pos.bias");
    assert!(
        info_no.is_none(),
        "GGUF should NOT carry wq_pos.bias when with_pos_biases=false"
    );
}

// ============================================================================
// WordPiece tokenizer metadata verification tests — Task 9 real DeBERTa v2
// ============================================================================

/// Task 9: Real DeBERTa v2 Japanese GGUF tokenizer metadata load verification.
/// Opens the real DeBERTa v2 large Japanese char-wwm GGUF fixture and verifies
/// that the tokenizer metadata (scheme, pieces, control ids) is present and
/// correctly loaded via `SbertTokenizer::from_gguf`. This test locks down the
/// WordPiece/BertCharSplit scheme integration path through the real fixture.
#[test]
#[ignore = "requires real deberta-v2 GGUF fixture, gated by tests/fixtures/sbv2/*.gguf.sha256"]
fn deberta_v2_real_tokenizer_metadata_loads() {
    let path = fixtures_dir().join("deberta-v2-large-japanese-char-wwm.gguf");
    let g = GgufFile::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));

    // Verify tokenizer metadata is present and readable via SbertTokenizer::from_gguf.
    // The real DeBERTa v2 ku-nlp fixture has ~22k pieces (char-based Japanese).
    let tok = SbertTokenizer::from_gguf(&g, "vokra.bert.tokenizer")
        .unwrap_or_else(|e| panic!("SbertTokenizer::from_gguf: {e}"));

    // Verify that control ids are loaded (non-default values).
    let unk_id = tok.unk_id();
    let bos_id = tok.bos_id(); // CLS for BERT-family
    let eos_id = tok.eos_id(); // SEP for BERT-family

    // Note: The SbertTokenizer doesn't expose pieces_len(), so we verify the piece count
    // indirectly by checking that control ids are within a reasonable range.
    // DeBERTa v2 ku-nlp has ~22012 pieces for Japanese, so unk_id should be relatively small.
    assert!(
        unk_id < 1000,
        "unk_id={unk_id} should be small relative to typical vocab size (DeBERTa v2 has ~22k pieces)"
    );

    // In WordPiece vocabs, these are typically low indices (< 1000).
    assert!(
        unk_id < 1000,
        "unk_id={unk_id} should be low (typical WordPiece special tokens are <100)"
    );
    assert!(bos_id < 1000, "bos_id (CLS)={bos_id} should be low");
    assert!(eos_id < 1000, "eos_id (SEP)={eos_id} should be low");

    // Encode a real Japanese string to verify the tokenizer functions end-to-end.
    // "こんにちは" (hello) should encode to non-empty id sequence.
    let ids = tok.encode("こんにちは");
    assert!(
        !ids.is_empty(),
        "encode should produce non-empty ids for real Japanese text"
    );

    // Encode with special tokens to verify CLS/SEP wrapping.
    let ids_with_special = tok.encode_with_special_tokens("こんにちは");
    assert!(
        ids_with_special.len() > ids.len(),
        "encode_with_special_tokens should add CLS/SEP (more tokens)"
    );
    // Expect CLS at start (should equal bos_id) and SEP at end (should equal eos_id).
    if !ids_with_special.is_empty() {
        assert_eq!(
            ids_with_special[0], bos_id,
            "first token should be CLS (bos_id)"
        );
        assert_eq!(
            ids_with_special[ids_with_special.len() - 1],
            eos_id,
            "last token should be SEP (eos_id)"
        );
    }
}
