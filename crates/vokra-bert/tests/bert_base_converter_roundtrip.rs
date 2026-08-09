//! WP-14 (2026-08-10): converter → loader round-trip for plain BERT.
//!
//! Mirror of `deberta_converter_roundtrip.rs`: build a shape-complete
//! synthetic safetensors and a tiny WordPiece vocab.txt, feed both
//! through [`vokra_convert::convert_bert_base_file`], then load the
//! emitted GGUF with both
//! [`vokra_bert::bert_base::BertBaseEncoder::from_gguf`] and
//! [`vokra_bert::wordpiece::BertWordpieceTokenizer::from_gguf`]. The
//! loader bails on the first missing tensor or metadata key, so this is
//! a fully load-bearing check that the converter's rename table,
//! hparam chunk group, and tokenizer prefix produce exactly the names
//! the two loaders read.
//!
//! Tiny hparams (`n_layers=2, hidden=8, heads=2, vocab=6, max_pos=8,
//! type_vocab=2, ffn=32`) keep the safetensors small and the test
//! fast — the mapping table is the contract, not the shape.
//!
//! The runtime `forward()` is exercised on a 4-token input to prove
//! the encoder is actually usable (would panic on a shape mismatch
//! propagated from a mis-mapped tensor), matching the assertion style
//! of the DeBERTa round-trip.
//!
//! # Placement rationale
//!
//! This test lives in `vokra-bert/tests/` alongside
//! `deberta_converter_roundtrip.rs` rather than in `vokra-convert/tests/`
//! because `vokra-bert` already carries `vokra-convert` as a
//! `[dev-dependencies]` edge for that pattern — the reverse direction
//! (`vokra-convert` → `vokra-bert` dev-dep) would introduce a Cargo
//! dev-dependency cycle that has no functional gain. WP-14's task
//! description hints at `vokra-convert/tests/`; the existing DeBERTa
//! precedent is the load-bearing rule here. Same test surface, no new
//! cycle, closes the same round-trip proof.

use std::path::PathBuf;

use vokra_bert::bert_base::BertBaseEncoder;
use vokra_bert::wordpiece::BertWordpieceTokenizer;
use vokra_convert::convert_bert_base_file;
use vokra_core::gguf::GgufFile;

/// f32 slice → little-endian byte payload (matches `SafetensorsFile::parse`).
fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Assembles a safetensors byte buffer from `(name, dtype, shape, payload)`.
/// Mirror of the DeBERTa round-trip helper.
fn safetensors_multi(entries: &[(&str, &str, &[u64], Vec<u8>)]) -> Vec<u8> {
    let mut body = Vec::new();
    let mut parts = Vec::new();
    let mut cursor: usize = 0;
    for (name, dtype, shape, payload) in entries {
        let start = cursor;
        let end = cursor + payload.len();
        let shape_str = shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        parts.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[{start},{end}]}}"#
        ));
        body.extend_from_slice(payload);
        cursor = end;
    }
    let header = format!("{{{}}}", parts.join(","));
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&body);
    out
}

/// Tiny but shape-complete plain-BERT fixture — every tensor the
/// loader reads is emitted with the correct shape:
/// `n_layers = 2`, `hidden = 8`, `heads = 2` (via `derive_heads` floor),
/// `vocab = 6`, `max_pos = 8`, `type_vocab = 2`, `ffn = 32` (4 × hidden).
///
/// `hidden = 8` gives `head_dim = 4` (not the HF-canonical 64 — the
/// synthetic geometry is chosen to divide evenly by the synthetic
/// `heads = 2` after the floor-of-1 fallback, so
/// [`BertBaseEncoder::from_gguf`]'s
/// `hidden % heads == 0` divisibility guard is satisfied without
/// needing a full 1024-hidden `-large` fixture).
fn bert_full_fixture() -> Vec<u8> {
    let n_layers: usize = 2;
    let hidden: usize = 8;
    let vocab: usize = 6;
    let max_pos: usize = 8;
    let type_vocab: usize = 2;
    let ffn: usize = 4 * hidden;

    let mut entries: Vec<(&'static str, &'static str, Vec<u64>, Vec<u8>)> = vec![
        // Embeddings.
        (
            "bert.embeddings.word_embeddings.weight",
            "F32",
            vec![vocab as u64, hidden as u64],
            f32_bytes(&vec![0.01f32; vocab * hidden]),
        ),
        (
            "bert.embeddings.position_embeddings.weight",
            "F32",
            vec![max_pos as u64, hidden as u64],
            f32_bytes(&vec![0.02f32; max_pos * hidden]),
        ),
        (
            "bert.embeddings.token_type_embeddings.weight",
            "F32",
            vec![type_vocab as u64, hidden as u64],
            f32_bytes(&vec![0.03f32; type_vocab * hidden]),
        ),
        (
            "bert.embeddings.LayerNorm.weight",
            "F32",
            vec![hidden as u64],
            f32_bytes(&vec![1.0f32; hidden]),
        ),
        (
            "bert.embeddings.LayerNorm.bias",
            "F32",
            vec![hidden as u64],
            f32_bytes(&vec![0.0f32; hidden]),
        ),
    ];

    // Per-layer tensors — the loader reads exactly these names.
    for i in 0..n_layers {
        let prefix = format!("bert.encoder.layer.{i}");
        // Attention Q/K/V projections (weight + bias).
        for proj in ["query", "key", "value"] {
            entries.push((
                Box::leak(format!("{prefix}.attention.self.{proj}.weight").into_boxed_str()),
                "F32",
                vec![hidden as u64, hidden as u64],
                f32_bytes(&vec![0.01f32; hidden * hidden]),
            ));
            entries.push((
                Box::leak(format!("{prefix}.attention.self.{proj}.bias").into_boxed_str()),
                "F32",
                vec![hidden as u64],
                f32_bytes(&vec![0.0f32; hidden]),
            ));
        }
        // Attention output projection (weight + bias) + post-norm.
        entries.push((
            Box::leak(format!("{prefix}.attention.output.dense.weight").into_boxed_str()),
            "F32",
            vec![hidden as u64, hidden as u64],
            f32_bytes(&vec![0.01f32; hidden * hidden]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.attention.output.dense.bias").into_boxed_str()),
            "F32",
            vec![hidden as u64],
            f32_bytes(&vec![0.0f32; hidden]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.attention.output.LayerNorm.weight").into_boxed_str()),
            "F32",
            vec![hidden as u64],
            f32_bytes(&vec![1.0f32; hidden]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.attention.output.LayerNorm.bias").into_boxed_str()),
            "F32",
            vec![hidden as u64],
            f32_bytes(&vec![0.0f32; hidden]),
        ));
        // FFN intermediate (w1) and output (w2) + post-norm.
        entries.push((
            Box::leak(format!("{prefix}.intermediate.dense.weight").into_boxed_str()),
            "F32",
            vec![ffn as u64, hidden as u64],
            f32_bytes(&vec![0.01f32; ffn * hidden]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.intermediate.dense.bias").into_boxed_str()),
            "F32",
            vec![ffn as u64],
            f32_bytes(&vec![0.0f32; ffn]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.output.dense.weight").into_boxed_str()),
            "F32",
            vec![hidden as u64, ffn as u64],
            f32_bytes(&vec![0.01f32; hidden * ffn]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.output.dense.bias").into_boxed_str()),
            "F32",
            vec![hidden as u64],
            f32_bytes(&vec![0.0f32; hidden]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.output.LayerNorm.weight").into_boxed_str()),
            "F32",
            vec![hidden as u64],
            f32_bytes(&vec![1.0f32; hidden]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.output.LayerNorm.bias").into_boxed_str()),
            "F32",
            vec![hidden as u64],
            f32_bytes(&vec![0.0f32; hidden]),
        ));
    }
    let borrowed: Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> = entries
        .into_iter()
        .map(|(n, d, s, p)| {
            let leaked: &'static [u64] = Box::leak(s.into_boxed_slice());
            (n, d, leaked, p)
        })
        .collect();
    safetensors_multi(&borrowed)
}

fn temp_pair(label: &str) -> (PathBuf, PathBuf) {
    let mut input = std::env::temp_dir();
    input.push(format!(
        "vokra-bert-base-conv-roundtrip-{label}-{}-in.safetensors",
        std::process::id()
    ));
    let mut output = std::env::temp_dir();
    output.push(format!(
        "vokra-bert-base-conv-roundtrip-{label}-{}-out.gguf",
        std::process::id()
    ));
    (input, output)
}

/// Load-bearing round-trip: convert a shape-complete plain-BERT
/// safetensors, then instantiate the runtime encoder from the emitted
/// GGUF. Must succeed with **zero missing tensors** — the loader's
/// `load_tensor_f32` bails on the first absence, so this is a full
/// end-to-end check that the converter's rename table produces exactly
/// the canonical `bert_base.*` names.
#[test]
fn converter_output_loads_via_bert_base_from_gguf_and_forward_runs() {
    let (input, output) = temp_pair("full");
    let blob = bert_full_fixture();
    std::fs::write(&input, &blob).expect("write input safetensors");

    let report = convert_bert_base_file(&input, &output, None, None, false)
        .expect("bert_base converter must succeed on the full fixture");
    // Expected count: 5 embedding tensors (word/pos/type_type embeds +
    // 2 LN) + n_layers × 16 per-layer tensors:
    //   - 3 Q/K/V projections × (weight + bias) = 6
    //   - attention.output.dense × (weight + bias) = 2
    //   - attention.output.LayerNorm × (γ + β) = 2
    //   - intermediate.dense × (weight + bias) = 2
    //   - output.dense × (weight + bias) = 2
    //   - output.LayerNorm × (γ + β) = 2
    //   → 16 per layer × 2 = 32, + 5 embeddings = 37.
    // A different count means a rename table gap (silent drop) or an
    // accidental duplicate — either way, catch it here rather than at
    // the loader failure site downstream.
    assert_eq!(
        report.written, 37,
        "expected 37 renamed tensors for n_layers=2 full fixture (5 embed + 2×16 per-layer), got {}",
        report.written
    );
    assert_eq!(
        report.skipped_unmapped, 0,
        "no upstream tensor should be unmapped"
    );
    assert_eq!(
        report.skipped_non_float, 0,
        "no non-float tensors in the fixture"
    );

    let g = GgufFile::open(&output).expect("open emitted BertBase GGUF");

    let enc = BertBaseEncoder::from_gguf(&g)
        .expect("BertBaseEncoder::from_gguf must load converter output");
    assert_eq!(enc.d_model(), 8, "hidden must be 8 as declared in fixture");

    // Sanity: the encoder actually forwards.
    let out = enc.forward(&[1, 2, 3, 4], None);
    assert_eq!(
        out.len(),
        4 * 8,
        "forward must return [seq_len=4, hidden=8] flat row-major, got len {}",
        out.len()
    );

    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
}

/// Tokenizer side-car round-trip: emit `vokra.bert.wordpiece.*` from a
/// tiny vocab.txt, then load [`BertWordpieceTokenizer::from_gguf`]
/// under the SBV2-compatible prefix `vokra.bert.wordpiece`. Encodes a
/// two-word input to prove the loaded tokenizer produces a `[CLS] …
/// [SEP]`-wrapped id sequence — the load-bearing contract SBV2's ZH
/// branch depends on.
#[test]
fn converter_wordpiece_tokenizer_loads_via_from_gguf() {
    let (input, output) = temp_pair("tokenizer");
    let blob = bert_full_fixture();
    std::fs::write(&input, &blob).expect("write input safetensors");

    // Tiny WordPiece vocab: 8 pieces including the four specials at
    // their canonical HF ids (`[PAD]=0`, `[UNK]=100`, `[CLS]=101`,
    // `[SEP]=102`). The runtime `BertWordpieceTokenizer` reads specials
    // by NAME lookup from the vocab (via
    // `from_gguf`'s `unk_id`/`cls_id`/`sep_id`/`pad_id` U32 keys the
    // converter stamps), so we need those exact strings at those exact
    // indices in the emitted GGUF. Padding to at least
    // max(BERT_UNK_ID) + 1 = 103 keeps the id-by-position mapping the
    // reader assumes intact.
    let mut vocab_lines: Vec<&str> = vec![""; 103];
    vocab_lines[0] = "[PAD]";
    vocab_lines[1] = "hello";
    vocab_lines[2] = "world";
    vocab_lines[3] = "##ing";
    vocab_lines[4] = "test";
    vocab_lines[100] = "[UNK]";
    vocab_lines[101] = "[CLS]";
    vocab_lines[102] = "[SEP]";
    // Fill unused slots with distinct dummy pieces so the loader does
    // not collapse duplicates.
    for (i, slot) in vocab_lines.iter_mut().enumerate() {
        if slot.is_empty() {
            *slot = Box::leak(format!("[unused{i}]").into_boxed_str());
        }
    }
    let vocab_txt = vocab_lines.join("\n");
    let vocab_bytes = vocab_txt.as_bytes();

    let report = convert_bert_base_file(&input, &output, None, Some(vocab_bytes), false)
        .expect("bert_base converter with --tokenizer must succeed");
    assert!(
        report.written > 0,
        "safetensors tensors still get written even with --tokenizer"
    );

    let g = GgufFile::open(&output).expect("open emitted BertBase GGUF with tokenizer");

    let tok = BertWordpieceTokenizer::from_gguf(&g, "vokra.bert.wordpiece").expect(
        "BertWordpieceTokenizer::from_gguf must load converter output at the prefix \
                 SbV2Model::from_gguf_with_zh_bert passes",
    );

    // Round-trip a two-word input. Exact ids depend on WordPiece
    // segmentation implementation details we do not need to pin here —
    // the load-bearing checks are (a) the tokenizer instantiated at
    // all, and (b) the output starts with `[CLS]` (101) and ends with
    // `[SEP]` (102) per HF WordPiece convention.
    let ids = tok
        .encode("hello world", true)
        .expect("encode must not error on in-vocab tokens");
    assert!(
        ids.first() == Some(&101),
        "encoded sequence must open with [CLS] (id 101), got {ids:?}"
    );
    assert!(
        ids.last() == Some(&102),
        "encoded sequence must close with [SEP] (id 102), got {ids:?}"
    );
    assert!(
        ids.len() >= 3,
        "encoded sequence must include ≥1 content token between [CLS] and [SEP], got {ids:?}"
    );

    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
}
