//! Wave-4 DEBERTA-CONV-NAMES verification: converter → loader round-trip.
//!
//! Audit finding (rank 9, blocker): DeBERTa v2/v3 converters historically
//! emitted verbatim HF names, so `from_gguf` could not load their own output.
//! Task-30 (2026-08-06) landed the rename table in the converters; this test
//! is the missing evidence that the round-trip is closed. Fails if any of the
//! canonical `bert.*` names the loader reads is absent from the GGUF the
//! converter produced from a shape-complete synthetic safetensors fixture.
//!
//! The fixture matches every tensor the loader reads (embeddings, LayerNorm,
//! per-layer attention Q/K/V/O + wq_pos/wk_pos, FFN, ln1/ln2), plus the
//! encoder-level `rel_embeddings.weight` (v2 duplicates it into per-layer
//! `pos_embed`; v3 keeps it as a single shared `bert.encoder.pos_embed.weight`).
//! Uses tiny hparams (n_layers=2, d_model=8, n_heads=2, vocab=6) to keep
//! the safetensors small and the test fast — the mapping table is the
//! contract, not the shape.

use std::path::PathBuf;

use vokra_bert::deberta_v2::DebertaV2Encoder;
use vokra_bert::deberta_v3::DebertaV3Encoder;
use vokra_convert::{convert_deberta_v2_file, convert_deberta_v3_file};
use vokra_core::gguf::GgufFile;

/// f32 slice → little-endian byte payload (matches `SafetensorsFile::parse`).
fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Assembles a safetensors byte buffer from `(name, dtype, shape, payload)`.
/// Mirrors the helper the two converter test modules already share, but
/// lives here as an integration-test-scoped copy (crate-private inside
/// `vokra-convert`).
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

/// Tiny but shape-complete DeBERTa v2/v3 fixture — every tensor the loader
/// reads is emitted with the correct shape. `n_layers=2, d_model=8, n_heads=2,
/// vocab=6, n_pos_buckets=4`. `d_model` is a multiple of `n_heads` per the
/// loader's assertion.
fn deberta_v2_full_fixture() -> Vec<u8> {
    let n_layers: usize = 2;
    let d_model: usize = 8;
    let vocab: usize = 6;
    let n_pos_buckets: usize = 4;

    let mut entries: Vec<(&'static str, &'static str, Vec<u64>, Vec<u8>)> = vec![
        // Embeddings + embed LN.
        (
            "deberta.embeddings.word_embeddings.weight",
            "F32",
            vec![vocab as u64, d_model as u64],
            f32_bytes(&vec![0.01f32; vocab * d_model]),
        ),
        (
            "deberta.embeddings.LayerNorm.weight",
            "F32",
            vec![d_model as u64],
            f32_bytes(&vec![1.0f32; d_model]),
        ),
        (
            "deberta.embeddings.LayerNorm.bias",
            "F32",
            vec![d_model as u64],
            f32_bytes(&vec![0.0f32; d_model]),
        ),
        // Shared rel_embeddings (v2: duplicated per layer as `pos_embed`;
        // v3: emitted once as `bert.encoder.pos_embed.weight`).
        (
            "deberta.encoder.rel_embeddings.weight",
            "F32",
            vec![n_pos_buckets as u64, d_model as u64],
            f32_bytes(&vec![0.001f32; n_pos_buckets * d_model]),
        ),
    ];

    // Per-layer tensors — the loader reads exactly these names.
    for i in 0..n_layers {
        let prefix = format!("deberta.encoder.layer.{i}");
        // Attention Q/K/V/O projections (weight + bias).
        for proj in ["query_proj", "key_proj", "value_proj"] {
            entries.push((
                Box::leak(format!("{prefix}.attention.self.{proj}.weight").into_boxed_str()),
                "F32",
                vec![d_model as u64, d_model as u64],
                f32_bytes(&vec![0.01f32; d_model * d_model]),
            ));
            entries.push((
                Box::leak(format!("{prefix}.attention.self.{proj}.bias").into_boxed_str()),
                "F32",
                vec![d_model as u64],
                f32_bytes(&vec![0.0f32; d_model]),
            ));
        }
        // Attention output projection (weight + bias).
        entries.push((
            Box::leak(format!("{prefix}.attention.output.dense.weight").into_boxed_str()),
            "F32",
            vec![d_model as u64, d_model as u64],
            f32_bytes(&vec![0.01f32; d_model * d_model]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.attention.output.dense.bias").into_boxed_str()),
            "F32",
            vec![d_model as u64],
            f32_bytes(&vec![0.0f32; d_model]),
        ));
        // Attention output LayerNorm → ln1.
        entries.push((
            Box::leak(format!("{prefix}.attention.output.LayerNorm.weight").into_boxed_str()),
            "F32",
            vec![d_model as u64],
            f32_bytes(&vec![1.0f32; d_model]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.attention.output.LayerNorm.bias").into_boxed_str()),
            "F32",
            vec![d_model as u64],
            f32_bytes(&vec![0.0f32; d_model]),
        ));
        // FFN intermediate (w1) and output (w2). FFN inner is `4 * d_model`.
        let d_ffn = 4 * d_model;
        entries.push((
            Box::leak(format!("{prefix}.intermediate.dense.weight").into_boxed_str()),
            "F32",
            vec![d_ffn as u64, d_model as u64],
            f32_bytes(&vec![0.01f32; d_ffn * d_model]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.intermediate.dense.bias").into_boxed_str()),
            "F32",
            vec![d_ffn as u64],
            f32_bytes(&vec![0.0f32; d_ffn]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.output.dense.weight").into_boxed_str()),
            "F32",
            vec![d_model as u64, d_ffn as u64],
            f32_bytes(&vec![0.01f32; d_model * d_ffn]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.output.dense.bias").into_boxed_str()),
            "F32",
            vec![d_model as u64],
            f32_bytes(&vec![0.0f32; d_model]),
        ));
        // Post-FFN LayerNorm → ln2.
        entries.push((
            Box::leak(format!("{prefix}.output.LayerNorm.weight").into_boxed_str()),
            "F32",
            vec![d_model as u64],
            f32_bytes(&vec![1.0f32; d_model]),
        ));
        entries.push((
            Box::leak(format!("{prefix}.output.LayerNorm.bias").into_boxed_str()),
            "F32",
            vec![d_model as u64],
            f32_bytes(&vec![0.0f32; d_model]),
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
        "vokra-deberta-conv-roundtrip-{label}-{}-in.safetensors",
        std::process::id()
    ));
    let mut output = std::env::temp_dir();
    output.push(format!(
        "vokra-deberta-conv-roundtrip-{label}-{}-out.gguf",
        std::process::id()
    ));
    (input, output)
}

/// **RED-in-2026-08-06-if-rename-tables-fail**: convert a shape-complete v2
/// safetensors, then load it via `DebertaV2Encoder::from_gguf`. Must succeed
/// with **zero missing tensors** — the loader's `load_tensor_f32` bails on
/// the first absence, so this is a fully load-bearing check that the
/// converter's rename table + per-layer `pos_embed` duplication produce
/// exactly the canonical names the loader reads.
#[test]
fn v2_converter_output_loads_via_from_gguf() {
    let (input, output) = temp_pair("v2");
    let blob = deberta_v2_full_fixture();
    std::fs::write(&input, &blob).expect("write input safetensors");

    let _report =
        convert_deberta_v2_file(&input, &output, None, None).expect("v2 converter must succeed");
    let g = GgufFile::open(&output).expect("open emitted v2 GGUF");

    // The single load-bearing assertion: from_gguf builds a full encoder.
    let _enc = DebertaV2Encoder::from_gguf(&g)
        .expect("DebertaV2Encoder::from_gguf must load converter output");

    // Sanity: the encoder is actually usable (would panic on shape
    // mismatch) — invoke forward on a 4-token input.
    let out = _enc.forward(&[1, 2, 3, 4]);
    assert!(!out.is_empty(), "forward returned empty hidden states");

    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
}

/// Same for v3: converter emits a single shared `bert.encoder.pos_embed.weight`
/// (v3 does not duplicate). Loader clones it into every layer at load time.
#[test]
fn v3_converter_output_loads_via_from_gguf() {
    let (input, output) = temp_pair("v3");
    let blob = deberta_v2_full_fixture(); // Same fixture shape (upstream tensor names identical)
    std::fs::write(&input, &blob).expect("write input safetensors");

    let _report =
        convert_deberta_v3_file(&input, &output, None, None).expect("v3 converter must succeed");
    let g = GgufFile::open(&output).expect("open emitted v3 GGUF");

    let _enc = DebertaV3Encoder::from_gguf(&g)
        .expect("DebertaV3Encoder::from_gguf must load converter output");

    let out = _enc.forward(&[1, 2, 3, 4]);
    assert!(!out.is_empty(), "v3 forward returned empty hidden states");

    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
}
