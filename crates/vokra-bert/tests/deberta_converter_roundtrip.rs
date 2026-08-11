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

/// Column-varying weight pattern for `[d_out, d_in]` — the same
/// pattern `deberta_v2_loader.rs::add_layer_tensors` uses. Non-uniform
/// weights are LOAD-BEARING for the LN-differential test below: with
/// uniform (e.g. all-0.01) `wq_pos`/`wk_pos`, projecting an LN'd
/// rel_embeddings (row-mean 0) gives a constant per-position output,
/// which then cancels through softmax's translation-invariance —
/// masking the LN effect at the encoder-output level even though the
/// per-layer pos_embed bytes were LN'd correctly. Row-varying weights
/// preserve position-dependent structure through the projection so the
/// differential is observable.
fn make_weight(d_out: usize, d_in: usize, tag: f32) -> Vec<f32> {
    (0..d_out)
        .flat_map(|o| (0..d_in).map(move |d| tag + 0.05 * (((o * 7 + d * 13) % 11) as f32 - 5.0)))
        .collect()
}

/// Fixture used by [`v2_rel_embeddings_ln_pre_normalization_changes_forward_output`].
/// Same tensor shape as [`deberta_v2_full_fixture`] BUT with two
/// LN-differential-critical differences:
///
/// - **rel_embeddings values are row-varying** (each row has a distinct
///   mean+variance) so LN produces a distinctive per-row transformation
///   — uniform rel_embeddings would collapse to all zeros under LN,
///   making the LN vs no-LN diff trivial-but-uninteresting.
/// - **Q/K/V/O + FFN projection weights are row-varying** via
///   [`make_weight`] — a uniform projection annihilates any
///   position-dependent structure in `pos_embed` (see [`make_weight`]'s
///   rustdoc for the softmax translation-invariance argument).
///
/// `rel_ln = Some((γ, β))` includes
/// `deberta.encoder.LayerNorm.{weight,bias}` in the emitted safetensors
/// (triggers the 2026-08-10 pre-normalization path in
/// `convert_deberta_v2_file`); `rel_ln = None` omits them (pre-2026-08-10
/// path — raw rel_embeddings duplicated into every layer's `pos_embed`).
///
/// # Reference
///
/// - HF `transformers/src/transformers/models/deberta_v2/modeling_deberta_v2.py`
///   `DebertaV2Encoder.get_rel_embedding` (Apache-2.0) — the ONE-per-forward
///   LN this fixture's `rel_ln = Some(...)` path exercises offline.
fn deberta_v2_ln_differential_fixture(rel_ln: Option<(f32, f32)>) -> Vec<u8> {
    let n_layers: usize = 2;
    let d_model: usize = 8;
    let vocab: usize = 6;
    let n_pos_buckets: usize = 4;

    // Row-varying rel_embeddings (see rustdoc rationale above).
    let rel_vals: Vec<f32> = (0..n_pos_buckets)
        .flat_map(|i| (0..d_model).map(move |j| (i as f32 + 1.0) * 0.1 + (j as f32 + 1.0) * 0.01))
        .collect();

    let mut entries: Vec<(&'static str, &'static str, Vec<u64>, Vec<u8>)> = vec![
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
        // Row-varying rel_embeddings — LN'able signal.
        (
            "deberta.encoder.rel_embeddings.weight",
            "F32",
            vec![n_pos_buckets as u64, d_model as u64],
            f32_bytes(&rel_vals),
        ),
    ];
    // The two tensors that gate the 2026-08-10 pre-normalization path.
    // Absent → converter emits raw rel_embeddings per layer;
    // present → converter emits LN(rel_embeddings, γ, β, ε=1e-7) per layer.
    if let Some((gamma_val, beta_val)) = rel_ln {
        entries.push((
            "deberta.encoder.LayerNorm.weight",
            "F32",
            vec![d_model as u64],
            f32_bytes(&vec![gamma_val; d_model]),
        ));
        entries.push((
            "deberta.encoder.LayerNorm.bias",
            "F32",
            vec![d_model as u64],
            f32_bytes(&vec![beta_val; d_model]),
        ));
    }

    // Row-varying per-projection tags (must differ so projections aren't
    // identical — matches `deberta_v2_loader.rs::add_layer_tensors`).
    let d_ffn = 4 * d_model;
    for i in 0..n_layers {
        let prefix = format!("deberta.encoder.layer.{i}");
        for (proj, tag) in [
            ("query_proj", 0.02_f32),
            ("key_proj", 0.025_f32),
            ("value_proj", 0.03_f32),
        ] {
            entries.push((
                Box::leak(format!("{prefix}.attention.self.{proj}.weight").into_boxed_str()),
                "F32",
                vec![d_model as u64, d_model as u64],
                f32_bytes(&make_weight(d_model, d_model, tag)),
            ));
            entries.push((
                Box::leak(format!("{prefix}.attention.self.{proj}.bias").into_boxed_str()),
                "F32",
                vec![d_model as u64],
                f32_bytes(&vec![0.0f32; d_model]),
            ));
        }
        entries.push((
            Box::leak(format!("{prefix}.attention.output.dense.weight").into_boxed_str()),
            "F32",
            vec![d_model as u64, d_model as u64],
            f32_bytes(&make_weight(d_model, d_model, 0.05)),
        ));
        entries.push((
            Box::leak(format!("{prefix}.attention.output.dense.bias").into_boxed_str()),
            "F32",
            vec![d_model as u64],
            f32_bytes(&vec![0.0f32; d_model]),
        ));
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
        entries.push((
            Box::leak(format!("{prefix}.intermediate.dense.weight").into_boxed_str()),
            "F32",
            vec![d_ffn as u64, d_model as u64],
            f32_bytes(&make_weight(d_ffn, d_model, 0.02)),
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
            f32_bytes(&make_weight(d_model, d_ffn, 0.02)),
        ));
        entries.push((
            Box::leak(format!("{prefix}.output.dense.bias").into_boxed_str()),
            "F32",
            vec![d_model as u64],
            f32_bytes(&vec![0.0f32; d_model]),
        ));
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

/// **End-to-end pin for the 2026-08-10 `rel_embeddings` LN
/// pre-normalization fix** (post-Wave-1 CI residual bert_hidden_ja Δ
/// 11.72 root cause #1).
///
/// Two identical DeBERTa v2 fixtures differ only in the presence of
/// `deberta.encoder.LayerNorm.{weight,bias}` alongside
/// `deberta.encoder.rel_embeddings.weight`. The fix pre-normalizes
/// rel_embeddings with LN(γ, β, ε=1e-7) before per-layer duplication
/// when those tensors are present; leaves rel_embeddings raw when
/// absent. Loading each converter output via
/// [`DebertaV2Encoder::from_gguf`] and running forward on the same
/// input must produce **observably different** outputs — this is the
/// runtime-level proof that (a) LN'd bytes actually flowed to the
/// per-layer `pos_embed.weight` slots and (b)
/// [`DisentangledAttention::forward`](vokra_bert::deberta_v2::DisentangledAttention::forward)
/// consumed them into its C2P + P2C softmax terms.
///
/// The forward path is heavily damped by the tiny fixture (n_layers=2,
/// d_model=8, uniform 0.01 projection weights), so we assert the
/// diff-magnitude order rather than any specific bound — the point is
/// **non-zero divergence**, not a tight parity envelope (that's the
/// real-checkpoint parity CI's job with the ku-nlp GGUF). A pre-fix
/// converter would produce byte-identical GGUFs (LN tensors dropped in
/// both cases), and this test would fail with `sum(|diff|) = 0`.
///
/// # Reference
///
/// - `crates/vokra-convert/src/models/deberta_v2.rs::apply_layer_norm_rows`
///   — the LN mirror the converter applies.
/// - HF `transformers/src/transformers/models/deberta_v2/modeling_deberta_v2.py`
///   `DebertaV2Encoder.get_rel_embedding` (Apache-2.0).
#[test]
fn v2_rel_embeddings_ln_pre_normalization_changes_forward_output() {
    let (in_no_ln, out_no_ln) = temp_pair("v2-rel-no-ln");
    let (in_with_ln, out_with_ln) = temp_pair("v2-rel-with-ln");
    let blob_no_ln = deberta_v2_ln_differential_fixture(None);
    let blob_with_ln = deberta_v2_ln_differential_fixture(Some((2.0_f32, 0.5_f32)));
    std::fs::write(&in_no_ln, &blob_no_ln).expect("write no-LN safetensors");
    std::fs::write(&in_with_ln, &blob_with_ln).expect("write with-LN safetensors");

    convert_deberta_v2_file(&in_no_ln, &out_no_ln, None, None).expect("convert no-LN fixture");
    convert_deberta_v2_file(&in_with_ln, &out_with_ln, None, None)
        .expect("convert with-LN fixture");

    let g_no_ln = GgufFile::open(&out_no_ln).expect("open no-LN GGUF");
    let g_with_ln = GgufFile::open(&out_with_ln).expect("open with-LN GGUF");

    let enc_no_ln =
        DebertaV2Encoder::from_gguf(&g_no_ln).expect("load no-LN encoder from converter output");
    let enc_with_ln = DebertaV2Encoder::from_gguf(&g_with_ln)
        .expect("load with-LN encoder from converter output");

    // Direct proof #1: the per-layer pos_embed bytes DIFFER between the
    // two GGUFs (this is the byte-level assertion; the converter-side
    // test `rel_embeddings_ln_prenormalized_when_ln_tensors_present`
    // pins the exact LN'd values, this one pins that the difference
    // survives round-trip through the runtime loader).
    let pe_no_ln = g_no_ln
        .tensor_f32("bert.encoder.layer.0.attn.pos_embed.weight")
        .expect("no-LN layer 0 pos_embed");
    let pe_with_ln = g_with_ln
        .tensor_f32("bert.encoder.layer.0.attn.pos_embed.weight")
        .expect("with-LN layer 0 pos_embed");
    let pe_diff: f32 = pe_no_ln
        .iter()
        .zip(&pe_with_ln)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        pe_diff > 1e-3,
        "per-layer pos_embed bytes must differ between with-LN and no-LN converter output; \
         sum(|diff|) = {pe_diff} (zero means LN was silently dropped, the pre-fix bug)"
    );

    // Direct proof #2: forward output at the runtime layer visibly
    // shifts — proves LN'd pos_embed reached
    // `DisentangledAttention::forward`'s C2P + P2C softmax terms rather
    // than being loaded and dead-lettered. Damped fixture → tiny diff
    // magnitude, so the test asserts non-zero rather than a fixed bound.
    let ids: &[u32] = &[1, 2, 3, 4];
    let fwd_no_ln = enc_no_ln.forward(ids);
    let fwd_with_ln = enc_with_ln.forward(ids);
    assert_eq!(
        fwd_no_ln.len(),
        fwd_with_ln.len(),
        "shape sanity — same fixture geometry, different LN presence"
    );
    let fwd_diff: f32 = fwd_no_ln
        .iter()
        .zip(&fwd_with_ln)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        fwd_diff > 0.0,
        "encoder forward output must observably differ between with-LN and no-LN paths — \
         a zero diff means either (a) the loader dropped the LN'd bytes or (b) forward \
         does not consume pos_embed (regression in DisentangledAttention.forward)"
    );

    std::fs::remove_file(&in_no_ln).ok();
    std::fs::remove_file(&in_with_ln).ok();
    std::fs::remove_file(&out_no_ln).ok();
    std::fs::remove_file(&out_with_ln).ok();
}
