//! **DeBERTa v2** (`ku-nlp/deberta-v2-large-japanese-char-wwm`): safetensors
//! → GGUF conversion (SBV2 v2 plan Task 11, 2026-07-26).
//!
//! Input: an upstream HF `transformers` DeBERTa v2 safetensors checkpoint.
//! Output: a GGUF carrying every float tensor plus the `vokra.provenance.*`
//! and `vokra.model.*` metadata chunks, and a best-effort
//! `vokra.bert.deberta_v2.*` hparam chunk group that
//! [`DebertaV2Encoder::from_gguf`](https://docs.rs/vokra-bert) is written
//! to read (`crates/vokra-bert/src/deberta_v2.rs`).
//!
//! # References (permissive only)
//!
//! - He, Liu, Gao, Chen 2021 (arXiv:2006.03654)
//! - HuggingFace `transformers` `deberta_v2` (Apache-2.0) — tensor-naming
//!   convention reference for the eventual mapping table (Task 30)
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
//! - github.com/fishaudio/Bert-VITS2 (AGPL-3.0)
//!
//! # Why this lives in `vokra-convert`, not `vokra-bert`
//!
//! The original SBV2 v2 plan drew this converter as `vokra-bert::converter`
//! (consuming `vokra_convert::safetensors`) and had a *later* task add a
//! `vokra-convert::models::deberta_v2` "thin shim" that calls back into
//! `vokra_bert::converter` for the `ModelKind` dispatch wiring. Chaining
//! those two steps as originally drawn creates a dependency cycle
//! (`vokra-bert -> vokra-convert -> vokra-bert`). This module resolves the
//! cycle by putting the real implementation directly in `vokra-convert`
//! (which already owns every other model's converter) — `vokra-bert` gains
//! no new dependency, and a future `ModelKind::DebertaV2` wiring pass needs
//! only to add a dispatch arm, not a new module.
//!
//! # BF16 pass-through — mirror of `funcodec` / `wespeaker` / `qwen3_tts`
//!
//! F32 / F16 / BF16 tensors pass through **verbatim** under their upstream
//! safetensors names (GGUF types 0 / 1 / 30). No convert-time widening —
//! the runtime widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact).
//!
//! # TODO(owner): tensor name mapping needs a real checkpoint (Task 30)
//!
//! Every tensor is emitted under its **upstream HF name, verbatim** — this
//! converter does not rename e.g. `deberta.encoder.layer.0.attention.self.
//! query_proj.weight`-shaped HF names to the `bert.encoder.layer.0.attn.wq.
//! weight`-shaped names `DebertaV2Encoder::from_gguf` reads. Building that
//! mapping table honestly requires a real checkpoint header dump
//! (`tools/parity/deberta_v2_prepare_checkpoint.py`, Task 30) — HF's actual
//! `DebertaV2Encoder` also computes `rel_embeddings` **once per encoder**
//! rather than once per layer, which may or may not match what
//! `DebertaV2Encoder::from_gguf` currently expects (a question Task 30's
//! real-checkpoint inspection settles, not this converter). Until the
//! mapping lands, a GGUF this converter produces is a provenance-correct,
//! byte-faithful **staging artifact**, not yet loadable by `from_gguf` —
//! mirroring the "Wiring status" posture `funcodec.rs` / `wespeaker.rs`
//! already carry for their own not-yet-consumed outputs.
//!
//! # Hparams — best-effort, not verified against a real checkpoint
//!
//! `n_layers` / `vocab_size` / `d_model` are derived from the checkpoint's
//! own tensor shapes (never invented — see [`count_layers`] /
//! `infer_vocab_and_d_model`). `n_heads` (16) and `n_pos_buckets` /
//! `max_pos_dist` (512 / 512) cannot be recovered from any single tensor's
//! shape (HF stores unsplit `d_model × d_model` projections, not
//! per-head-split matrices) and are written as the same "large-variant"
//! placeholder [`crate::models::deberta_v2`]'s own GGUF-reading
//! counterpart already defaults to when the key is absent — **assumed,
//! not verified, pending Task 30**.
//!
//! # No ONNX (permanent)
//!
//! DeBERTa v2 ships as an HF safetensors checkpoint; this converter never
//! touches ONNX (FR-LD-05).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::{SafeTensorInfo, SafetensorsFile};

/// `vokra.model.arch` for DeBERTa v2 GGUFs.
pub(crate) const ARCH: &str = "deberta_v2";
/// `vokra.model.name` — short slug, distinct from the full HF `org/repo`
/// path (mirrors the `funcodec` / `wespeaker` / `speaker_3d` convention).
pub(crate) const NAME: &str = "deberta-v2-large-japanese-char-wwm";
/// Upstream Hugging Face repository path — provenance breadcrumb.
pub(crate) const UPSTREAM_HF: &str = "ku-nlp/deberta-v2-large-japanese-char-wwm";
/// Upstream declared weight license (SPDX id, lower-case per
/// `docs/license-audit.md` §3.1). `cc-by-sa-4.0` classifies as
/// [`LicenseClass::Copyleft`] (share-alike propagates to the GGUF this
/// converter emits — see `LicenseClass::from_license_str`'s ordering
/// rationale).
pub(crate) const DEFAULT_LICENSE: &str = "cc-by-sa-4.0";
/// `vokra.model.category` for every BERT-family encoder GGUF (v2 and v3
/// alike — [`crate::models::deberta_v3`] reuses this constant).
pub(crate) const CATEGORY: &str = "bert";

/// `vokra.model.category` metadata key — converter-local per the
/// established `funcodec.rs` / `wespeaker.rs` precedent (not yet
/// centralized in `vokra-core::gguf::chunks`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// `vokra.provenance.upstream_hf` metadata key — converter-local for the
/// same reason as [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Metadata-key prefix consumed by
/// `vokra_bert::tokenizer::SbertTokenizer::from_gguf` — shared by both
/// the DeBERTa v2 (JA, `vocab.txt`) and v3 (EN, `spm.model`) converters
/// because the runtime reader itself is prefix-parameterized and every
/// SBV2 v2 call site (see `crates/vokra-models/src/sbv2/mod.rs`) passes
/// `"vokra.bert.tokenizer"` regardless of which BERT sibling is loaded.
pub(crate) const KEY_TOKENIZER_PREFIX: &str = "vokra.bert.tokenizer";

/// Discriminator value stamped under `vokra.bert.tokenizer.kind` for a
/// v2 (`BertJapaneseTokenizer` with `subword_tokenizer_type = "character"`)
/// tokenizer. Distinguishes v2's char-level vocab from v3's
/// `sentencepiece-unigram` bytes so a downstream tool can loud-refuse
/// rather than silently mis-tokenize.
pub(crate) const KIND_BERT_CHARSPLIT: &str = "bert-charsplit";

/// Discriminator value stamped under `vokra.bert.tokenizer.kind` for a
/// v3 SentencePiece Unigram tokenizer (`spm.model`).
pub(crate) const KIND_SENTENCEPIECE_UNIGRAM: &str = "sentencepiece-unigram";

/// v2 special-token ids, hard-coded to `[PAD] [CLS] [SEP] [UNK]` = 0/1/2/3
/// (verified by a direct read of the real fixture's `vocab.txt` header at
/// `/tmp/sbv2-fixtures/deberta-v2-ja/vocab.txt`). The
/// `vokra_bert::tokenizer::SbertTokenizer::from_gguf` reader's own
/// defaults (`unk=1, bos=2, eos=3`) disagree with what the real fixture
/// actually ships, so this converter **must always write these keys
/// explicitly** — silently accepting the loader default would produce a
/// tokenizer that maps every `[UNK]` id to the wrong piece.
pub(crate) const V2_PAD_ID: u32 = 0;
pub(crate) const V2_BOS_ID: u32 = 1;
pub(crate) const V2_EOS_ID: u32 = 2;
pub(crate) const V2_UNK_ID: u32 = 3;

/// Outcome of a DeBERTa conversion — shared shape for both v2
/// ([`convert_deberta_v2_file`]) and
/// [v3](crate::models::deberta_v3::convert_deberta_v3_file), since both
/// count the exact same four things over the exact same tensor-dtype
/// union. Mirrors `FuncodecReport` / `WespeakerReport`'s
/// `read == written + skipped_non_float` invariant.
#[derive(Debug, Default)]
pub struct ConvertReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time today, so
    /// this arm is unreachable in practice; kept for parity with the
    /// sibling converters' counter shape).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens BF16 →
    /// f32 losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Converts a DeBERTa v2 safetensors checkpoint at `input` into a Vokra
/// GGUF at `output`. `license` overrides the upstream `cc-by-sa-4.0` stamp
/// (mirror of the `convert_file --license <spdx>` boundary in `lib.rs`).
///
/// `tokenizer_bytes` optionally stamps the `vokra.bert.tokenizer.*` chunk
/// group `vokra_bert::tokenizer::SbertTokenizer::from_gguf` reads. The
/// bytes are treated as a v2-flavored `vocab.txt` (one piece per line,
/// UTF-8) — the char-based `BertJapaneseTokenizer` upstream ships (with
/// `subword_tokenizer_type = "character"`), whose lack of per-piece score
/// data is honestly recorded by writing `scores = 0.0` for every piece and
/// stamping `vokra.bert.tokenizer.kind = "bert-charsplit"` so a downstream
/// tool can loud-refuse rather than silently mis-tokenize. When
/// `tokenizer_bytes` is [`None`] no `vokra.bert.tokenizer.*` metadata is
/// written (SBV2 v2 loader-side `from_gguf` will then loud-fail — that's
/// FR-EX-08 by design).
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input, an
/// empty `tokenizer_bytes` argument, or when no tensor looks like a
/// token-embedding table (see `infer_vocab_and_d_model` — `vocab_size`
/// has no default in `DebertaV2Encoder::from_gguf`, so this converter
/// refuses to invent one); [`ConvertError::Gguf`] if the GGUF
/// serialization fails.
pub fn convert_deberta_v2_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    tokenizer_bytes: Option<&[u8]>,
) -> Result<ConvertReport, ConvertError> {
    // Front-load the tokenizer refuse — a zero-length side-car cannot
    // become a valid `vokra.bert.tokenizer.*` chunk group (SbertTokenizer's
    // reader would loud-fail on the empty `pieces` array anyway), so drop
    // out here before touching the safetensors input. Mirror of the
    // Voxtral `--tokenizer` gate in `vokra-cli`.
    if let Some(t) = tokenizer_bytes
        && t.is_empty()
    {
        return Err(ConvertError::Parse(
            "deberta-v2 --tokenizer: file is empty — refusing to emit a zero-length \
             vokra.bert.tokenizer.* chunk group (SbertTokenizer::from_gguf would fail to load)"
                .to_owned(),
        ));
    }

    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Hparams — best-effort, checkpoint-shape-derived where possible (see
    // module doc "Hparams" section). NOT independently verified against a
    // real checkpoint; Task 30 fixup.
    let (vocab_size, d_model) = infer_vocab_and_d_model(&st)?;
    let n_layers = count_layers(&st);
    // Wave-4 DEBERTA-CONV-NAMES (2026-08-09): derive n_pos_buckets from the
    // rel_embeddings tensor shape (genuinely shape-derivable), rather than
    // stamping the -large placeholder 512 that no longer matches the
    // duplicated per-layer `pos_embed.weight` bytes.
    let n_pos_buckets = infer_n_pos_buckets(&st);
    write_hparams(
        &mut b,
        "vokra.bert.deberta_v2",
        n_layers,
        d_model,
        vocab_size,
        n_pos_buckets,
    );

    // Tokenizer side-car — Blocker 5 (2026-08-06). Front-loads the
    // `vokra.bert.tokenizer.*` chunk group so a downstream
    // `SbertTokenizer::from_gguf` call succeeds without falling back to
    // the reader's (wrong-for-this-fixture) default ids.
    if let Some(bytes) = tokenizer_bytes {
        write_tokenizer_vocab_txt(&mut b, bytes)?;
    }

    let mut report = ConvertReport::default();
    // Task 30 (2026-08-06): upstream HF names → `bert.*` names that
    // `DebertaV2Encoder::from_gguf` reads. The mapping table below is derived
    // from a real `ku-nlp/deberta-v2-large-japanese-char-wwm` safetensors
    // header dump (400 tensors, verified 2026-08-06). Tensors not matching a
    // known upstream pattern (`cls.*` MLM head, `deberta.embeddings.
    // position_ids` int buffer, `deberta.encoder.conv.*` — v2-specific but
    // not currently consumed by the Rust loader) are skipped with a
    // structured stderr log so a future reader can trace what was dropped
    // and why (FR-EX-08 posture — mirrors `sbv2.rs`'s Task 30 skip logs).
    //
    // The Rust loader expects per-layer `wq_pos`/`wk_pos`/`pos_embed`
    // separate tensors, but upstream HF stores rel_embeddings **once per
    // encoder** (`deberta.encoder.rel_embeddings.weight [512, 1024]`). The
    // Rust implementation's own doc notes that its `wq_pos`/`wk_pos` apply
    // *the same content projections' weights* — the "position projection is
    // separate" is a Rust-side struct-layout convention, not a genuine
    // upstream weight-duplication. We resolve this by **copying**:
    // `query_proj.weight` → `wq.weight` + `wq_pos.weight`, likewise
    // `key_proj.weight` → `wk.weight` + `wk_pos.weight`. And the shared
    // `encoder.rel_embeddings.weight [512, 1024]` gets duplicated into every
    // layer's `pos_embed.weight`. All three copies are semantically
    // equivalent to what the upstream forward pass computes — see the
    // "duplication is semantic, not adding new weight capacity" note.
    let mut skipped_names: Vec<(String, &'static str)> = Vec::new();
    let mut renamed_count = 0usize;
    let mut duplicated_count = 0usize;
    // Grab a copy of the shared rel_embeddings up front for per-layer
    // duplication (there is exactly one `deberta.encoder.rel_embeddings.
    // weight` and it feeds every layer's `pos_embed.weight`).
    let rel_embeddings: Option<(GgmlType, Vec<u64>, Vec<u8>)> = st
        .tensors()
        .iter()
        .find(|t| t.name == "deberta.encoder.rel_embeddings.weight")
        .map(|t| (t.dtype, t.shape.clone(), st.tensor_bytes(t).to_vec()));

    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                // Try to map to `bert.*` name. `None` → skip with log.
                match map_deberta_name(&t.name) {
                    Some(MapAction::Rename(new_name)) => {
                        b.add_tensor(
                            &new_name,
                            t.dtype,
                            t.shape.clone(),
                            st.tensor_bytes(t).to_vec(),
                        )?;
                        report.written += 1;
                        renamed_count += 1;
                        if t.dtype == GgmlType::BF16 {
                            report.bf16_passthrough += 1;
                        }
                    }
                    Some(MapAction::Duplicate(name1, name2)) => {
                        // `query_proj.weight` / `key_proj.weight` /
                        // `query_proj.bias` / `key_proj.bias` — the same
                        // upstream tensor is emitted under both the
                        // content and position projection names to satisfy
                        // the Rust loader's per-projection struct layout
                        // without changing forward-pass semantics.
                        let bytes = st.tensor_bytes(t).to_vec();
                        b.add_tensor(&name1, t.dtype, t.shape.clone(), bytes.clone())?;
                        report.written += 1;
                        if t.dtype == GgmlType::BF16 {
                            report.bf16_passthrough += 1;
                        }
                        b.add_tensor(&name2, t.dtype, t.shape.clone(), bytes)?;
                        report.written += 1;
                        if t.dtype == GgmlType::BF16 {
                            report.bf16_passthrough += 1;
                        }
                        duplicated_count += 1;
                    }
                    None => {
                        // Skip with reason. Emit stderr log for provenance.
                        let reason = classify_skip(&t.name);
                        skipped_names.push((t.name.clone(), reason));
                    }
                }
            }
            _ => report.skipped_non_float += 1,
        }
    }
    // Duplicate the shared rel_embeddings into every layer's pos_embed.weight.
    // n_layers is derived from the checkpoint's own tensor shapes above.
    if let Some((dtype, shape, bytes)) = rel_embeddings {
        for i in 0..n_layers {
            let name = format!("bert.encoder.layer.{i}.attn.pos_embed.weight");
            b.add_tensor(&name, dtype, shape.clone(), bytes.clone())?;
            report.written += 1;
            duplicated_count += 1;
            if dtype == GgmlType::BF16 {
                report.bf16_passthrough += 1;
            }
        }
    }
    // Structured stderr log for skipped tensors (matches sbv2.rs Task 30
    // posture).
    for (name, reason) in &skipped_names {
        eprintln!("convert_deberta_v2: skipping tensor `{name}` ({reason})");
    }
    eprintln!(
        "convert_deberta_v2: {} renamed, {} duplicated (rel_embeddings shared into per-layer pos_embed), {} skipped",
        renamed_count,
        duplicated_count,
        skipped_names.len(),
    );

    let spdx = license.unwrap_or(DEFAULT_LICENSE);
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(&mut b, class, spdx, Some(NAME), Some(UPSTREAM_HF));

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;

    Ok(report)
}

/// What to do with a single upstream tensor name during Task 30 mapping.
///
/// - [`MapAction::Rename`]: emit under a new name (the common case).
/// - [`MapAction::Duplicate`]: emit twice, under two distinct names
///   (used for `query_proj` / `key_proj` weight/bias to satisfy the Rust
///   loader's per-projection `wq`+`wq_pos` / `wk`+`wk_pos` struct layout
///   — semantically equivalent to upstream where the same content
///   projection is applied to both content and position representations).
///
/// A `None` return from [`map_deberta_name`] means "skip this tensor" (see
/// [`classify_skip`] for the reason categories).
pub(crate) enum MapAction {
    Rename(String),
    Duplicate(String, String),
}

/// Maps one upstream HF DeBERTa v2 tensor name to the `bert.*` name(s) the
/// Rust loader (`crates/vokra-bert/src/deberta_v2.rs`) expects. Returns
/// `None` for tensors that intentionally are not consumed (see
/// [`classify_skip`]). Shared by [`crate::models::deberta_v3`] via the
/// identical HF `deberta.*` prefix convention — v3 only differs in the
/// vocabulary size (128100 vs 22012) and the MLM head names (`lm_predictions.
/// *` / `mask_predictions.*` vs `cls.*`), both of which are handled by
/// `classify_skip`.
pub(crate) fn map_deberta_name(upstream: &str) -> Option<MapAction> {
    // Embeddings.
    if upstream == "deberta.embeddings.word_embeddings.weight" {
        return Some(MapAction::Rename("bert.embed.weight".into()));
    }
    if upstream == "deberta.embeddings.LayerNorm.weight" {
        return Some(MapAction::Rename("bert.embed.ln.gamma".into()));
    }
    if upstream == "deberta.embeddings.LayerNorm.bias" {
        return Some(MapAction::Rename("bert.embed.ln.beta".into()));
    }

    // Per-encoder-layer transformer stack.
    if let Some(rest) = upstream.strip_prefix("deberta.encoder.layer.") {
        // rest = "<N>.<sub>..."
        let (idx_str, tail) = rest.split_once('.')?;
        let i: usize = idx_str.parse().ok()?;
        let p = format!("bert.encoder.layer.{i}");
        return match tail {
            // Attention Q/K/V/O projections.
            // Rust expects wq + wq_pos (both content and position variants)
            // for query and key. Upstream stores just one query_proj /
            // key_proj (share_att_key=True in ku-nlp/deberta-v2-large-japanese
            // -char-wwm and every SBV2-v2 checkpoint we ship for) — duplicate
            // both the weight AND the bias to satisfy the Rust struct
            // layout. WP-15: `wq_pos.bias` / `wk_pos.bias` were previously
            // dropped, forcing the Rust forward to fall back to `bq` / `bk`
            // for the position projection; explicitly stamping the same bias
            // tensor under both names makes the loader-side wiring first-class
            // (see `crates/vokra-bert/src/deberta_v2.rs` `AttnWeights`
            // "Position-aware biases").
            "attention.self.query_proj.weight" => Some(MapAction::Duplicate(
                format!("{p}.attn.wq.weight"),
                format!("{p}.attn.wq_pos.weight"),
            )),
            "attention.self.query_proj.bias" => Some(MapAction::Duplicate(
                format!("{p}.attn.wq.bias"),
                format!("{p}.attn.wq_pos.bias"),
            )),
            "attention.self.key_proj.weight" => Some(MapAction::Duplicate(
                format!("{p}.attn.wk.weight"),
                format!("{p}.attn.wk_pos.weight"),
            )),
            "attention.self.key_proj.bias" => Some(MapAction::Duplicate(
                format!("{p}.attn.wk.bias"),
                format!("{p}.attn.wk_pos.bias"),
            )),
            "attention.self.value_proj.weight" => {
                Some(MapAction::Rename(format!("{p}.attn.wv.weight")))
            }
            "attention.self.value_proj.bias" => {
                Some(MapAction::Rename(format!("{p}.attn.wv.bias")))
            }
            "attention.output.dense.weight" => {
                Some(MapAction::Rename(format!("{p}.attn.w_out.weight")))
            }
            "attention.output.dense.bias" => {
                Some(MapAction::Rename(format!("{p}.attn.w_out.bias")))
            }
            // Attention output LayerNorm (post-attention residual norm) →
            // ln1 by the loader's convention (pre-FFN norm).
            "attention.output.LayerNorm.weight" => {
                Some(MapAction::Rename(format!("{p}.ln1.gamma")))
            }
            "attention.output.LayerNorm.bias" => Some(MapAction::Rename(format!("{p}.ln1.beta"))),
            // FFN.
            "intermediate.dense.weight" => Some(MapAction::Rename(format!("{p}.ffn.w1.weight"))),
            "intermediate.dense.bias" => Some(MapAction::Rename(format!("{p}.ffn.w1.bias"))),
            "output.dense.weight" => Some(MapAction::Rename(format!("{p}.ffn.w2.weight"))),
            "output.dense.bias" => Some(MapAction::Rename(format!("{p}.ffn.w2.bias"))),
            // Post-FFN LayerNorm → ln2.
            "output.LayerNorm.weight" => Some(MapAction::Rename(format!("{p}.ln2.gamma"))),
            "output.LayerNorm.bias" => Some(MapAction::Rename(format!("{p}.ln2.beta"))),
            _ => None,
        };
    }

    // Shared rel_embeddings gets duplicated into per-layer pos_embed by the
    // caller after the main loop (see the `if let Some((dtype, shape,
    // bytes)) = rel_embeddings` block). Return None here so the main loop
    // does not accidentally emit it under some other name.
    if upstream == "deberta.encoder.rel_embeddings.weight" {
        return None;
    }

    // Not consumed by the Rust loader — skip with a categorized reason.
    None
}

/// Categorizes why an upstream tensor was skipped, for structured stderr
/// logs (FR-EX-08 posture — never silently drop tensors without stating
/// the reason).
pub(crate) fn classify_skip(name: &str) -> &'static str {
    // MLM head (v2 uses `cls.predictions.*`, v3 uses `lm_predictions.*` +
    // `mask_predictions.*` for the RTD auxiliary head). SBV2 v2 never
    // consumes the MLM output — only encoder hidden states — so drop.
    if name.starts_with("cls.")
        || name.starts_with("lm_predictions.")
        || name.starts_with("mask_predictions.")
    {
        return "MLM head — SBV2 consumes encoder hidden states only";
    }
    // Position-id buffer (int, not float, and derivable at runtime as
    // arange(0, seq_len)).
    if name == "deberta.embeddings.position_ids" {
        return "position_ids buffer — derivable at runtime as arange(0, seq_len)";
    }
    // Absolute position embeddings (v3 has them, v2 does not; the Rust
    // DeBERTa loader is disentangled-attention-only and does not consume
    // absolute position embeddings).
    if name == "deberta.embeddings.position_embeddings.weight" {
        return "absolute position embeddings — disentangled attention uses rel_embeddings only";
    }
    // v2-specific conv layer inside the encoder. The Rust DeBERTa v2
    // loader does not consume this today (its struct has no conv slot);
    // future support would require both a struct-layout change and a
    // rename entry here.
    if name.starts_with("deberta.encoder.conv.") {
        return "v2-specific encoder conv — not consumed by Rust loader today";
    }
    // Top-level encoder LayerNorm (applied after the last transformer
    // block, before returning hidden states). The Rust loader applies its
    // own final normalization inside `EncoderLayer.ln2` on the last layer;
    // dropping this upstream `encoder.LayerNorm.*` matches the loader's
    // current struct shape. Explicitly documented in `classify_skip` so a
    // future reader can trace what happened.
    if name.starts_with("deberta.encoder.LayerNorm.") {
        return "top-level encoder LayerNorm — loader applies its own final norm inside ln2";
    }
    "unmapped tensor — no rename rule matched"
}

/// Parses a v2 `vocab.txt` (one piece per line, UTF-8) and stamps the
/// `vokra.bert.tokenizer.*` chunk group [`SbertTokenizer::from_gguf`]
/// reads. Trailing blank lines are stripped (upstream
/// `BertJapaneseTokenizer` treats them as `[UNK]` placeholders, but the
/// reader assigns ids by position and a blank piece is ambiguous — see
/// this converter's Blocker-5 rationale). Scores are written as `0.0`
/// for every piece because the char-based upstream carries no per-piece
/// log-probabilities; the resulting viterbi degenerates to
/// "longest-match" (for a char vocab with all pieces length-1 that's a
/// no-op — matches the observed upstream behavior on pure-JA inputs).
///
/// Also stamps `vokra.bert.tokenizer.kind = "bert-charsplit"` and the
/// hard-coded `unk_id=3 / bos_id=1 / eos_id=2` triple verified against
/// the real fixture (`/tmp/sbv2-fixtures/deberta-v2-ja/vocab.txt` +
/// `special_tokens_map.json`).
///
/// # Errors
///
/// [`ConvertError::Parse`] when the bytes are not valid UTF-8.
pub(crate) fn write_tokenizer_vocab_txt(
    b: &mut GgufBuilder,
    bytes: &[u8],
) -> Result<(), ConvertError> {
    let text = std::str::from_utf8(bytes).map_err(|_| {
        ConvertError::Parse("deberta-v2 --tokenizer: vocab.txt is not valid UTF-8".to_owned())
    })?;
    // `.lines()` handles both `\n` and `\r\n`, and does NOT emit a
    // trailing empty element for a file that ends in newline (BufRead
    // convention — trimming trailing blanks is a separate concern).
    let mut pieces: Vec<String> = text.lines().map(str::to_owned).collect();
    // Strip trailing blank lines — the real fixture ends with 8 empty
    // lines that upstream `BertJapaneseTokenizer` interprets as `[UNK]`
    // placeholders; since the loader assigns ids by index we cannot
    // safely reproduce that (a blank piece string would never match
    // viterbi's byte-prefix probe).
    while pieces.last().is_some_and(String::is_empty) {
        pieces.pop();
    }
    if pieces.is_empty() {
        return Err(ConvertError::Parse(
            "deberta-v2 --tokenizer: vocab.txt has no non-empty lines".to_owned(),
        ));
    }
    let scores: Vec<f32> = vec![0.0; pieces.len()];
    let vocab_size = pieces.len() as u32;

    b.add_string(&format!("{KEY_TOKENIZER_PREFIX}.kind"), KIND_BERT_CHARSPLIT);
    add_string_array(b, &format!("{KEY_TOKENIZER_PREFIX}.pieces"), &pieces);
    add_f32_array(b, &format!("{KEY_TOKENIZER_PREFIX}.scores"), &scores);
    b.add_u32(&format!("{KEY_TOKENIZER_PREFIX}.unk_id"), V2_UNK_ID);
    b.add_u32(&format!("{KEY_TOKENIZER_PREFIX}.bos_id"), V2_BOS_ID);
    b.add_u32(&format!("{KEY_TOKENIZER_PREFIX}.eos_id"), V2_EOS_ID);
    b.add_u32(&format!("{KEY_TOKENIZER_PREFIX}.pad_id"), V2_PAD_ID);
    b.add_u32(&format!("{KEY_TOKENIZER_PREFIX}.vocab_size"), vocab_size);
    Ok(())
}

/// Emit a `String` array under `key` — follows the kokoro / piper-plus
/// pattern (`add_metadata(GgufMetadataValue::Array(...))`, no typed
/// shortcut on `GgufBuilder`). Shared by both DeBERTa v2 and v3
/// tokenizer emitters.
pub(crate) fn add_string_array(b: &mut GgufBuilder, key: &str, values: &[String]) {
    b.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: values
                .iter()
                .map(|s| GgufMetadataValue::String(s.clone()))
                .collect(),
        }),
    );
}

/// Emit an `F32` array under `key` — mirror of `fsmn_vad.rs`'s
/// `f32_array_chunk` helper (kept local to avoid unrelated crate churn).
pub(crate) fn add_f32_array(b: &mut GgufBuilder, key: &str, values: &[f32]) {
    b.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::F32,
            values: values.iter().map(|&v| GgufMetadataValue::F32(v)).collect(),
        }),
    );
}

/// Wave-4 DEBERTA-CONV-NAMES (2026-08-09): head-dim convention for the HF
/// DeBERTa family. Both `ku-nlp/deberta-v2-large-japanese-char-wwm` and
/// `microsoft/deberta-v3-large` use `head_dim = 64` (16 heads × 64 =
/// 1024 = d_model); the "base" variants use 12 heads × 64 = 768. Every
/// public HF DeBERTa checkpoint we ever plan to load holds this constant.
/// `d_model / HEAD_DIM_HF_DEFAULT` is therefore a shape-derived (not
/// invented) value that matches every real target exactly and keeps the
/// synthetic round-trip test [d_model=8 → 1 head] loader-loadable.
pub(crate) const HEAD_DIM_HF_DEFAULT: u32 = 64;

/// Derives `n_heads` from `d_model` via [`HEAD_DIM_HF_DEFAULT`] with a
/// floor of 1 (single-head fallback for tiny synthetic fixtures — real
/// HF checkpoints always have `d_model >= 768`, so the floor never fires
/// on real inputs). Guarantees `d_model % n_heads == 0` for every
/// `d_model` that is a multiple of `HEAD_DIM_HF_DEFAULT` OR smaller than
/// it, which covers every checkpoint the converter can encounter.
pub(crate) fn derive_n_heads(d_model: u64) -> u32 {
    let raw = (d_model / u64::from(HEAD_DIM_HF_DEFAULT)) as u32;
    raw.max(1)
}

/// Writes the shared six-key `vokra.bert.<arch>.*` hparam chunk group
/// (`n_layers` / `d_model` / `n_heads` / `vocab_size` / `n_pos_buckets` /
/// `max_pos_dist`) under `prefix`.
///
/// **`n_heads`** is derived from `d_model` via
/// [`derive_n_heads`] (shape-driven — HF DeBERTa convention of
/// `head_dim=64`). The previous hard-coded `16` was a placeholder that
/// only happened to be correct for `-large` variants and tripped the
/// loader's `d_model % n_heads == 0` divisibility check on any smaller
/// checkpoint (Wave-4 DEBERTA-CONV-NAMES round-trip finding).
///
/// **`n_pos_buckets`** is the caller-supplied value derived from the
/// upstream `rel_embeddings.weight` first-axis extent
/// (`[n_pos_buckets, d_model]`) — genuinely shape-derivable and required
/// for the loader's `pos_embed` slice-length math to match the tensor
/// the converter emitted. `None` falls back to the `-large` default 512
/// (only reachable when the input safetensors contains no rel_embeddings
/// at all, an unrealistic edge case that keeps existing per-tensor
/// tests running).
///
/// **`max_pos_dist`** is the `-large` convention default (512). No
/// public HF DeBERTa checkpoint uses a different value and no loader
/// assertion depends on it — can be tightened by owner in a follow-up
/// (Task 30 follow-up) once a real fine-tune with a non-default value
/// is inspected.
pub(crate) fn write_hparams(
    b: &mut GgufBuilder,
    prefix: &str,
    n_layers: u32,
    d_model: u64,
    vocab_size: u64,
    n_pos_buckets: Option<u32>,
) {
    b.add_u32(&format!("{prefix}.n_layers"), n_layers);
    b.add_u32(&format!("{prefix}.d_model"), d_model as u32);
    b.add_u32(&format!("{prefix}.n_heads"), derive_n_heads(d_model));
    b.add_u32(&format!("{prefix}.vocab_size"), vocab_size as u32);
    b.add_u32(
        &format!("{prefix}.n_pos_buckets"),
        n_pos_buckets.unwrap_or(512),
    );
    b.add_u32(&format!("{prefix}.max_pos_dist"), 512); // -large convention — Task 30 follow-up
}

/// Recovers `n_pos_buckets` from the upstream `deberta.encoder.
/// rel_embeddings.weight` tensor's first-axis extent, which is exactly
/// `[n_pos_buckets, d_model]` in every real HF DeBERTa checkpoint (both
/// v2 char-JA and v3-large). Returns `None` when the tensor is absent
/// (unrealistic — HF DeBERTa always ships it) so the caller can fall
/// back to a per-arch default.
pub(crate) fn infer_n_pos_buckets(st: &SafetensorsFile) -> Option<u32> {
    st.tensors()
        .iter()
        .find(|t| t.name == "deberta.encoder.rel_embeddings.weight")
        .and_then(|t| t.shape.first().copied())
        .and_then(|v| u32::try_from(v).ok())
}

/// Counts the highest `N` in any tensor name containing the literal
/// substring `layer.N` (prefix-agnostic — scans wherever the token
/// occurs, rather than anchoring to one asserted prefix).
///
/// Distinct from the repo's established `count_layers(st, prefix)` helper
/// (`distil_whisper.rs` / `kotoba_whisper.rs` / `kokoro.rs` /
/// `whisper.rs`): those anchor to a prefix already *verified* against a
/// real checkpoint (e.g. `"model.encoder.layers."`). DeBERTa v2's real
/// tensor-name prefix is not yet confirmed (Task 30), so this scans for
/// the bare `layer.` token — a digit immediately following `layer.` is a
/// strong, low-false-positive signal for "this is a per-block tensor"
/// regardless of what precedes it. Falls back to `24` (the "large"
/// convention both default checkpoints target) when no such tensor name
/// exists at all — e.g. an empty or non-BERT-shaped input.
pub(crate) fn count_layers(st: &SafetensorsFile) -> u32 {
    let mut max_idx: Option<u32> = None;
    for t in st.tensors() {
        let mut rest = t.name.as_str();
        while let Some(pos) = rest.find("layer.") {
            let after = &rest[pos + "layer.".len()..];
            if let Some(end) = after.find('.') {
                if let Ok(idx) = after[..end].parse::<u32>() {
                    max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
                }
            }
            rest = after;
        }
    }
    max_idx.map_or(24, |m| m + 1)
}

/// Finds the tensor most likely to be the token-embedding table — name
/// contains `embed` (case-insensitive), rank 2, largest first dimension
/// among such tensors — and reads `(vocab_size, d_model)` off its shape.
///
/// `[vocab_size, d_model]` is universal across HF BERT-family checkpoints
/// regardless of the exact tensor name, and picking the *largest*
/// first-dimension among `embed`-named 2-D tensors reliably distinguishes
/// the vocab table from e.g. a `position_embeddings` `[512, d_model]` or
/// `token_type_embeddings` `[2, d_model]` table also matching the
/// substring — vocab sizes (thousands+) dwarf position/type-vocab sizes
/// (hundreds) for every real tokenizer. Errors, rather than guessing, when
/// no candidate exists: `vocab_size` has no default in
/// `DebertaV2Encoder::from_gguf` (FR-EX-08 — loud, not silent).
pub(crate) fn infer_vocab_and_d_model(st: &SafetensorsFile) -> Result<(u64, u64), ConvertError> {
    st.tensors()
        .iter()
        .filter(|t: &&SafeTensorInfo| {
            t.name.to_ascii_lowercase().contains("embed") && t.shape.len() == 2
        })
        .max_by_key(|t| t.shape[0])
        .map(|t| (t.shape[0], t.shape[1]))
        .ok_or_else(|| {
            ConvertError::Parse(
                "no rank-2 tensor with 'embed' in its name found (expected a token \
                 embedding table to derive vocab_size / d_model from)"
                    .to_owned(),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Builds a multi-tensor safetensors byte buffer from `(name, dtype,
    /// shape, payload)` entries — a generalization of the fixture helpers
    /// `funcodec.rs` / `wespeaker.rs` use, needed here because hparam
    /// inference exercises *several* named tensors per fixture rather
    /// than one or two.
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

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Distinctive BF16 payload — top 16 bits of a handful of exact f32
    /// values, mirroring `funcodec.rs`'s `distinctive_bf16_payload`.
    fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn temp_pair(label: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let mut input = std::env::temp_dir();
        input.push(format!(
            "vokra-deberta-v2-{label}-{}-in.safetensors",
            std::process::id()
        ));
        let mut output = std::env::temp_dir();
        output.push(format!(
            "vokra-deberta-v2-{label}-{}-out.gguf",
            std::process::id()
        ));
        (input, output)
    }

    /// A minimal but shape-meaningful fixture: a `word_embeddings` table
    /// (vocab=6, d_model=4 — deliberately larger `shape[0]` than the
    /// `position_embeddings` table below, so `infer_vocab_and_d_model`
    /// must pick the right one, not just "any embed-named tensor"), a
    /// `position_embeddings` table that *also* matches the `embed`
    /// substring (regression guard against a naive "first embed match"
    /// bug), and two per-layer tensors (`layer.0`, `layer.2`) so
    /// [`count_layers`] must resolve `n_layers = 3`, not `2`.
    fn base_fixture() -> Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> {
        vec![
            (
                "deberta.embeddings.word_embeddings.weight",
                "F32",
                &[6, 4],
                f32_bytes(&[0.01; 24]),
            ),
            (
                "deberta.embeddings.position_embeddings.weight",
                "F32",
                &[3, 4],
                f32_bytes(&[0.02; 12]),
            ),
            (
                "deberta.encoder.layer.0.attention.self.query_proj.weight",
                "F32",
                &[4, 4],
                f32_bytes(&[0.03; 16]),
            ),
            (
                "deberta.encoder.layer.2.attention.self.query_proj.weight",
                "BF16",
                &[4, 4],
                bf16_bytes(&[0.04; 16]),
            ),
        ]
    }

    /// RED-phase pin: hparams are derived from the fixture's real tensor
    /// shapes (vocab=6, d_model=4, n_layers=3 — highest layer index 2,
    /// +1), not invented; BF16 passes through as GGUF type 30 verbatim;
    /// provenance defaults to `cc-by-sa-4.0` / `Copyleft`.
    #[test]
    fn hparams_and_bf16_derived_from_real_shapes() {
        let blob = safetensors_multi(&base_fixture());
        let (input, output) = temp_pair("hparams");
        std::fs::write(&input, &blob).expect("write input safetensors");

        // After Task 30 rename table (2026-08-06): 4 upstream tensors map to
        // 5 written entries:
        //   - `word_embeddings` → renamed to `bert.embed.weight` (1 written)
        //   - `position_embeddings` → skipped (absolute pos embed, disent
        //     attention uses rel_embeddings only — see classify_skip)
        //   - `layer.0.attention.self.query_proj.weight` → duplicated
        //     (`wq.weight` + `wq_pos.weight`, both F32 = 2 written)
        //   - `layer.2.attention.self.query_proj.weight` → duplicated (BF16,
        //     both copies BF16 = 2 written, 2 bf16_passthrough)
        let report = convert_deberta_v2_file(&input, &output, None, None).expect("convert");
        assert_eq!(report.read, 4);
        assert_eq!(
            report.written, 5,
            "1 renamed + 0 skipped + 2*2 duplicated = 5 (position_embeddings skipped)"
        );
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 2,
            "the BF16 query_proj is duplicated twice, both copies BF16"
        );

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        assert_eq!(
            file.get("vokra.bert.deberta_v2.vocab_size")
                .and_then(|v| v.as_u64()),
            Some(6),
            "vocab_size must come from the larger embed-named tensor, not position_embeddings"
        );
        assert_eq!(
            file.get("vokra.bert.deberta_v2.d_model")
                .and_then(|v| v.as_u64()),
            Some(4)
        );
        assert_eq!(
            file.get("vokra.bert.deberta_v2.n_layers")
                .and_then(|v| v.as_u64()),
            Some(3),
            "highest layer index observed is 2 (layer.0 and layer.2) -> n_layers = 3"
        );
        // Wave-4 DEBERTA-CONV-NAMES (2026-08-09): `n_heads` is derived from
        // `d_model` via `HEAD_DIM_HF_DEFAULT = 64` with a floor of 1
        // (single-head fallback for tiny synthetic fixtures — the base
        // fixture's `d_model = 4` gives `4 / 64 = 0 → 1`). Real HF
        // checkpoints (`d_model = 1024`) yield the correct `n_heads = 16`.
        assert_eq!(
            file.get("vokra.bert.deberta_v2.n_heads")
                .and_then(|v| v.as_u64()),
            Some(1),
            "n_heads is derive_n_heads(d_model=4) = max(1, 0) = 1, not the old placeholder 16"
        );
        assert_eq!(
            file.get("vokra.bert.deberta_v2.n_pos_buckets")
                .and_then(|v| v.as_u64()),
            Some(512)
        );
        assert_eq!(
            file.get("vokra.bert.deberta_v2.max_pos_dist")
                .and_then(|v| v.as_u64()),
            Some(512)
        );

        // Task 30 (2026-08-06): the upstream `query_proj.weight` name is
        // renamed to `bert.encoder.layer.<i>.attn.wq.weight` (and duplicated
        // to `wq_pos.weight`). Check both duplicates preserve BF16 dtype
        // verbatim — no convert-time widening on either copy.
        for name in [
            "bert.encoder.layer.2.attn.wq.weight",
            "bert.encoder.layer.2.attn.wq_pos.weight",
        ] {
            let bf16_info = file
                .tensor_info(name)
                .unwrap_or_else(|| panic!("BF16 tensor `{name}` present after rename"));
            assert_eq!(
                bf16_info.dtype,
                GgmlType::BF16,
                "no convert-time widening — {name} GGUF dtype must remain BF16 (type 30)"
            );
        }

        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Copyleft.as_str()),
            "cc-by-sa-4.0 must classify as Copyleft (share-alike propagates)"
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// `--license <spdx>` override replaces the default `cc-by-sa-4.0`
    /// stamp entirely (mirror of the `convert_file --license` boundary
    /// every sibling converter honors).
    #[test]
    fn license_override_replaces_default() {
        let blob = safetensors_multi(&base_fixture());
        let (input, output) = temp_pair("license-override");
        std::fs::write(&input, &blob).expect("write input safetensors");

        convert_deberta_v2_file(&input, &output, Some("apache-2.0"), None).expect("convert");

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// WP-15: `attention.self.query_proj.bias` / `key_proj.bias` are
    /// duplicated into both content (`wq.bias`/`wk.bias`) and
    /// position-projection (`wq_pos.bias`/`wk_pos.bias`) names —
    /// mirroring the existing `query_proj.weight` / `key_proj.weight`
    /// weight duplication for upstream `share_att_key=True` semantics.
    /// Without this the runtime forward silently falls back to the
    /// content bias for the position projection; this test locks in
    /// that the loader-visible `wq_pos.bias` / `wk_pos.bias` names are
    /// emitted (see `crates/vokra-bert/src/deberta_v2.rs`
    /// `AttnWeights::bq_pos`).
    #[test]
    fn query_key_bias_is_duplicated_into_pos_projection_bias() {
        // Bare minimum: one embed table + one layer with query/key/value
        // proj weights *and* biases. The bias-duplication is what this
        // test pins; other tensors are along for the ride.
        let entries: Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> = vec![
            (
                "deberta.embeddings.word_embeddings.weight",
                "F32",
                &[6, 4],
                f32_bytes(&[0.01; 24]),
            ),
            (
                "deberta.encoder.layer.0.attention.self.query_proj.weight",
                "F32",
                &[4, 4],
                f32_bytes(&[0.03; 16]),
            ),
            (
                "deberta.encoder.layer.0.attention.self.query_proj.bias",
                "F32",
                &[4],
                f32_bytes(&[0.11; 4]),
            ),
            (
                "deberta.encoder.layer.0.attention.self.key_proj.weight",
                "F32",
                &[4, 4],
                f32_bytes(&[0.04; 16]),
            ),
            (
                "deberta.encoder.layer.0.attention.self.key_proj.bias",
                "F32",
                &[4],
                f32_bytes(&[0.13; 4]),
            ),
        ];
        let blob = safetensors_multi(&entries);
        let (input, output) = temp_pair("qk-bias-dupe");
        std::fs::write(&input, &blob).expect("write input safetensors");

        convert_deberta_v2_file(&input, &output, None, None).expect("convert");
        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        // Both content and position bias names must be present, and
        // must carry the SAME bytes (share_att_key=True — the same
        // upstream tensor emitted twice under two names).
        let wq_bias = file
            .tensor_f32("bert.encoder.layer.0.attn.wq.bias")
            .expect("wq.bias present");
        let wq_pos_bias = file
            .tensor_f32("bert.encoder.layer.0.attn.wq_pos.bias")
            .expect("wq_pos.bias present (WP-15)");
        assert_eq!(
            wq_bias, wq_pos_bias,
            "wq.bias and wq_pos.bias are the same upstream tensor emitted twice"
        );
        assert_eq!(wq_bias, vec![0.11_f32; 4]);

        let wk_bias = file
            .tensor_f32("bert.encoder.layer.0.attn.wk.bias")
            .expect("wk.bias present");
        let wk_pos_bias = file
            .tensor_f32("bert.encoder.layer.0.attn.wk_pos.bias")
            .expect("wk_pos.bias present (WP-15)");
        assert_eq!(
            wk_bias, wk_pos_bias,
            "wk.bias and wk_pos.bias are the same upstream tensor emitted twice"
        );
        assert_eq!(wk_bias, vec![0.13_f32; 4]);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// A checkpoint with no `embed`-named tensor at all is a loud error
    /// (`ConvertError::Parse`), never a silently-invented `vocab_size`
    /// (FR-EX-08).
    #[test]
    fn missing_embed_tensor_is_a_loud_error() {
        let entries = vec![(
            "deberta.encoder.layer.0.attention.self.query_proj.weight",
            "F32",
            &[4u64, 4][..],
            f32_bytes(&[0.03; 16]),
        )];
        let blob = safetensors_multi(&entries);
        let (input, output) = temp_pair("no-embed");
        std::fs::write(&input, &blob).expect("write input safetensors");

        let err =
            convert_deberta_v2_file(&input, &output, None, None).expect_err("must fail loudly");
        assert!(matches!(err, ConvertError::Parse(_)));
        assert!(!output.exists(), "no partial GGUF must be left behind");

        std::fs::remove_file(&input).ok();
    }

    /// Blocker 5 (2026-08-06) — tokenizer_bytes = None emits **no**
    /// `vokra.bert.tokenizer.*` metadata (the runtime side loud-fails on
    /// `SbertTokenizer::from_gguf` — that's FR-EX-08 by design; silently
    /// stamping placeholder data would produce a GGUF that appears
    /// loadable but tokenizes wrong).
    #[test]
    fn no_tokenizer_bytes_emits_no_tokenizer_chunk() {
        let blob = safetensors_multi(&base_fixture());
        let (input, output) = temp_pair("no-tokenizer");
        std::fs::write(&input, &blob).expect("write input safetensors");

        convert_deberta_v2_file(&input, &output, None, None).expect("convert");

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");
        for suffix in [
            "kind",
            "pieces",
            "scores",
            "unk_id",
            "bos_id",
            "eos_id",
            "pad_id",
            "vocab_size",
        ] {
            let key = format!("{KEY_TOKENIZER_PREFIX}.{suffix}");
            assert!(
                file.get(&key).is_none(),
                "no --tokenizer supplied: `{key}` must NOT be present"
            );
        }

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Blocker 5 (2026-08-06) — tokenizer_bytes = Some(vocab_txt) stamps
    /// the full `vokra.bert.tokenizer.*` chunk group under the prefix
    /// [`SbertTokenizer::from_gguf`] reads. Uses a synthetic 6-line
    /// `vocab.txt` mirroring the real `deberta-v2-ja` fixture's header
    /// (`[PAD] [CLS] [SEP] [UNK] [MASK] ▁`); the special-token ids
    /// (`unk=3, bos=1, eos=2, pad=0`) are the hard-coded values verified
    /// against `/tmp/sbv2-fixtures/deberta-v2-ja/vocab.txt` at scout
    /// time.
    #[test]
    fn converter_stamps_vocab_txt_tokenizer_metadata() {
        let vocab_txt = b"[PAD]\n[CLS]\n[SEP]\n[UNK]\n[MASK]\n\xe2\x96\x81\n";
        let blob = safetensors_multi(&base_fixture());
        let (input, output) = temp_pair("tok-vocab-txt");
        std::fs::write(&input, &blob).expect("write input safetensors");

        convert_deberta_v2_file(&input, &output, None, Some(vocab_txt)).expect("convert");

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        assert_eq!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.kind"))
                .and_then(|v| v.as_str()),
            Some(KIND_BERT_CHARSPLIT),
            "v2 discriminator must be `bert-charsplit`, not `sentencepiece-unigram`"
        );
        let pieces = file
            .get(&format!("{KEY_TOKENIZER_PREFIX}.pieces"))
            .and_then(|v| v.as_array())
            .expect("pieces array present");
        assert_eq!(pieces.values.len(), 6, "6-line vocab.txt → 6 pieces");
        let piece_at = |i: usize| pieces.values[i].as_str().unwrap();
        assert_eq!(piece_at(0), "[PAD]");
        assert_eq!(piece_at(1), "[CLS]");
        assert_eq!(piece_at(2), "[SEP]");
        assert_eq!(piece_at(3), "[UNK]");
        assert_eq!(piece_at(4), "[MASK]");
        assert_eq!(
            piece_at(5),
            "▁",
            "U+2581 word-start piece preserved verbatim"
        );

        let scores = file
            .get(&format!("{KEY_TOKENIZER_PREFIX}.scores"))
            .and_then(|v| v.as_array())
            .expect("scores array present");
        assert_eq!(scores.values.len(), 6, "scores length must match pieces");

        assert_eq!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.unk_id"))
                .and_then(|v| v.as_u64()),
            Some(u64::from(V2_UNK_ID)),
            "v2 unk_id must be 3, NOT the SbertTokenizer::from_gguf default of 1"
        );
        assert_eq!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.bos_id"))
                .and_then(|v| v.as_u64()),
            Some(u64::from(V2_BOS_ID)),
            "v2 bos_id must be 1 ([CLS]), NOT the reader default of 2"
        );
        assert_eq!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.eos_id"))
                .and_then(|v| v.as_u64()),
            Some(u64::from(V2_EOS_ID)),
            "v2 eos_id must be 2 ([SEP]), NOT the reader default of 3"
        );
        assert_eq!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.pad_id"))
                .and_then(|v| v.as_u64()),
            Some(u64::from(V2_PAD_ID))
        );
        assert_eq!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.vocab_size"))
                .and_then(|v| v.as_u64()),
            Some(6),
            "vocab_size stamp mirrors pieces.len()"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Blocker 5 (2026-08-06) — trailing blank lines in `vocab.txt`
    /// (the real fixture ends with 8 empties that upstream
    /// `BertJapaneseTokenizer` uses as `[UNK]` placeholders) are stripped
    /// so `SbertTokenizer::from_gguf`'s viterbi never has to probe an
    /// empty piece string.
    #[test]
    fn trailing_blank_lines_are_stripped() {
        let vocab_txt = b"[PAD]\n[CLS]\n[SEP]\n[UNK]\n\n\n\n";
        let blob = safetensors_multi(&base_fixture());
        let (input, output) = temp_pair("tok-trailing-blanks");
        std::fs::write(&input, &blob).expect("write input safetensors");
        convert_deberta_v2_file(&input, &output, None, Some(vocab_txt)).expect("convert");
        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");
        let pieces = file
            .get(&format!("{KEY_TOKENIZER_PREFIX}.pieces"))
            .and_then(|v| v.as_array())
            .expect("pieces array present");
        assert_eq!(pieces.values.len(), 4, "trailing blanks stripped");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Blocker 5 (2026-08-06) — an empty `--tokenizer` argument is a
    /// loud usage error (FR-EX-08): silently emitting a zero-length
    /// `pieces` array would produce a GGUF that `SbertTokenizer::from_gguf`
    /// itself refuses to load (the reader defaults `unk=1` to id 1, which
    /// does not exist in a 0-piece vocab).
    #[test]
    fn empty_tokenizer_bytes_is_a_loud_error() {
        let blob = safetensors_multi(&base_fixture());
        let (input, output) = temp_pair("tok-empty");
        std::fs::write(&input, &blob).expect("write input safetensors");
        let err = convert_deberta_v2_file(&input, &output, None, Some(&[]))
            .expect_err("empty tokenizer must be refused");
        assert!(matches!(err, ConvertError::Parse(_)));
        assert!(!output.exists(), "no partial GGUF left behind");
        std::fs::remove_file(&input).ok();
    }

    /// Wave-4 DEBERTA-CONV-NAMES (2026-08-09) — `derive_n_heads` matches
    /// the HF DeBERTa convention (`head_dim = 64`) for every real target
    /// checkpoint. Every value listed here comes from the corresponding
    /// upstream `config.json`; the floor-of-1 branch protects tiny
    /// synthetic fixtures without silently falling to 0 (which would
    /// trip the loader's `n_heads == 0` guard).
    #[test]
    fn derive_n_heads_matches_hf_convention() {
        // Real HF DeBERTa checkpoints.
        assert_eq!(derive_n_heads(1024), 16, "deberta-*-large: 16 heads");
        assert_eq!(derive_n_heads(768), 12, "deberta-*-base: 12 heads");
        assert_eq!(derive_n_heads(384), 6, "deberta-*-small: 6 heads");
        // Synthetic tiny fixtures — must not collapse to 0 (would trip
        // loader's `n_heads == 0` guard, silently break the round-trip
        // test); must divide `d_model` evenly (loader's `d_model %
        // n_heads == 0` guard).
        assert_eq!(derive_n_heads(8), 1, "tiny fixture d_model=8 → single head");
        assert_eq!(derive_n_heads(4), 1, "even tinier — floor still 1, not 0");
        assert_eq!(derive_n_heads(64), 1, "d_model=head_dim exactly → 1 head");
        assert_eq!(derive_n_heads(128), 2, "d_model=2*head_dim → 2 heads");
    }

    /// Blocker 5 (2026-08-06) — a `vocab.txt` that is not valid UTF-8
    /// is a loud parse error, never silently truncated to what happens
    /// to be UTF-8 up to the first invalid byte.
    #[test]
    fn non_utf8_vocab_txt_is_a_loud_error() {
        // A single high byte 0xFF that is never a valid leading UTF-8
        // byte (RFC 3629 §4). Emits a `Parse` error.
        let vocab_txt = b"[PAD]\n\xff\n";
        let blob = safetensors_multi(&base_fixture());
        let (input, output) = temp_pair("tok-non-utf8");
        std::fs::write(&input, &blob).expect("write input safetensors");
        let err = convert_deberta_v2_file(&input, &output, None, Some(vocab_txt))
            .expect_err("non-UTF-8 tokenizer bytes must be refused");
        assert!(matches!(err, ConvertError::Parse(_)));
        std::fs::remove_file(&input).ok();
    }
}
