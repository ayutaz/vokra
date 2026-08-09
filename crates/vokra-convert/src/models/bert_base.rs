//! **Plain BERT** (`hfl/chinese-roberta-wwm-ext-large` first consumer):
//! safetensors → GGUF conversion (WP-14, 2026-08-10).
//!
//! Input: an upstream HF `transformers` `BertForMaskedLM` /
//! `BertModel` safetensors checkpoint. Output: a GGUF carrying every
//! float tensor renamed into the `bert_base.*` hierarchy that
//! [`BertBaseEncoder::from_gguf`](https://docs.rs/vokra-bert) reads
//! (`crates/vokra-bert/src/bert_base.rs`), plus the `vokra.bert_base.*`
//! hparam chunk group and — optionally — a
//! [`BertWordpieceTokenizer::from_gguf`](https://docs.rs/vokra-bert)-compatible
//! `vokra.bert.wordpiece.*` chunk group built from an upstream
//! `vocab.txt`.
//!
//! # First consumer
//!
//! `hfl/chinese-roberta-wwm-ext-large` (`BertForMaskedLM`, **Apache-2.0**,
//! whole-word-masking Chinese BERT-large from HFL; 21128-piece WordPiece
//! vocab, 1024 hidden, 24 layers, 16 heads). Wired into the SBV2 v2
//! `language_id = 2` (ZH) BERT slot at load time — see
//! `crates/vokra-models/src/sbv2/mod.rs::SbV2Model::from_gguf_with_zh_bert`.
//!
//! # References (permissive only)
//!
//! - Devlin, Chang, Lee, Toutanova 2018 (arXiv:1810.04805) — BERT paper
//! - google-research/bert (Apache-2.0) — reference tensor names
//! - HuggingFace transformers `modeling_bert.py` (Apache-2.0) — the
//!   authoritative `bert.embeddings.*` / `bert.encoder.layer.<i>.*`
//!   tensor-name convention this converter maps FROM.
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
//! - github.com/fishaudio/Bert-VITS2 (AGPL-3.0)
//!
//! # Structural difference vs [`crate::models::deberta_v2`]
//!
//! Plain BERT is arch-different from DeBERTa v2/v3 (see the target
//! module's own doc `crates/vokra-bert/src/bert_base.rs`):
//!
//! - **Embeddings**: sum of three learned tables (word + learned
//!   absolute position + token_type) → LayerNorm. DeBERTa v2/v3 use
//!   disentangled relative position instead.
//! - **Self-attention**: standard `softmax(Q·K^T / sqrt(d_head)) · V`.
//!   No `wq_pos` / `wk_pos` / `pos_embed` duplication path (there is no
//!   `deberta.encoder.rel_embeddings.weight` analogue in BERT).
//! - **Layer order**: **post-norm** — LayerNorm is applied *after* the
//!   residual add. Weight names carry `output.LayerNorm` (attn) /
//!   `output.LayerNorm` (FFN) rather than v2's `attention.output.LayerNorm`
//!   → `ln1` / `output.LayerNorm` → `ln2` pre-norm convention. This
//!   converter renames the two attn/FFN norm tensor pairs into
//!   `attention.output.layernorm.{gamma,beta}` / `output.layernorm.{gamma,beta}`
//!   so `BertBaseEncoder::from_gguf` loads them at the post-norm slot
//!   inside each `BertSelfOutput` / `BertOutput`.
//!
//! # Tokenizer emission (`vokra.bert.wordpiece.*`, prefix rationale)
//!
//! The runtime-side loader is [`BertWordpieceTokenizer::from_gguf`], whose
//! metadata contract is `{prefix}.vocab` (ARRAY<STRING>) plus the four
//! special-token id keys and the optional `do_lower_case` boolean. The
//! prefix is caller-picked; SBV2 v2 passes `vokra.bert.wordpiece` (see
//! `SbV2Model::from_gguf_with_zh_bert`), so **this converter emits under
//! the same prefix** — a standalone `BertBaseEncoder` GGUF loads
//! identically. A `vokra.bert.wordpiece.kind = "bert-wordpiece"`
//! discriminator is also stamped for auditability (mirror of v2/v3's
//! `vokra.bert.tokenizer.kind`), even though the current loader does
//! not read it — future kind-checked variants (Chinese char split,
//! byte-level, ...) can then loud-refuse a mis-schema'd artifact.
//!
//! **Distinct from the SBV2 SentencePiece prefix**: the JA/EN BERT
//! branches use `vokra.bert.tokenizer.*` (SentencePiece unigram / char
//! split — see [`crate::models::deberta_v2::KEY_TOKENIZER_PREFIX`]);
//! WordPiece and SentencePiece read different schema keys
//! (`{prefix}.vocab` vs `{prefix}.pieces` + `{prefix}.scores`), so a
//! distinct chunk-group prefix is what prevents a downstream tool from
//! silently loading a WordPiece vocab as a SentencePiece piece table.
//!
//! # BF16 pass-through — mirror of `funcodec` / `wespeaker` / `deberta_v2`
//!
//! F32 / F16 / BF16 tensors pass through **verbatim** under their
//! renamed target names (GGUF types 0 / 1 / 30). No convert-time
//! widening — the runtime widens BF16 → f32 losslessly at load via the
//! single choke point `crates/vokra-core/src/gguf/quant/mod.rs
//! decode_bf16` (BF16 is the top 16 bits of an f32 — `bits << 16` is
//! exact).
//!
//! # `do_lower_case` default
//!
//! HF `hfl/chinese-roberta-wwm-ext-large` ships
//! `tokenizer_config.json { "do_lower_case": false }` — the Chinese
//! branch tokenizer preserves case (a CJK glyph has no case, so
//! lower-casing is a Latin-only concern and disabling it is the correct
//! primary-source-verified default). The [`convert_bert_base_file`]
//! contract exposes `do_lower_case` as a caller argument (with a
//! `false` default for the first consumer) so the same converter serves
//! future English WordPiece checkpoints that need `true` without a new
//! `--variant` flag axis.
//!
//! # No ONNX (permanent)
//!
//! BERT ships as an HF safetensors checkpoint (or an older `pytorch_model.bin`
//! which callers convert via `tools/parity/bin_to_safetensors.py`); this
//! converter never touches ONNX (FR-LD-05).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for plain-BERT GGUFs.
pub(crate) const ARCH: &str = "bert_base";
/// `vokra.model.name` — short slug, distinct from the full HF `org/repo`
/// path (mirrors the `deberta_v2` / `funcodec` / `wespeaker` convention).
pub(crate) const NAME: &str = "chinese-roberta-wwm-ext-large";
/// Upstream Hugging Face repository path — provenance breadcrumb.
pub(crate) const UPSTREAM_HF: &str = "hfl/chinese-roberta-wwm-ext-large";
/// Upstream declared weight license (SPDX id, lower-case per
/// `docs/license-audit.md` §3.1). `apache-2.0` classifies as
/// [`LicenseClass::Permissive`] — no runtime-side attribution
/// obligation, no share-alike cascade. Verified 2026-08-10 via HF
/// cardData `license: apache-2.0` on `hfl/chinese-roberta-wwm-ext-large`
/// (primary source — CLAUDE.md「ハルシネーション厳禁」).
pub(crate) const DEFAULT_LICENSE: &str = "apache-2.0";
/// `vokra.model.category` for every BERT-family encoder GGUF (v2 / v3 /
/// plain-BERT alike — reuses the DeBERTa converter's constant).
pub(crate) const CATEGORY: &str = "bert";

/// `vokra.model.category` metadata key — mirror of the same constant in
/// [`crate::models::deberta_v2::KEY_MODEL_CATEGORY`]; kept local to this
/// module rather than re-imported so the two converters remain
/// independent (a future refactor could centralize).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// `vokra.provenance.upstream_hf` metadata key — converter-local for
/// the same reason as [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Metadata-key prefix consumed by
/// `vokra_bert::wordpiece::BertWordpieceTokenizer::from_gguf` when SBV2
/// v2 loads its ZH branch (see `SbV2Model::from_gguf_with_zh_bert`).
/// A standalone `BertBaseEncoder` GGUF that shares this prefix loads
/// identically — the prefix is caller-supplied, not hard-coded in the
/// tokenizer reader.
pub(crate) const KEY_WORDPIECE_PREFIX: &str = "vokra.bert.wordpiece";

/// Discriminator value stamped under `{prefix}.kind` for a WordPiece
/// tokenizer. Distinguishes WordPiece's `{prefix}.vocab` schema from
/// v2/v3's `{prefix}.pieces` + `{prefix}.scores` SentencePiece schema so
/// a downstream tool can loud-refuse rather than silently mis-tokenize.
pub(crate) const KIND_BERT_WORDPIECE: &str = "bert-wordpiece";

/// HF BERT canonical special-token ids
/// (`google-bert/bert-base-uncased` / `hfl/chinese-roberta-wwm-ext-large`
/// / every WordPiece checkpoint that follows the reference layout):
/// `[PAD] = 0`, `[UNK] = 100`, `[CLS] = 101`, `[SEP] = 102`. Verified
/// against upstream `tokenizer_config.json` /
/// `special_tokens_map.json` on `hfl/chinese-roberta-wwm-ext-large`
/// (2026-08-10). Matches
/// [`vokra_bert::wordpiece::BertWordpieceTokenizer::from_gguf`]'s
/// defaults (the reader falls back to these when the converter omits
/// the keys; this converter always writes them explicitly so the
/// artifact does not depend on a loader-side default).
pub(crate) const BERT_PAD_ID: u32 = 0;
pub(crate) const BERT_UNK_ID: u32 = 100;
pub(crate) const BERT_CLS_ID: u32 = 101;
pub(crate) const BERT_SEP_ID: u32 = 102;

/// Outcome of a plain-BERT conversion — mirror of
/// [`crate::models::deberta_v2::ConvertReport`], but not shared: keeping
/// the type local prevents a converter-boundary rename in DeBERTa
/// (share-alike ID → attribution ID drift, for instance) from silently
/// changing this converter's reported field semantics.
#[derive(Debug, Default)]
pub struct BertBaseReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors renamed and written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time today,
    /// so this arm is unreachable in practice; kept for parity with the
    /// sibling converters' counter shape).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim.
    pub bf16_passthrough: usize,
    /// Tensors that did not match any known upstream pattern and were
    /// intentionally dropped (see [`classify_skip`] — MLM head, pooler,
    /// position_ids int buffer, …). Structured stderr log lists each
    /// name+reason at conversion time.
    pub skipped_unmapped: usize,
}

/// Converts a plain BERT safetensors checkpoint at `input` into a Vokra
/// GGUF at `output`.
///
/// - `license` overrides the upstream `apache-2.0` stamp (mirror of the
///   `convert_file --license <spdx>` boundary in
///   [`crate::convert_file_licensed`]). Useful when the caller ships a
///   downstream fine-tune under a distinct SPDX id, or wants an
///   audit-visible re-declaration of the upstream permissive stamp.
/// - `vocab_txt_bytes` optionally stamps the
///   `vokra.bert.wordpiece.*` chunk group
///   `BertWordpieceTokenizer::from_gguf` reads. The bytes are treated as
///   a WordPiece `vocab.txt` (one piece per line, UTF-8) — the upstream
///   `hfl/chinese-roberta-wwm-ext-large` layout. When [`None`], no
///   tokenizer metadata is written — SBV2 v2's
///   `from_gguf_with_zh_bert` will then loud-fail per FR-EX-08 rather
///   than silently substituting an all-`[UNK]` tokenizer.
/// - `do_lower_case` controls the `vokra.bert.wordpiece.do_lower_case`
///   bool. For the first consumer (`hfl/chinese-roberta-wwm-ext-large`),
///   pass `false` (Chinese branch — verified against upstream
///   `tokenizer_config.json`). English WordPiece checkpoints
///   (`bert-base-uncased`, ...) that carry cased inputs but expect
///   lower-casing pass `true`.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input,
/// an empty `vocab_txt_bytes` argument, non-UTF-8 vocab bytes, or when
/// the checkpoint carries no `bert.embeddings.word_embeddings.weight`
/// tensor (`vocab_size` / `hidden_size` are derived from its shape and
/// have no default — FR-EX-08); [`ConvertError::Gguf`] if the GGUF
/// serialization fails.
pub fn convert_bert_base_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    vocab_txt_bytes: Option<&[u8]>,
    do_lower_case: bool,
) -> Result<BertBaseReport, ConvertError> {
    // Front-load the tokenizer refuse — an empty vocab.txt cannot become
    // a valid `vokra.bert.wordpiece.*` chunk group
    // (`BertWordpieceTokenizer::from_vocab` refuses a zero-length vocab
    // with its own explicit error), so drop out here before touching the
    // safetensors input. Mirror of the `deberta_v2 --tokenizer` empty
    // gate.
    if let Some(t) = vocab_txt_bytes
        && t.is_empty()
    {
        return Err(ConvertError::Parse(
            "bert-base --tokenizer: file is empty — refusing to emit a zero-length \
             vokra.bert.wordpiece.* chunk group (BertWordpieceTokenizer::from_gguf would \
             fail to load)"
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

    // Hparams — shape-derived from the checkpoint where possible; the
    // remaining three (`heads`, `ffn`, `max_pos`) come from
    // shape-derivable siblings so a hypothetical -base variant with
    // different geometry loads correctly without a per-variant flag.
    let dims = infer_dims(&st)?;
    write_hparams(&mut b, &dims);

    // Tokenizer side-car (optional — see the function's own doc).
    if let Some(vt) = vocab_txt_bytes {
        write_tokenizer_vocab_txt(&mut b, vt, do_lower_case)?;
    }

    let mut report = BertBaseReport::default();
    let mut skipped_names: Vec<(String, &'static str)> = Vec::new();
    let mut renamed_count = 0usize;

    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => match map_bert_name(&t.name) {
                Some(new_name) => {
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
                None => {
                    let reason = classify_skip(&t.name);
                    skipped_names.push((t.name.clone(), reason));
                    report.skipped_unmapped += 1;
                }
            },
            _ => report.skipped_non_float += 1,
        }
    }

    for (name, reason) in &skipped_names {
        eprintln!("convert_bert_base: skipping tensor `{name}` ({reason})");
    }
    eprintln!(
        "convert_bert_base: {renamed_count} renamed, {} unmapped skipped, {} non-float skipped",
        skipped_names.len(),
        report.skipped_non_float,
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

/// Hparam bundle recovered from the checkpoint shapes, plus the two
/// tokenizer-independent axes that carry HF-canonical defaults.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BertDims {
    pub n_layers: u32,
    pub hidden: u32,
    pub heads: u32,
    pub ffn: u32,
    pub vocab: u32,
    pub max_pos: u32,
    pub type_vocab: u32,
    pub layer_norm_eps: f32,
}

/// HF BERT convention: `head_dim = 64` (both `bert-base-uncased` at
/// 768/12 and `bert-large-uncased` / `hfl/chinese-roberta-wwm-ext-large`
/// at 1024/16 satisfy `hidden / 64`). Every public HF BERT checkpoint
/// we target holds this constant; `hidden / HEAD_DIM_HF_DEFAULT` is
/// therefore a shape-derived (not invented) value.
pub(crate) const HEAD_DIM_HF_DEFAULT: u32 = 64;

/// HF BERT convention: `FFN inner = 4 × hidden`. Every public HF BERT
/// checkpoint we target holds this constant.
pub(crate) const FFN_MULTIPLIER_HF_DEFAULT: u32 = 4;

/// HF BERT convention: `max_position_embeddings = 512`. Every public HF
/// BERT checkpoint we target holds this constant; if a fine-tune with a
/// longer context arrives it can override via a future `--config` arg.
pub(crate) const MAX_POSITION_HF_DEFAULT: u32 = 512;

/// HF BERT convention: `type_vocab_size = 2` (segment A / B). NSP-free
/// variants collapse to 1; `hfl/chinese-roberta-wwm-ext-large` ships
/// the canonical `2`. Shape-derived from
/// `bert.embeddings.token_type_embeddings.weight` when present.
pub(crate) const TYPE_VOCAB_HF_DEFAULT: u32 = 2;

/// HF BERT convention: `layer_norm_eps = 1e-12`
/// (`configuration_bert.BertConfig.layer_norm_eps`, unchanged since
/// upstream commit 6bd028e in 2019). Matches the loader-side default.
pub(crate) const LAYER_NORM_EPS_HF_DEFAULT: f32 = 1e-12;

/// Derive `heads` from `hidden` via [`HEAD_DIM_HF_DEFAULT`] with a
/// floor of 1 (single-head fallback for tiny synthetic fixtures; the
/// real HF checkpoints we target always yield the correct 12 / 16).
pub(crate) fn derive_heads(hidden: u32) -> u32 {
    (hidden / HEAD_DIM_HF_DEFAULT).max(1)
}

/// Reads `(vocab, hidden)` off the token-embedding table's shape (must
/// be present — no default), `n_layers` from the highest per-layer
/// tensor index (offset by 1), `max_pos` and `type_vocab` from their
/// own embedding-table shapes when present (else the HF defaults). The
/// FFN inner width is 4×hidden by the canonical HF BERT convention.
pub(crate) fn infer_dims(st: &SafetensorsFile) -> Result<BertDims, ConvertError> {
    // Word embedding table: `[vocab, hidden]`. Mandatory — see the
    // `no_word_embedding_is_a_loud_error` test.
    let we = st
        .tensor_info("bert.embeddings.word_embeddings.weight")
        .ok_or_else(|| {
            ConvertError::Parse(
                "no `bert.embeddings.word_embeddings.weight` tensor — expected the token \
                 embedding table to derive vocab_size / hidden_size from"
                    .to_owned(),
            )
        })?;
    if we.shape.len() != 2 {
        return Err(ConvertError::Parse(format!(
            "bert.embeddings.word_embeddings.weight rank {} != 2",
            we.shape.len()
        )));
    }
    let vocab = u32::try_from(we.shape[0]).map_err(|_| {
        ConvertError::Parse("vocab exceeds u32::MAX in word_embeddings shape".to_owned())
    })?;
    let hidden = u32::try_from(we.shape[1]).map_err(|_| {
        ConvertError::Parse("hidden exceeds u32::MAX in word_embeddings shape".to_owned())
    })?;
    if hidden == 0 {
        return Err(ConvertError::Parse(
            "word_embeddings.weight second dim is 0 — hidden must be positive".to_owned(),
        ));
    }

    // Position + token-type tables — shape-derivable when present, else
    // fall back to HF defaults (fixtures without them are for synthetic
    // shape testing).
    let max_pos = st
        .tensor_info("bert.embeddings.position_embeddings.weight")
        .and_then(|t| t.shape.first().copied())
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(MAX_POSITION_HF_DEFAULT);
    let type_vocab = st
        .tensor_info("bert.embeddings.token_type_embeddings.weight")
        .and_then(|t| t.shape.first().copied())
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(TYPE_VOCAB_HF_DEFAULT);

    // n_layers: highest `bert.encoder.layer.<N>.` index observed + 1.
    // Mirrors the deberta_v2 `count_layers` scan approach but anchored
    // to the BERT prefix so v2's `deberta.encoder.layer.<N>` names
    // cannot accidentally count (defense in depth against a mistaken
    // shared fixture).
    let mut max_layer_idx: Option<u32> = None;
    for t in st.tensors() {
        if let Some(rest) = t.name.strip_prefix("bert.encoder.layer.")
            && let Some((idx_str, _)) = rest.split_once('.')
            && let Ok(idx) = idx_str.parse::<u32>()
        {
            max_layer_idx = Some(max_layer_idx.map_or(idx, |m| m.max(idx)));
        }
    }
    // Small default (2) rather than the -large 24 so a synthetic
    // fixture with only embeddings still round-trips through
    // `BertBaseEncoder::from_gguf` (which needs `n_layers >= 1`
    // consistent with the actual tensors present).
    let n_layers = max_layer_idx.map_or(0, |m| m + 1);

    // FFN inner width — shape-derived from the `intermediate.dense` FFN
    // tensor when present (any layer will do — they all share the same
    // hparam), else the HF `4 * hidden` default. Prefer the explicit
    // shape read so a future 8×hidden or 2×hidden fine-tune converts
    // correctly without a caller-supplied flag.
    let ffn = st
        .tensors()
        .iter()
        .find(|t| {
            t.name
                .strip_prefix("bert.encoder.layer.")
                .and_then(|r| r.split_once('.'))
                .map(|(_, tail)| tail == "intermediate.dense.weight")
                .unwrap_or(false)
        })
        .and_then(|t| t.shape.first().copied())
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(hidden.saturating_mul(FFN_MULTIPLIER_HF_DEFAULT));

    Ok(BertDims {
        n_layers,
        hidden,
        heads: derive_heads(hidden),
        ffn,
        vocab,
        max_pos,
        type_vocab,
        layer_norm_eps: LAYER_NORM_EPS_HF_DEFAULT,
    })
}

/// Writes the `vokra.bert_base.*` hparam chunk group. Every key
/// mirrors what [`vokra_bert::bert_base::BertBaseEncoder::from_gguf`]
/// reads — a mismatch here surfaces as a loud `VokraError::ModelLoad`
/// at load time (FR-EX-08).
pub(crate) fn write_hparams(b: &mut GgufBuilder, dims: &BertDims) {
    b.add_u32("vokra.bert_base.n_layers", dims.n_layers);
    b.add_u32("vokra.bert_base.hidden", dims.hidden);
    b.add_u32("vokra.bert_base.heads", dims.heads);
    b.add_u32("vokra.bert_base.ffn", dims.ffn);
    b.add_u32("vokra.bert_base.vocab", dims.vocab);
    b.add_u32("vokra.bert_base.max_pos", dims.max_pos);
    b.add_u32("vokra.bert_base.type_vocab", dims.type_vocab);
    b.add_f32("vokra.bert_base.layer_norm_eps", dims.layer_norm_eps);
}

/// Maps one upstream HF BERT tensor name to the `bert_base.*` name that
/// [`vokra_bert::bert_base::BertBaseEncoder::from_gguf`] expects.
/// Returns [`None`] for tensors that are intentionally not consumed by
/// the encoder (MLM head, pooler, `position_ids` int buffer — see
/// [`classify_skip`] for the reason categories).
pub(crate) fn map_bert_name(upstream: &str) -> Option<String> {
    // Embeddings.
    if upstream == "bert.embeddings.word_embeddings.weight" {
        return Some("bert_base.embeddings.word_embed".to_owned());
    }
    if upstream == "bert.embeddings.position_embeddings.weight" {
        return Some("bert_base.embeddings.position_embed".to_owned());
    }
    if upstream == "bert.embeddings.token_type_embeddings.weight" {
        return Some("bert_base.embeddings.token_type_embed".to_owned());
    }
    if upstream == "bert.embeddings.LayerNorm.weight" {
        return Some("bert_base.embeddings.layernorm.gamma".to_owned());
    }
    if upstream == "bert.embeddings.LayerNorm.bias" {
        return Some("bert_base.embeddings.layernorm.beta".to_owned());
    }

    // Per-encoder-layer transformer stack. HF ships `attention.self.
    // {query,key,value}.{weight,bias}` — plain BERT with no `query_proj`
    // suffix (that's DeBERTa v2's naming), so `map_deberta_name` cannot
    // be shared verbatim.
    if let Some(rest) = upstream.strip_prefix("bert.encoder.layer.") {
        let (idx_str, tail) = rest.split_once('.')?;
        let i: usize = idx_str.parse().ok()?;
        let p = format!("bert_base.encoder.layer.{i}");
        return match tail {
            // Attention Q/K/V projections.
            "attention.self.query.weight" => Some(format!("{p}.attention.query.weight")),
            "attention.self.query.bias" => Some(format!("{p}.attention.query.bias")),
            "attention.self.key.weight" => Some(format!("{p}.attention.key.weight")),
            "attention.self.key.bias" => Some(format!("{p}.attention.key.bias")),
            "attention.self.value.weight" => Some(format!("{p}.attention.value.weight")),
            "attention.self.value.bias" => Some(format!("{p}.attention.value.bias")),
            // Attention output projection + post-norm.
            "attention.output.dense.weight" => Some(format!("{p}.attention.output.dense.weight")),
            "attention.output.dense.bias" => Some(format!("{p}.attention.output.dense.bias")),
            "attention.output.LayerNorm.weight" => {
                Some(format!("{p}.attention.output.layernorm.gamma"))
            }
            "attention.output.LayerNorm.bias" => {
                Some(format!("{p}.attention.output.layernorm.beta"))
            }
            // FFN inner (BertIntermediate) + outer + post-norm (BertOutput).
            "intermediate.dense.weight" => Some(format!("{p}.intermediate.dense.weight")),
            "intermediate.dense.bias" => Some(format!("{p}.intermediate.dense.bias")),
            "output.dense.weight" => Some(format!("{p}.output.dense.weight")),
            "output.dense.bias" => Some(format!("{p}.output.dense.bias")),
            "output.LayerNorm.weight" => Some(format!("{p}.output.layernorm.gamma")),
            "output.LayerNorm.bias" => Some(format!("{p}.output.layernorm.beta")),
            _ => None,
        };
    }

    None
}

/// Categorizes why an upstream tensor was skipped, for structured
/// stderr logs (FR-EX-08 posture — never silently drop tensors without
/// stating the reason). Mirror of
/// [`crate::models::deberta_v2::classify_skip`] adapted to the plain
/// BERT namespace.
pub(crate) fn classify_skip(name: &str) -> &'static str {
    // MLM head (`BertForMaskedLM`). `BertBaseEncoder` returns encoder
    // hidden states only.
    if name.starts_with("cls.") {
        return "MLM head — BertBaseEncoder consumes encoder hidden states only";
    }
    // NSP / classifier head.
    if name.starts_with("classifier.") || name.starts_with("qa_outputs.") {
        return "downstream classifier head — encoder-only loader does not consume";
    }
    // BERT pooler (mean-pool over CLS).
    if name.starts_with("bert.pooler.") {
        return "pooler — BertBaseEncoder returns last hidden state, not a pooled vector";
    }
    // Position-id buffer (int, not float, derivable at runtime as
    // `arange(0, seq_len)`).
    if name == "bert.embeddings.position_ids" {
        return "position_ids buffer — derivable at runtime as arange(0, seq_len)";
    }
    "unmapped tensor — no rename rule matched"
}

/// Parses a WordPiece `vocab.txt` (one piece per line, UTF-8) and
/// stamps the `vokra.bert.wordpiece.*` chunk group
/// [`vokra_bert::wordpiece::BertWordpieceTokenizer::from_gguf`] reads.
/// Trailing blank lines are stripped (upstream WordPiece treats them as
/// `[UNK]` placeholders, but the reader assigns ids by position and a
/// blank piece is ambiguous — mirror of the `deberta_v2` handling).
///
/// # Errors
///
/// [`ConvertError::Parse`] when the bytes are not valid UTF-8 or the
/// vocab has no non-empty lines.
pub(crate) fn write_tokenizer_vocab_txt(
    b: &mut GgufBuilder,
    bytes: &[u8],
    do_lower_case: bool,
) -> Result<(), ConvertError> {
    let text = std::str::from_utf8(bytes).map_err(|_| {
        ConvertError::Parse("bert-base --tokenizer: vocab.txt is not valid UTF-8".to_owned())
    })?;
    let mut pieces: Vec<String> = text.lines().map(str::to_owned).collect();
    while pieces.last().is_some_and(String::is_empty) {
        pieces.pop();
    }
    if pieces.is_empty() {
        return Err(ConvertError::Parse(
            "bert-base --tokenizer: vocab.txt has no non-empty lines".to_owned(),
        ));
    }
    let vocab_size = u32::try_from(pieces.len()).map_err(|_| {
        ConvertError::Parse(format!(
            "bert-base --tokenizer: pieces.len() ({}) exceeds u32::MAX",
            pieces.len()
        ))
    })?;

    // `kind` discriminator — auditability only; the reader does not
    // check it today (see this module's own doc "Tokenizer emission"
    // section).
    b.add_string(&format!("{KEY_WORDPIECE_PREFIX}.kind"), KIND_BERT_WORDPIECE);
    // `vocab` — the load-bearing array the reader consumes verbatim.
    b.add_metadata(
        &format!("{KEY_WORDPIECE_PREFIX}.vocab"),
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: pieces
                .iter()
                .map(|s| GgufMetadataValue::String(s.clone()))
                .collect(),
        }),
    );
    // Special-token ids — always written explicitly so the artifact
    // does not depend on the reader's default-fallback behavior.
    b.add_u32(&format!("{KEY_WORDPIECE_PREFIX}.unk_id"), BERT_UNK_ID);
    b.add_u32(&format!("{KEY_WORDPIECE_PREFIX}.cls_id"), BERT_CLS_ID);
    b.add_u32(&format!("{KEY_WORDPIECE_PREFIX}.sep_id"), BERT_SEP_ID);
    b.add_u32(&format!("{KEY_WORDPIECE_PREFIX}.pad_id"), BERT_PAD_ID);
    b.add_bool(
        &format!("{KEY_WORDPIECE_PREFIX}.do_lower_case"),
        do_lower_case,
    );
    // `vocab_size` mirrors what v2/v3 stamp for auditability; the
    // reader itself derives length from the `vocab` array's length so
    // this is informational (matches v2 precedent).
    b.add_u32(&format!("{KEY_WORDPIECE_PREFIX}.vocab_size"), vocab_size);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Builds a multi-tensor safetensors byte buffer from `(name,
    /// dtype, shape, payload)` entries — mirror of the helper the
    /// DeBERTa converter tests share.
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

    fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn temp_pair(label: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let mut input = std::env::temp_dir();
        input.push(format!(
            "vokra-bert-base-{label}-{}-in.safetensors",
            std::process::id()
        ));
        let mut output = std::env::temp_dir();
        output.push(format!(
            "vokra-bert-base-{label}-{}-out.gguf",
            std::process::id()
        ));
        (input, output)
    }

    /// Minimal shape-meaningful fixture: word_embeddings (vocab=6,
    /// hidden=4), a position_embeddings, a token_type_embeddings, one
    /// per-layer tensor at `layer.0` and one at `layer.2` (so
    /// `infer_dims` must resolve `n_layers = 3`, not `2`), and a BF16
    /// value_proj to prove pass-through.
    fn base_fixture() -> Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> {
        vec![
            (
                "bert.embeddings.word_embeddings.weight",
                "F32",
                &[6, 4],
                f32_bytes(&[0.01; 24]),
            ),
            (
                "bert.embeddings.position_embeddings.weight",
                "F32",
                &[8, 4],
                f32_bytes(&[0.02; 32]),
            ),
            (
                "bert.embeddings.token_type_embeddings.weight",
                "F32",
                &[2, 4],
                f32_bytes(&[0.03; 8]),
            ),
            (
                "bert.encoder.layer.0.attention.self.query.weight",
                "F32",
                &[4, 4],
                f32_bytes(&[0.04; 16]),
            ),
            (
                "bert.encoder.layer.2.attention.self.value.weight",
                "BF16",
                &[4, 4],
                bf16_bytes(&[0.05; 16]),
            ),
        ]
    }

    /// Hparams are derived from real tensor shapes (vocab=6, hidden=4,
    /// max_pos=8, type_vocab=2, n_layers=3 — highest layer index 2 + 1);
    /// BF16 passes through as GGUF type 30; provenance defaults to
    /// `apache-2.0` / `Permissive`.
    #[test]
    fn hparams_and_bf16_derived_from_real_shapes() {
        let blob = safetensors_multi(&base_fixture());
        let (input, output) = temp_pair("hparams");
        std::fs::write(&input, &blob).expect("write input safetensors");

        let report = convert_bert_base_file(&input, &output, None, None, false).expect("convert");
        // 5 input tensors, all matching known rename rules, all F32/BF16.
        assert_eq!(report.read, 5);
        assert_eq!(report.written, 5);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.skipped_unmapped, 0);
        assert_eq!(report.bf16_passthrough, 1, "the layer.2 value_proj is BF16");

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        assert_eq!(
            file.get("vokra.bert_base.vocab").and_then(|v| v.as_u64()),
            Some(6)
        );
        assert_eq!(
            file.get("vokra.bert_base.hidden").and_then(|v| v.as_u64()),
            Some(4)
        );
        assert_eq!(
            file.get("vokra.bert_base.heads").and_then(|v| v.as_u64()),
            Some(1),
            "heads floor of 1 for tiny fixture (4/64=0 → 1)"
        );
        assert_eq!(
            file.get("vokra.bert_base.ffn").and_then(|v| v.as_u64()),
            Some(16),
            "FFN inner defaults to 4×hidden (16) when no intermediate.dense present"
        );
        assert_eq!(
            file.get("vokra.bert_base.n_layers")
                .and_then(|v| v.as_u64()),
            Some(3),
            "highest layer index observed is 2 (layer.0 + layer.2) → n_layers = 3"
        );
        assert_eq!(
            file.get("vokra.bert_base.max_pos").and_then(|v| v.as_u64()),
            Some(8),
            "shape-derived from position_embeddings first dim"
        );
        assert_eq!(
            file.get("vokra.bert_base.type_vocab")
                .and_then(|v| v.as_u64()),
            Some(2)
        );

        // Renamed tensor names.
        assert!(
            file.tensor_info("bert_base.embeddings.word_embed")
                .is_some()
        );
        assert!(
            file.tensor_info("bert_base.embeddings.position_embed")
                .is_some()
        );
        assert!(
            file.tensor_info("bert_base.embeddings.token_type_embed")
                .is_some()
        );
        assert!(
            file.tensor_info("bert_base.encoder.layer.0.attention.query.weight")
                .is_some()
        );
        let bf16_info = file
            .tensor_info("bert_base.encoder.layer.2.attention.value.weight")
            .expect("renamed BF16 tensor present");
        assert_eq!(
            bf16_info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 dtype preserved"
        );

        // Provenance.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 must classify as Permissive"
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

    /// `--license <spdx>` override replaces the default `apache-2.0`
    /// stamp entirely.
    #[test]
    fn license_override_replaces_default() {
        let blob = safetensors_multi(&base_fixture());
        let (input, output) = temp_pair("license-override");
        std::fs::write(&input, &blob).expect("write input safetensors");

        convert_bert_base_file(&input, &output, Some("mit"), None, false).expect("convert");

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// A checkpoint without the mandatory `bert.embeddings.
    /// word_embeddings.weight` is a loud [`ConvertError::Parse`], never
    /// a silently-invented vocab_size (FR-EX-08).
    #[test]
    fn no_word_embedding_is_a_loud_error() {
        let entries = vec![(
            "bert.encoder.layer.0.attention.self.query.weight",
            "F32",
            &[4u64, 4][..],
            f32_bytes(&[0.01; 16]),
        )];
        let blob = safetensors_multi(&entries);
        let (input, output) = temp_pair("no-word-embed");
        std::fs::write(&input, &blob).expect("write input safetensors");

        let err = convert_bert_base_file(&input, &output, None, None, false)
            .expect_err("must fail loudly");
        assert!(matches!(err, ConvertError::Parse(_)));
        assert!(!output.exists(), "no partial GGUF left behind");

        std::fs::remove_file(&input).ok();
    }

    /// `vocab_txt_bytes = None` emits **no** `vokra.bert.wordpiece.*`
    /// metadata — the runtime side loud-fails on
    /// `BertWordpieceTokenizer::from_gguf` (FR-EX-08 by design).
    #[test]
    fn no_tokenizer_bytes_emits_no_tokenizer_chunk() {
        let blob = safetensors_multi(&base_fixture());
        let (input, output) = temp_pair("no-tokenizer");
        std::fs::write(&input, &blob).expect("write input safetensors");

        convert_bert_base_file(&input, &output, None, None, false).expect("convert");

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");
        for suffix in [
            "kind",
            "vocab",
            "unk_id",
            "cls_id",
            "sep_id",
            "pad_id",
            "do_lower_case",
            "vocab_size",
        ] {
            let key = format!("{KEY_WORDPIECE_PREFIX}.{suffix}");
            assert!(
                file.get(&key).is_none(),
                "no --tokenizer supplied: `{key}` must NOT be present"
            );
        }

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// `vocab_txt_bytes = Some(vocab.txt)` stamps the full
    /// `vokra.bert.wordpiece.*` chunk group with `do_lower_case=false`
    /// (the ZH default) and every special-token id explicit.
    #[test]
    fn tokenizer_bytes_stamps_wordpiece_chunk_group() {
        // A tiny 5-line vocab.txt (the real fixture ships 21,128
        // lines; this proves the schema, not the size). Includes
        // `[PAD]` at line 0 (id 0) and `[CLS]` at line 4 (id 4).
        let vocab_txt = b"[PAD]\n[unused1]\n[unused2]\n[UNK]\n[CLS]\n";
        let blob = safetensors_multi(&base_fixture());
        let (input, output) = temp_pair("tokenizer");
        std::fs::write(&input, &blob).expect("write input safetensors");

        convert_bert_base_file(&input, &output, None, Some(vocab_txt), false).expect("convert");

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        assert_eq!(
            file.get(&format!("{KEY_WORDPIECE_PREFIX}.kind"))
                .and_then(|v| v.as_str()),
            Some(KIND_BERT_WORDPIECE)
        );
        let pieces = file
            .get(&format!("{KEY_WORDPIECE_PREFIX}.vocab"))
            .and_then(|v| v.as_array())
            .expect("vocab array present");
        assert_eq!(pieces.values.len(), 5);
        assert_eq!(pieces.values[0].as_str(), Some("[PAD]"));
        assert_eq!(pieces.values[3].as_str(), Some("[UNK]"));
        assert_eq!(pieces.values[4].as_str(), Some("[CLS]"));
        assert_eq!(
            file.get(&format!("{KEY_WORDPIECE_PREFIX}.unk_id"))
                .and_then(|v| v.as_u64()),
            Some(u64::from(BERT_UNK_ID))
        );
        assert_eq!(
            file.get(&format!("{KEY_WORDPIECE_PREFIX}.cls_id"))
                .and_then(|v| v.as_u64()),
            Some(u64::from(BERT_CLS_ID))
        );
        assert_eq!(
            file.get(&format!("{KEY_WORDPIECE_PREFIX}.sep_id"))
                .and_then(|v| v.as_u64()),
            Some(u64::from(BERT_SEP_ID))
        );
        assert_eq!(
            file.get(&format!("{KEY_WORDPIECE_PREFIX}.pad_id"))
                .and_then(|v| v.as_u64()),
            Some(u64::from(BERT_PAD_ID))
        );
        assert_eq!(
            file.get(&format!("{KEY_WORDPIECE_PREFIX}.do_lower_case"))
                .and_then(|v| v.as_bool()),
            Some(false),
            "ZH default is do_lower_case=false — verified against \
             hfl/chinese-roberta-wwm-ext-large tokenizer_config.json"
        );
        assert_eq!(
            file.get(&format!("{KEY_WORDPIECE_PREFIX}.vocab_size"))
                .and_then(|v| v.as_u64()),
            Some(5)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Empty `vocab_txt_bytes` is a loud usage error (FR-EX-08) —
    /// silently emitting a zero-length vocab would produce a GGUF that
    /// `BertWordpieceTokenizer::from_gguf` itself refuses.
    #[test]
    fn empty_tokenizer_bytes_is_a_loud_error() {
        let blob = safetensors_multi(&base_fixture());
        let (input, output) = temp_pair("tok-empty");
        std::fs::write(&input, &blob).expect("write input safetensors");
        let err = convert_bert_base_file(&input, &output, None, Some(&[]), false)
            .expect_err("empty tokenizer must be refused");
        assert!(matches!(err, ConvertError::Parse(_)));
        assert!(!output.exists(), "no partial GGUF left behind");
        std::fs::remove_file(&input).ok();
    }

    /// A `vocab.txt` that is not valid UTF-8 is a loud parse error,
    /// never silently truncated to the first invalid byte.
    #[test]
    fn non_utf8_vocab_txt_is_a_loud_error() {
        let vocab_txt = b"[PAD]\n\xff\n";
        let blob = safetensors_multi(&base_fixture());
        let (input, output) = temp_pair("tok-non-utf8");
        std::fs::write(&input, &blob).expect("write input safetensors");
        let err = convert_bert_base_file(&input, &output, None, Some(vocab_txt), false)
            .expect_err("non-UTF-8 tokenizer bytes must be refused");
        assert!(matches!(err, ConvertError::Parse(_)));
        std::fs::remove_file(&input).ok();
    }

    /// [`classify_skip`] returns audit-visible reasons for every arm.
    #[test]
    fn classify_skip_covers_common_head_and_buffer_names() {
        assert_eq!(
            classify_skip("cls.predictions.transform.dense.weight"),
            "MLM head — BertBaseEncoder consumes encoder hidden states only"
        );
        assert_eq!(
            classify_skip("classifier.weight"),
            "downstream classifier head — encoder-only loader does not consume"
        );
        assert_eq!(
            classify_skip("bert.pooler.dense.weight"),
            "pooler — BertBaseEncoder returns last hidden state, not a pooled vector"
        );
        assert_eq!(
            classify_skip("bert.embeddings.position_ids"),
            "position_ids buffer — derivable at runtime as arange(0, seq_len)"
        );
        assert_eq!(
            classify_skip("unknown.tensor.name"),
            "unmapped tensor — no rename rule matched"
        );
    }

    /// `derive_heads` matches the HF BERT convention (`head_dim = 64`)
    /// for every real target checkpoint. Floor-of-1 protects tiny
    /// synthetic fixtures without silently collapsing to 0 (which would
    /// trip the loader's `num_heads == 0` guard).
    #[test]
    fn derive_heads_matches_hf_convention() {
        assert_eq!(
            derive_heads(1024),
            16,
            "bert-large / chinese-roberta-wwm-ext-large"
        );
        assert_eq!(derive_heads(768), 12, "bert-base");
        assert_eq!(derive_heads(384), 6, "bert-small variant");
        assert_eq!(derive_heads(64), 1, "hidden = head_dim → 1 head");
        assert_eq!(derive_heads(4), 1, "tiny fixture — floor still 1, not 0");
    }
}
