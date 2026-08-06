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
//! [`infer_vocab_and_d_model`]). `n_heads` (16) and `n_pos_buckets` /
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
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

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
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input, or
/// when no tensor looks like a token-embedding table (see
/// [`infer_vocab_and_d_model`] — `vocab_size` has no default in
/// `DebertaV2Encoder::from_gguf`, so this converter refuses to invent one);
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_deberta_v2_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ConvertReport, ConvertError> {
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
    write_hparams(
        &mut b,
        "vokra.bert.deberta_v2",
        n_layers,
        d_model,
        vocab_size,
    );

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
            // key_proj — duplicate it to satisfy the Rust struct layout.
            "attention.self.query_proj.weight" => Some(MapAction::Duplicate(
                format!("{p}.attn.wq.weight"),
                format!("{p}.attn.wq_pos.weight"),
            )),
            "attention.self.query_proj.bias" => {
                Some(MapAction::Rename(format!("{p}.attn.wq.bias")))
            }
            "attention.self.key_proj.weight" => Some(MapAction::Duplicate(
                format!("{p}.attn.wk.weight"),
                format!("{p}.attn.wk_pos.weight"),
            )),
            "attention.self.key_proj.bias" => Some(MapAction::Rename(format!("{p}.attn.wk.bias"))),
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

/// Writes the shared six-key `vokra.bert.<arch>.*` hparam chunk group
/// (`n_layers` / `d_model` / `n_heads` / `vocab_size` / `n_pos_buckets` /
/// `max_pos_dist`) under `prefix`. `n_heads` / `n_pos_buckets` /
/// `max_pos_dist` are not derivable from tensor shapes (see module doc)
/// and are written as the same placeholder values
/// `DebertaV2Encoder::from_gguf` / `DebertaV3Encoder::from_gguf` already
/// default to when the key is absent — this converter writes them
/// explicitly so the GGUF is self-describing rather than relying on a
/// downstream default.
pub(crate) fn write_hparams(
    b: &mut GgufBuilder,
    prefix: &str,
    n_layers: u32,
    d_model: u64,
    vocab_size: u64,
) {
    b.add_u32(&format!("{prefix}.n_layers"), n_layers);
    b.add_u32(&format!("{prefix}.d_model"), d_model as u32);
    b.add_u32(&format!("{prefix}.n_heads"), 16); // "large" convention — assumed, unverified (Task 30)
    b.add_u32(&format!("{prefix}.vocab_size"), vocab_size as u32);
    b.add_u32(&format!("{prefix}.n_pos_buckets"), 512); // loader default — not independently derived
    b.add_u32(&format!("{prefix}.max_pos_dist"), 512); // loader default — not independently derived
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
    /// `position_embeddings` table below, so [`infer_vocab_and_d_model`]
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

        let report = convert_deberta_v2_file(&input, &output, None).expect("convert");
        // After Task 30 rename table (2026-08-06): 4 upstream tensors map to
        // 5 written entries:
        //   - `word_embeddings` → renamed to `bert.embed.weight` (1 written)
        //   - `position_embeddings` → skipped (absolute pos embed, disent
        //     attention uses rel_embeddings only — see classify_skip)
        //   - `layer.0.attention.self.query_proj.weight` → duplicated
        //     (`wq.weight` + `wq_pos.weight`, both F32 = 2 written)
        //   - `layer.2.attention.self.query_proj.weight` → duplicated (BF16,
        //     both copies BF16 = 2 written, 2 bf16_passthrough)
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
        assert_eq!(
            file.get("vokra.bert.deberta_v2.n_heads")
                .and_then(|v| v.as_u64()),
            Some(16)
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

        convert_deberta_v2_file(&input, &output, Some("apache-2.0")).expect("convert");

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

        let err = convert_deberta_v2_file(&input, &output, None).expect_err("must fail loudly");
        assert!(matches!(err, ConvertError::Parse(_)));
        assert!(!output.exists(), "no partial GGUF must be left behind");

        std::fs::remove_file(&input).ok();
    }
}
