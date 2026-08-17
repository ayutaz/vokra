//! **DeBERTa v3** (`microsoft/deberta-v3-large`): safetensors → GGUF
//! conversion (SBV2 v2 plan Task 11, 2026-07-26).
//!
//! Structurally this shares DeBERTa v2's shape-derived hparams, HF→`bert.*`
//! tensor-name mapping, Q/K content↔position duplication, tokenizer metadata,
//! and `ConvertReport`. The inference-relevant v3 delta is a single shared
//! relative-position table (§3.1, "gradient-disentangled embedding sharing"):
//! this converter applies the checkpoint's encoder-level LayerNorm once and
//! emits `bert.encoder.pos_embed.weight` once for every runtime layer to share.
//! It also stamps v3's distinct arch, MIT provenance, upstream repository, and
//! `vokra.bert.deberta_v3.*` metadata prefix.
//!
//! # References (permissive only)
//!
//! - He, Gao, Chen 2021 (arXiv:2111.09543)
//! - HuggingFace `transformers` `deberta_v3` (Apache-2.0)
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
//! - github.com/fishaudio/Bert-VITS2 (AGPL-3.0)
//!
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};
use vokra_core::json::{self, JsonValue};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;
use crate::spm_proto::{PieceType, parse_model};

use super::deberta_v2::ConvertReport;
use super::deberta_v2::{
    CATEGORY, KEY_MODEL_CATEGORY, KEY_PROVENANCE_UPSTREAM_HF, KEY_TOKENIZER_PREFIX,
    KIND_SENTENCEPIECE_UNIGRAM, MapAction, TokenizerMetadata, add_f32_array, add_string_array,
    apply_layer_norm_rows, classify_skip, count_layers, f32_slice_to_le_bytes, infer_n_pos_buckets,
    infer_vocab_and_d_model, map_deberta_name, widen_bytes_to_f32, write_hparams,
    write_tokenizer_metadata,
};

/// `vokra.model.arch` for DeBERTa v3 GGUFs.
pub(crate) const ARCH: &str = "deberta_v3";
/// `vokra.model.name` — short slug (mirrors `deberta_v2::NAME`'s
/// convention: no org prefix).
pub(crate) const NAME: &str = "deberta-v3-large";
/// Upstream Hugging Face repository path — provenance breadcrumb.
pub(crate) const UPSTREAM_HF: &str = "microsoft/deberta-v3-large";
/// Upstream declared weight license (SPDX id). `mit` classifies as
/// [`LicenseClass::Permissive`] — no runtime-side attribution obligation,
/// unlike v2's `cc-by-sa-4.0` default.
pub(crate) const DEFAULT_LICENSE: &str = "mit";

/// Converts a DeBERTa v3 safetensors checkpoint at `input` into a Vokra
/// GGUF at `output`. `license` overrides the upstream `mit` stamp (mirror
/// of the `convert_file --license <spdx>` boundary in `lib.rs`).
///
/// `tokenizer_bytes` optionally stamps the `vokra.bert.tokenizer.*` chunk
/// group `vokra_bert::tokenizer::SbertTokenizer::from_gguf` reads. The
/// bytes are treated as a JSON side-car produced by
/// `tools/parity/extract_spm_metadata.py` from an upstream `spm.model`
/// — the intermediate JSON keeps a SentencePiece protobuf parser out of
/// the runtime (NFR-DS-02 zero-dep: `vokra-convert` reuses
/// `vokra_core::json`, no `prost` / `protobuf` crate is added). Shape:
/// `{ "pieces": [str, ...], "scores": [f32, ...], "unk_id": u32,
/// "bos_id": u32, "eos_id": u32, "pad_id": u32 }`. Also stamps
/// `vokra.bert.tokenizer.kind = "sentencepiece-unigram"`. When
/// `tokenizer_bytes` is [`None`] no `vokra.bert.tokenizer.*` metadata
/// is written (SBV2 v2 loader-side `from_gguf` will then loud-fail —
/// that's FR-EX-08 by design).
///
/// # Errors
///
/// Same three arms as
/// [`convert_deberta_v2_file`](crate::models::deberta_v2::convert_deberta_v2_file):
/// [`ConvertError::Io`], [`ConvertError::Parse`] (I/O, malformed
/// safetensors, empty `tokenizer_bytes`, JSON parse or schema failure,
/// pieces/scores length disagreement, or no token-embedding-shaped
/// tensor found — `vocab_size` has no default in
/// `DebertaV3Encoder::from_gguf`, so this converter refuses to invent
/// one), [`ConvertError::Gguf`].
pub fn convert_deberta_v3_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    tokenizer_bytes: Option<&[u8]>,
) -> Result<ConvertReport, ConvertError> {
    // Front-load the tokenizer refuse — an empty JSON side-car would
    // parse to a top-level parse error anyway, but this gives a clearer
    // error message and never touches the safetensors input on failure.
    if let Some(t) = tokenizer_bytes
        && t.is_empty()
    {
        return Err(ConvertError::Parse(
            "deberta-v3 --tokenizer: file is empty — refusing to emit a zero-length \
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

    // Hparams — best-effort, checkpoint-shape-derived where possible
    // (shared helpers, see `deberta_v2`'s module doc "Hparams" section).
    // NOT independently verified against a real checkpoint; Task 30 fixup.
    let (vocab_size, d_model) = infer_vocab_and_d_model(&st)?;
    let n_layers = count_layers(&st);
    // Wave-4 DEBERTA-CONV-NAMES (2026-08-09): shape-derived n_pos_buckets
    // (same rationale as v2 — see `infer_n_pos_buckets` doc).
    let n_pos_buckets = infer_n_pos_buckets(&st);
    write_hparams(
        &mut b,
        "vokra.bert.deberta_v3",
        n_layers,
        d_model,
        vocab_size,
        n_pos_buckets,
    );

    // Tokenizer side-car — Blocker 5. Two mechanisms coexist:
    //
    // 1. HEAD (2026-08-06): explicit `tokenizer_bytes` = JSON produced by
    //    `tools/parity/extract_spm_metadata.py` (offline, uses upstream
    //    `sentencepiece` in a Python venv). See `write_tokenizer_spm_json`.
    //
    // 2. Wave 3 (2026-08-10): native sibling `spm.model` discovery via
    //    the hand-rolled proto3 reader (`crate::spm_proto`). One-step,
    //    no Python round-trip.
    //
    // Precedence: caller-supplied JSON wins (backward compat); when
    // absent, fall back to sibling-file discovery. When neither is
    // available, leave the metadata unwritten — the runtime
    // `SbertTokenizer::from_gguf` loud-errors on the missing `.pieces`
    // key which is the correct FR-EX-08 outcome.
    if let Some(bytes) = tokenizer_bytes {
        write_tokenizer_spm_json(&mut b, bytes)?;
    } else if let Some(spm_path) = discover_spm_model(input) {
        let tokenizer = parse_spm_model(&spm_path)?;
        write_tokenizer_metadata(&mut b, &tokenizer);
    }

    let mut report = ConvertReport::default();
    // Task 30 (2026-08-06): upstream HF names → `bert.*` names that
    // `DebertaV3Encoder::from_gguf` reads. The `map_deberta_name` helper is
    // shared with v2 (same `deberta.embeddings.*` / `deberta.encoder.layer.
    // <N>.*` naming convention), the only inference-relevant delta being v3's
    // "gradient-disentangled embedding sharing" (§3.1 arXiv:2111.09543) —
    // v3 reads `rel_embeddings` **once per encoder** and clones it into
    // every layer's `AttnWeights.pos_embed` at load time. We honor this by
    // emitting the shared table once under `bert.encoder.pos_embed.weight`
    // (not duplicated per layer, unlike v2's own convention).
    let mut skipped_names: Vec<(String, &'static str)> = Vec::new();
    let mut renamed_count = 0usize;
    let mut duplicated_count = 0usize;

    // DeBERTa-v3-large sets `norm_rel_ebd = "layer_norm"`. HF's
    // `DebertaV2Encoder.get_rel_embedding` therefore applies the
    // encoder-level LayerNorm to the shared relative-position table once
    // per forward before every layer consumes it. The runtime GGUF format
    // stores only the already-ready shared table, so perform that one-time
    // normalization here (same converter-side boundary as DeBERTa v2).
    let rel_embeddings = st
        .tensors()
        .iter()
        .find(|t| t.name == "deberta.encoder.rel_embeddings.weight")
        .map(|t| (t.dtype, t.shape.clone(), st.tensor_bytes(t).to_vec()));
    let rel_ln_gamma = st
        .tensors()
        .iter()
        .find(|t| t.name == "deberta.encoder.LayerNorm.weight")
        .map(|t| {
            (
                t.dtype,
                t.element_count() as usize,
                st.tensor_bytes(t).to_vec(),
            )
        });
    let rel_ln_beta = st
        .tensors()
        .iter()
        .find(|t| t.name == "deberta.encoder.LayerNorm.bias")
        .map(|t| {
            (
                t.dtype,
                t.element_count() as usize,
                st.tensor_bytes(t).to_vec(),
            )
        });
    if rel_ln_gamma.is_some() != rel_ln_beta.is_some() {
        let missing = if rel_ln_gamma.is_none() {
            "weight (gamma)"
        } else {
            "bias (beta)"
        };
        return Err(ConvertError::Parse(format!(
            "deberta.encoder.LayerNorm: partial pair — missing {missing} but the other \
             half is present. LN cannot pre-normalize rel_embeddings without both halves; \
             refusing rather than synthesizing a default (FR-EX-08)"
        )));
    }

    if let Some((dtype, shape, bytes)) = rel_embeddings {
        let (out_dtype, out_bytes) = match (&rel_ln_gamma, &rel_ln_beta) {
            (Some((g_dt, g_n, g_bytes)), Some((b_dt, b_n, b_bytes))) => {
                let n_elts = shape.iter().product::<u64>() as usize;
                let d = *shape.last().ok_or_else(|| {
                    ConvertError::Parse(
                        "deberta.encoder.rel_embeddings.weight: empty shape — cannot \
                         derive d_model for LN pre-normalization"
                            .to_owned(),
                    )
                })? as usize;
                if *g_n != d || *b_n != d {
                    return Err(ConvertError::Parse(format!(
                        "deberta.encoder.LayerNorm: gamma/beta element count ({} / {}) \
                         does not match rel_embeddings d_model ({d}); LN cannot be \
                         applied (FR-EX-08)",
                        g_n, b_n
                    )));
                }
                let rel_f32 = widen_bytes_to_f32(dtype, &bytes, n_elts)?;
                let gamma_f32 = widen_bytes_to_f32(*g_dt, g_bytes, *g_n)?;
                let beta_f32 = widen_bytes_to_f32(*b_dt, b_bytes, *b_n)?;
                let normed = apply_layer_norm_rows(&rel_f32, &gamma_f32, &beta_f32, d, 1e-7);
                (GgmlType::F32, f32_slice_to_le_bytes(&normed))
            }
            (None, None) => (dtype, bytes),
            _ => unreachable!("partial LayerNorm pair is refused above"),
        };
        b.add_tensor("bert.encoder.pos_embed.weight", out_dtype, shape, out_bytes)?;
        report.written += 1;
        renamed_count += 1;
        if out_dtype == GgmlType::BF16 {
            report.bf16_passthrough += 1;
        }
    }

    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => match map_deberta_name(&t.name) {
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
                    // These tensors were consumed by the out-of-band shared
                    // rel-embedding normalization above; do not misreport
                    // them as skipped.
                    if t.name != "deberta.encoder.rel_embeddings.weight"
                        && !t.name.starts_with("deberta.encoder.LayerNorm.")
                    {
                        let reason = classify_skip(&t.name);
                        skipped_names.push((t.name.clone(), reason));
                    }
                }
            },
            _ => report.skipped_non_float += 1,
        }
    }
    for (name, reason) in &skipped_names {
        eprintln!("convert_deberta_v3: skipping tensor `{name}` ({reason})");
    }
    eprintln!(
        "convert_deberta_v3: {} renamed (incl. 1 encoder-level rel_embeddings → pos_embed), {} duplicated (Q/K content ↔ position projection), {} skipped",
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

/// Parses a `tools/parity/extract_spm_metadata.py`-produced JSON side-car
/// and stamps the `vokra.bert.tokenizer.*` chunk group
/// [`SbertTokenizer::from_gguf`] reads. The Python-side extraction runs
/// upstream `sentencepiece.SentencePieceProcessor` (Apache-2.0 —
/// permissive) against the `spm.model` protobuf and dumps the four fields
/// this converter needs into a flat JSON object; keeping the protobuf
/// parser out of Rust is a deliberate NFR-DS-02 boundary (no `prost`
/// dependency, no `protobuf` crate — the runtime touches no protobuf, in
/// keeping with FR-LD-05).
///
/// Expected JSON shape (see the Python doc for the authoritative
/// spec):
///
/// ```json
/// {
///   "pieces":  [str, str, ...],  // id = array index
///   "scores":  [f32, f32, ...],  // same length as pieces
///   "unk_id":  int,              // e.g. 3
///   "bos_id":  int,              // e.g. 1
///   "eos_id":  int,              // e.g. 2
///   "pad_id":  int               // optional; when absent, PAD is not
///                                // reachable via a single id but the
///                                // reader still functions (SBV2 side
///                                // does not query pad_id today).
/// }
/// ```
///
/// # Errors
///
/// [`ConvertError::Parse`] when the JSON does not parse, the top-level
/// value is not an object, a required key is missing or the wrong type,
/// or `pieces.len() != scores.len()`.
pub(crate) fn write_tokenizer_spm_json(
    b: &mut GgufBuilder,
    bytes: &[u8],
) -> Result<(), ConvertError> {
    let root = json::parse(bytes).map_err(|e| {
        ConvertError::Parse(format!(
            "deberta-v3 --tokenizer: JSON parse failure: {e}. See \
             tools/parity/extract_spm_metadata.py --help for the expected schema."
        ))
    })?;

    let pieces_val = root.get("pieces").ok_or_else(|| {
        ConvertError::Parse("deberta-v3 --tokenizer: missing top-level `pieces` array".to_owned())
    })?;
    let pieces_arr = pieces_val.as_array().ok_or_else(|| {
        ConvertError::Parse("deberta-v3 --tokenizer: `pieces` must be an array".to_owned())
    })?;
    let pieces: Vec<String> = pieces_arr
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_str().map(str::to_owned).ok_or_else(|| {
                ConvertError::Parse(format!(
                    "deberta-v3 --tokenizer: `pieces[{i}]` is not a string"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let scores_val = root.get("scores").ok_or_else(|| {
        ConvertError::Parse("deberta-v3 --tokenizer: missing top-level `scores` array".to_owned())
    })?;
    let scores_arr = scores_val.as_array().ok_or_else(|| {
        ConvertError::Parse("deberta-v3 --tokenizer: `scores` must be an array".to_owned())
    })?;
    let scores: Vec<f32> = scores_arr
        .iter()
        .enumerate()
        .map(|(i, v)| match v {
            JsonValue::Float(f) => Ok(*f as f32),
            JsonValue::Int(n) => Ok(*n as f32),
            _ => Err(ConvertError::Parse(format!(
                "deberta-v3 --tokenizer: `scores[{i}]` is not a number"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;

    if pieces.len() != scores.len() {
        return Err(ConvertError::Parse(format!(
            "deberta-v3 --tokenizer: `pieces` length ({}) disagrees with `scores` length ({})",
            pieces.len(),
            scores.len()
        )));
    }
    if pieces.is_empty() {
        return Err(ConvertError::Parse(
            "deberta-v3 --tokenizer: `pieces` is empty".to_owned(),
        ));
    }

    let read_id = |key: &str| -> Result<u32, ConvertError> {
        root.get(key)
            .and_then(JsonValue::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| {
                ConvertError::Parse(format!(
                    "deberta-v3 --tokenizer: missing or non-u32 `{key}`"
                ))
            })
    };
    let unk_id = read_id("unk_id")?;
    let bos_id = read_id("bos_id")?;
    let eos_id = read_id("eos_id")?;
    // pad_id is optional — SPM `<pad>` may not exist in every checkpoint;
    // if omitted we stamp nothing (the reader does not query pad_id today).
    let pad_id: Option<u32> = root
        .get("pad_id")
        .and_then(JsonValue::as_u64)
        .and_then(|v| u32::try_from(v).ok());

    let vocab_size = u32::try_from(pieces.len()).map_err(|_| {
        ConvertError::Parse(format!(
            "deberta-v3 --tokenizer: `pieces.len()` ({}) exceeds u32::MAX",
            pieces.len()
        ))
    })?;

    b.add_string(
        &format!("{KEY_TOKENIZER_PREFIX}.kind"),
        KIND_SENTENCEPIECE_UNIGRAM,
    );
    add_string_array(b, &format!("{KEY_TOKENIZER_PREFIX}.pieces"), &pieces);
    add_f32_array(b, &format!("{KEY_TOKENIZER_PREFIX}.scores"), &scores);
    b.add_u32(&format!("{KEY_TOKENIZER_PREFIX}.unk_id"), unk_id);
    b.add_u32(&format!("{KEY_TOKENIZER_PREFIX}.bos_id"), bos_id);
    b.add_u32(&format!("{KEY_TOKENIZER_PREFIX}.eos_id"), eos_id);
    if let Some(p) = pad_id {
        b.add_u32(&format!("{KEY_TOKENIZER_PREFIX}.pad_id"), p);
    }
    b.add_u32(&format!("{KEY_TOKENIZER_PREFIX}.vocab_size"), vocab_size);
    Ok(())
}

/// Look for a SentencePiece `spm.model` alongside `input` — the
/// tokenizer file HF's DeBERTa v3 releases ship next to the safetensors
/// checkpoint. Returns `None` when it is not present at either
/// `<parent>/spm.model` or the nested `<parent>/tokenizer/spm.model`.
pub(crate) fn discover_spm_model(input: &Path) -> Option<std::path::PathBuf> {
    let parent = input.parent()?;
    let direct = parent.join("spm.model");
    if direct.exists() {
        return Some(direct);
    }
    let nested = parent.join("tokenizer").join("spm.model");
    if nested.exists() {
        return Some(nested);
    }
    None
}

/// Parse a SentencePiece `spm.model` at `path` into a shared
/// [`TokenizerMetadata`] with `scheme = "unigram"`. The special-token
/// ids are discovered by walking the parsed piece list for the
/// SentencePiece sentinels (`<unk>` / `<s>` / `</s>`), falling back to
/// the standard SentencePiece defaults (0 / 1 / 2) when absent so a
/// custom vocab that redefines the sentinels is honored.
///
/// # Errors
///
/// [`ConvertError::Io`] on read failure; [`ConvertError::Parse`] with
/// the underlying [`crate::spm_proto::SpmProtoError`] rendered as the
/// message on malformed proto3 input.
pub(crate) fn parse_spm_model(path: &Path) -> Result<TokenizerMetadata, ConvertError> {
    let bytes = std::fs::read(path)?;
    let model = parse_model(&bytes).map_err(|e| {
        ConvertError::Parse(format!("deberta-v3 spm.model at {}: {e}", path.display()))
    })?;

    let mut pieces = Vec::with_capacity(model.pieces.len());
    let mut scores = Vec::with_capacity(model.pieces.len());
    let mut unk_id: Option<u32> = None;
    let mut bos_id: Option<u32> = None;
    let mut eos_id: Option<u32> = None;
    for (i, p) in model.pieces.iter().enumerate() {
        // Discover sentinels by SentencePiece piece-type first (the
        // canonical signal), then by name as a fallback (some vocabs
        // ship `<unk>` as UserDefined rather than Unknown).
        let idx = i as u32;
        if matches!(p.piece_type, PieceType::Unknown) {
            unk_id.get_or_insert(idx);
        }
        match p.piece.as_str() {
            "<unk>" => {
                unk_id.get_or_insert(idx);
            }
            "<s>" => {
                bos_id.get_or_insert(idx);
            }
            "</s>" => {
                eos_id.get_or_insert(idx);
            }
            _ => {}
        }
        pieces.push(p.piece.clone());
        scores.push(p.score);
    }

    Ok(TokenizerMetadata {
        pieces,
        scores,
        unk_id: unk_id.unwrap_or(0),
        bos_id: bos_id.unwrap_or(1),
        eos_id: eos_id.unwrap_or(2),
        scheme: "unigram",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn safetensors_multi(entries: &[(&str, &str, &[u64], Vec<u8>)]) -> Vec<u8> {
        let mut body = Vec::new();
        let mut parts = Vec::new();
        let mut cursor = 0usize;
        for (name, dtype, shape, payload) in entries {
            let start = cursor;
            let end = start + payload.len();
            let shape = shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            parts.push(format!(
                r#""{name}":{{"dtype":"{dtype}","shape":[{shape}],"data_offsets":[{start},{end}]}}"#
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

    fn rel_ln_fixture(with_gamma: bool, with_beta: bool) -> (Vec<u8>, Vec<f32>) {
        let d_model = 4usize;
        let n_pos_buckets = 8usize;
        let embed_shape = [6u64, d_model as u64];
        let rel_shape = [n_pos_buckets as u64, d_model as u64];
        let ln_shape = [d_model as u64];
        let rel: Vec<f32> = (0..n_pos_buckets)
            .flat_map(|i| {
                (0..d_model).map(move |j| (i as f32 + 1.0) * 0.1 + (j as f32 + 1.0) * 0.01)
            })
            .collect();
        let mut entries = vec![
            (
                "deberta.embeddings.word_embeddings.weight",
                "F32",
                embed_shape.as_slice(),
                f32_bytes(&vec![0.01; 6 * d_model]),
            ),
            (
                "deberta.encoder.rel_embeddings.weight",
                "F32",
                rel_shape.as_slice(),
                f32_bytes(&rel),
            ),
        ];
        if with_gamma {
            entries.push((
                "deberta.encoder.LayerNorm.weight",
                "F32",
                ln_shape.as_slice(),
                f32_bytes(&[2.0; 4]),
            ));
        }
        if with_beta {
            entries.push((
                "deberta.encoder.LayerNorm.bias",
                "F32",
                ln_shape.as_slice(),
                f32_bytes(&[0.5; 4]),
            ));
        }
        (safetensors_multi(&entries), rel)
    }

    /// Single-tensor safetensors buffer — a shared `pos_embed` table plus
    /// the token-embedding table, enough to exercise hparam inference
    /// (vocab=5, d_model=4) without needing per-layer tensors (v3's own
    /// module doc explains why `count_layers`'s fallback-to-24 path is
    /// acceptable here: this file's job is proving the v3-specific
    /// metadata deltas, not re-proving `count_layers` itself, which
    /// `deberta_v2`'s test module already pins).
    fn v3_fixture() -> Vec<u8> {
        let embed = f32_bytes(&[0.01; 20]); // [5, 4]
        let header = format!(
            r#"{{"deberta.embeddings.word_embeddings.weight":{{"dtype":"F32","shape":[5,4],"data_offsets":[0,{}]}}}}"#,
            embed.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&embed);
        out
    }

    fn temp_pair(label: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let mut input = std::env::temp_dir();
        input.push(format!(
            "vokra-deberta-v3-{label}-{}-in.safetensors",
            std::process::id()
        ));
        let mut output = std::env::temp_dir();
        output.push(format!(
            "vokra-deberta-v3-{label}-{}-out.gguf",
            std::process::id()
        ));
        (input, output)
    }

    /// RED-phase pin: v3's arch tag, metadata-key prefix, and `mit` /
    /// `Permissive` default all differ from v2's — this is the delta this
    /// file exists to prove, given the tensor walk itself is identical to
    /// (and re-pinned by) `deberta_v2`'s test module.
    #[test]
    fn v3_defaults_and_prefix_differ_from_v2() {
        let blob = v3_fixture();
        let (input, output) = temp_pair("defaults");
        std::fs::write(&input, &blob).expect("write input safetensors");

        let report = convert_deberta_v3_file(&input, &output, None, None).expect("convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get("vokra.bert.deberta_v3.vocab_size")
                .and_then(|v| v.as_u64()),
            Some(5)
        );
        assert_eq!(
            file.get("vokra.bert.deberta_v3.d_model")
                .and_then(|v| v.as_u64()),
            Some(4)
        );
        // No `vokra.bert.deberta_v2.*` keys must leak into a v3 GGUF.
        assert!(file.get("vokra.bert.deberta_v2.vocab_size").is_none());

        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "mit must classify as Permissive, unlike v2's cc-by-sa-4.0 -> Copyleft default"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn shared_rel_embeddings_are_layer_normalized_when_pair_is_present() {
        let (blob, rel) = rel_ln_fixture(true, true);
        let (input, output) = temp_pair("rel-ln-present");
        std::fs::write(&input, &blob).expect("write input safetensors");

        convert_deberta_v3_file(&input, &output, None, None).expect("convert");
        let file = GgufFile::parse(std::fs::read(&output).expect("read gguf")).expect("parse gguf");
        let got = file
            .tensor_f32("bert.encoder.pos_embed.weight")
            .expect("shared normalized pos_embed");
        let expected = apply_layer_norm_rows(&rel, &[2.0; 4], &[0.5; 4], 4, 1e-7);
        assert_eq!(got.len(), expected.len());
        for (i, (&a, &b)) in got.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() < 1e-6, "pos_embed[{i}]={a}, expected={b}");
        }
        assert!(
            got.iter().zip(&rel).any(|(a, b)| (a - b).abs() > 1e-6),
            "non-trivial LayerNorm fixture must differ from raw rel_embeddings"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn shared_rel_embeddings_stay_raw_without_layer_norm_pair() {
        let (blob, rel) = rel_ln_fixture(false, false);
        let (input, output) = temp_pair("rel-ln-absent");
        std::fs::write(&input, &blob).expect("write input safetensors");

        convert_deberta_v3_file(&input, &output, None, None).expect("convert");
        let file = GgufFile::parse(std::fs::read(&output).expect("read gguf")).expect("parse gguf");
        assert_eq!(
            file.tensor_f32("bert.encoder.pos_embed.weight")
                .expect("shared raw pos_embed"),
            rel
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn partial_shared_rel_layer_norm_pair_is_a_loud_error() {
        let (blob, _) = rel_ln_fixture(true, false);
        let (input, output) = temp_pair("rel-ln-partial");
        std::fs::write(&input, &blob).expect("write input safetensors");

        let err = convert_deberta_v3_file(&input, &output, None, None)
            .expect_err("partial LayerNorm pair must fail");
        assert!(matches!(err, ConvertError::Parse(_)));
        assert!(err.to_string().contains("partial pair"));
        assert!(!output.exists(), "failed conversion must not leave a GGUF");

        std::fs::remove_file(&input).ok();
    }

    /// Blocker 5 (2026-08-06) — tokenizer_bytes = None emits no
    /// `vokra.bert.tokenizer.*` metadata (loader-side FR-EX-08).
    #[test]
    fn no_tokenizer_bytes_emits_no_tokenizer_chunk_v3() {
        let blob = v3_fixture();
        let (input, output) = temp_pair("no-tokenizer");
        std::fs::write(&input, &blob).expect("write input safetensors");

        convert_deberta_v3_file(&input, &output, None, None).expect("convert");

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

    /// Blocker 5 (2026-08-06) — tokenizer_bytes = Some(spm.json) stamps
    /// the full `vokra.bert.tokenizer.*` chunk group. The JSON is a
    /// hand-crafted minimal `tools/parity/extract_spm_metadata.py`-style
    /// dump; specials mirror the real DeBERTa v3 spm.model header
    /// (`[PAD]=0, [CLS]=1, [SEP]=2, [UNK]=3`).
    #[test]
    fn converter_stamps_spm_model_tokenizer_metadata() {
        // `\u{2581}` is `▁` (SentencePiece word-start) — must not appear
        // as a literal in a raw byte-string; a UTF-8 String is what the
        // JSON parser walks anyway.
        // Score `-3.125` is picked over `-3.14` deliberately: clippy's
        // `approx_constant` lint refuses `-3.14_f32` (mistakes it for PI)
        // — `-3.125` is representable exactly in f32 and equally
        // arbitrary. `-7.5` follows the same "exact-in-f32" choice.
        let spm_json = format!(
            r#"{{
            "pieces":  ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "{s}the", "{s}cat"],
            "scores":  [0.0, 0.0, 0.0, 0.0, -3.125, -7.5],
            "unk_id":  3,
            "bos_id":  1,
            "eos_id":  2,
            "pad_id":  0
        }}"#,
            s = "\u{2581}",
        );
        let spm_json = spm_json.as_bytes();
        let blob = v3_fixture();
        let (input, output) = temp_pair("tok-spm-json");
        std::fs::write(&input, &blob).expect("write input safetensors");

        convert_deberta_v3_file(&input, &output, None, Some(spm_json)).expect("convert");

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        assert_eq!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.kind"))
                .and_then(|v| v.as_str()),
            Some(KIND_SENTENCEPIECE_UNIGRAM),
            "v3 discriminator must be `sentencepiece-unigram`, not `bert-charsplit`"
        );
        let pieces = file
            .get(&format!("{KEY_TOKENIZER_PREFIX}.pieces"))
            .and_then(|v| v.as_array())
            .expect("pieces array present");
        assert_eq!(pieces.values.len(), 6);
        assert_eq!(pieces.values[0].as_str(), Some("[PAD]"));
        assert_eq!(pieces.values[4].as_str(), Some("▁the"));
        assert_eq!(pieces.values[5].as_str(), Some("▁cat"));

        let scores = file
            .get(&format!("{KEY_TOKENIZER_PREFIX}.scores"))
            .and_then(|v| v.as_array())
            .expect("scores array present");
        assert_eq!(scores.values.len(), 6);
        // The trained pieces carry non-zero log-probabilities from the
        // upstream Unigram model — v3 is decidedly NOT a bert-charsplit.
        if let Some(GgufMetadataValue::F32(f)) = scores.values.get(4) {
            assert_eq!(*f, -3.125_f32, "-3.125 preserved bit-exact through F32");
        } else {
            panic!("scores[4] should be F32");
        }

        assert_eq!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.unk_id"))
                .and_then(|v| v.as_u64()),
            Some(3)
        );
        assert_eq!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.bos_id"))
                .and_then(|v| v.as_u64()),
            Some(1)
        );
        assert_eq!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.eos_id"))
                .and_then(|v| v.as_u64()),
            Some(2)
        );
        assert_eq!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.pad_id"))
                .and_then(|v| v.as_u64()),
            Some(0)
        );
        assert_eq!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.vocab_size"))
                .and_then(|v| v.as_u64()),
            Some(6)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Blocker 5 (2026-08-06) — pad_id is optional (SentencePiece may
    /// not carry a PAD control piece). The other three specials are
    /// required.
    #[test]
    fn spm_json_pad_id_is_optional() {
        let spm_json = br#"{
            "pieces": ["[UNK]", "[CLS]", "[SEP]"],
            "scores": [0.0, 0.0, 0.0],
            "unk_id": 0,
            "bos_id": 1,
            "eos_id": 2
        }"#;
        let blob = v3_fixture();
        let (input, output) = temp_pair("tok-no-pad");
        std::fs::write(&input, &blob).expect("write input safetensors");
        convert_deberta_v3_file(&input, &output, None, Some(spm_json)).expect("convert");
        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");
        assert!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.pad_id"))
                .is_none(),
            "pad_id missing in JSON → not stamped in GGUF"
        );
        // The three required specials still made it in.
        assert_eq!(
            file.get(&format!("{KEY_TOKENIZER_PREFIX}.unk_id"))
                .and_then(|v| v.as_u64()),
            Some(0)
        );
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Blocker 5 (2026-08-06) — malformed JSON (top-level not an
    /// object, missing required key, wrong type) is a loud parse error
    /// (FR-EX-08).
    #[test]
    fn spm_json_missing_pieces_is_loud_error() {
        let bad = br#"{"scores": [0.0], "unk_id": 0, "bos_id": 1, "eos_id": 2}"#;
        let blob = v3_fixture();
        let (input, output) = temp_pair("tok-bad-no-pieces");
        std::fs::write(&input, &blob).expect("write input safetensors");
        let err = convert_deberta_v3_file(&input, &output, None, Some(bad))
            .expect_err("missing pieces must fail");
        assert!(matches!(err, ConvertError::Parse(_)));
        std::fs::remove_file(&input).ok();
    }

    #[test]
    fn spm_json_length_mismatch_is_loud_error() {
        let bad = br#"{"pieces":["a","b"], "scores":[0.0], "unk_id":0, "bos_id":1, "eos_id":2}"#;
        let blob = v3_fixture();
        let (input, output) = temp_pair("tok-bad-len");
        std::fs::write(&input, &blob).expect("write input safetensors");
        let err = convert_deberta_v3_file(&input, &output, None, Some(bad))
            .expect_err("pieces/scores length disagreement must fail");
        assert!(matches!(err, ConvertError::Parse(_)));
        std::fs::remove_file(&input).ok();
    }

    #[test]
    fn spm_json_empty_bytes_is_loud_error() {
        let blob = v3_fixture();
        let (input, output) = temp_pair("tok-empty");
        std::fs::write(&input, &blob).expect("write input safetensors");
        let err = convert_deberta_v3_file(&input, &output, None, Some(&[]))
            .expect_err("empty tokenizer must be refused");
        assert!(matches!(err, ConvertError::Parse(_)));
        std::fs::remove_file(&input).ok();
    }

    #[test]
    fn spm_json_invalid_json_is_loud_error() {
        let bad = br#"{not valid json"#;
        let blob = v3_fixture();
        let (input, output) = temp_pair("tok-bad-json");
        std::fs::write(&input, &blob).expect("write input safetensors");
        let err = convert_deberta_v3_file(&input, &output, None, Some(bad))
            .expect_err("invalid JSON must fail");
        assert!(matches!(err, ConvertError::Parse(_)));
        std::fs::remove_file(&input).ok();
    }
}
