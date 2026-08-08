//! **DeBERTa v3** (`microsoft/deberta-v3-large`): safetensors → GGUF
//! conversion (SBV2 v2 plan Task 11, 2026-07-26).
//!
//! Structurally this mirrors [`crate::models::deberta_v2`] byte-for-byte —
//! same BF16-pass-through posture, same shape-derived hparam inference,
//! same verbatim tensor-name pass-through, same `ConvertReport` shape
//! (imported from `deberta_v2`, not redefined — see that module's doc for
//! why one shared type serves both). The only real deltas are the ones
//! DeBERTa v3 itself introduces at inference time: a distinct arch tag
//! (`deberta_v3`), default license (`mit`, not `cc-by-sa-4.0`), upstream
//! repository, and `vokra.bert.deberta_v3.*` metadata-key prefix. v3's one
//! architecturally-relevant change vs v2 — a single shared position
//! embedding table instead of one per layer (§3.1, "gradient-disentangled
//! embedding sharing") — does not affect *this* converter at all, since
//! tensor renaming (which is where that distinction would matter) is
//! deferred to Task 30 (see `deberta_v2`'s module doc "TODO(owner)"
//! section) — this file only proves the metadata/BF16/provenance contract
//! independently for v3's own defaults.
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
//! # TODO(owner): tensor name mapping — see `deberta_v2`'s module doc
//!
//! Same unresolved question, same Task 30 dependency: every tensor here
//! passes through under its verbatim upstream HF name (no `wq_pos`
//! shared/per-layer resolution attempted).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};
use vokra_core::json::{self, JsonValue};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

use super::deberta_v2::{
    CATEGORY, ConvertReport, KEY_MODEL_CATEGORY, KEY_PROVENANCE_UPSTREAM_HF, KEY_TOKENIZER_PREFIX,
    KIND_SENTENCEPIECE_UNIGRAM, MapAction, add_f32_array, add_string_array, classify_skip,
    count_layers, infer_n_pos_buckets, infer_vocab_and_d_model, map_deberta_name, write_hparams,
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
/// group [`vokra_bert::tokenizer::SbertTokenizer::from_gguf`] reads. The
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

    // Tokenizer side-car — Blocker 5 (2026-08-06). See `write_tokenizer_spm_json`.
    if let Some(bytes) = tokenizer_bytes {
        write_tokenizer_spm_json(&mut b, bytes)?;
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

    // Emit the shared rel_embeddings under v3's expected name if present.
    for t in st.tensors() {
        if t.name == "deberta.encoder.rel_embeddings.weight" {
            b.add_tensor(
                "bert.encoder.pos_embed.weight",
                t.dtype,
                t.shape.clone(),
                st.tensor_bytes(t).to_vec(),
            )?;
            report.written += 1;
            renamed_count += 1;
            if t.dtype == GgmlType::BF16 {
                report.bf16_passthrough += 1;
            }
            break;
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
                    let reason = classify_skip(&t.name);
                    skipped_names.push((t.name.clone(), reason));
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

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
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
