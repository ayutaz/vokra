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
    // Float tensors pass through **verbatim** — no convert-time widening,
    // no renaming (see module doc "TODO(owner)" section).
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                // TODO(owner): map this upstream HF tensor name to the
                // `bert.*` name `DebertaV2Encoder::from_gguf` expects.
                // Verbatim pass-through until Task 30 confirms the real
                // checkpoint's tensor-name manifest.
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => report.skipped_non_float += 1,
        }
    }

    let spdx = license.unwrap_or(DEFAULT_LICENSE);
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(&mut b, class, spdx, Some(NAME), Some(UPSTREAM_HF));

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;

    Ok(report)
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
        assert_eq!(report.read, 4);
        assert_eq!(report.written, 4, "F32 and BF16 both pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1, "exactly one BF16 tensor");

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

        let bf16_info = file
            .tensor_info("deberta.encoder.layer.2.attention.self.query_proj.weight")
            .expect("BF16 tensor present under its verbatim upstream name");
        assert_eq!(
            bf16_info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16 (type 30)"
        );

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
