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

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

use super::deberta_v2::{
    CATEGORY, ConvertReport, KEY_MODEL_CATEGORY, KEY_PROVENANCE_UPSTREAM_HF, count_layers,
    infer_vocab_and_d_model, write_hparams,
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
/// # Errors
///
/// Same three arms as
/// [`convert_deberta_v2_file`](crate::models::deberta_v2::convert_deberta_v2_file):
/// [`ConvertError::Io`], [`ConvertError::Parse`] (I/O, malformed
/// safetensors, or no token-embedding-shaped tensor found —
/// `vocab_size` has no default in `DebertaV3Encoder::from_gguf`, so this
/// converter refuses to invent one), [`ConvertError::Gguf`].
pub fn convert_deberta_v3_file(
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

    // Hparams — best-effort, checkpoint-shape-derived where possible
    // (shared helpers, see `deberta_v2`'s module doc "Hparams" section).
    // NOT independently verified against a real checkpoint; Task 30 fixup.
    let (vocab_size, d_model) = infer_vocab_and_d_model(&st)?;
    let n_layers = count_layers(&st);
    write_hparams(
        &mut b,
        "vokra.bert.deberta_v3",
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
                // `bert.*` name `DebertaV3Encoder::from_gguf` expects
                // (including resolving the shared `pos_embed` table to
                // `bert.encoder.pos_embed.weight`). Verbatim pass-through
                // until Task 30 confirms the real checkpoint's tensor-name
                // manifest.
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

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

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

        let report = convert_deberta_v3_file(&input, &output, None).expect("convert");
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
}
