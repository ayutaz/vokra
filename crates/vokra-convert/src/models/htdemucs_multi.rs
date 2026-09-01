//! **HT-Demucs Multi** (`facebook/htdemucs` + `facebook/htdemucs_6s`,
//! MIT): fail-closed ensemble inspection contract.
//!
//! Input: the upstream Meta HT-Demucs release — Hybrid Transformer
//! Demucs (Rouard et al. 2023 ICASSP arXiv:2211.08553, "Hybrid
//! Transformers for Music Source Separation") — as either the 4-source
//! fine-tuned variant (`htdemucs_ft`: drums / bass / other / vocals) or
//! the 6-source variant (`htdemucs_6s`: 4 + piano + guitar). Both
//! variants share the same encoder / cross-domain transformer trunk
//! and differ only in the terminal output projection width. The source count
//! and member ordering are not inferred from a flattened tensor bag. The
//! upstream release ships torch pickles distributed via
//! `torch.hub` / `demucs.pretrained.get_model()`; this converter does not
//! flatten or accept those files. The VAST inspector first authenticates the
//! official ensemble contract without enabling an unsafe pickle fallback.
//!
//! Product GGUF output is disabled while the live member tensor manifests,
//! weight terms, and dependency/source review remain unauthenticated.
//! Inspection records member manifests, source hashes, and the official
//! ordering/weight matrix only. A historical flattened 2,132-tensor bag is
//! never treated as an ensemble.
//!
//! # License
//!
//! - SPDX: **MIT** ([`vokra_core::LicenseClass::Permissive`]).
//!   Verified against
//!   `github.com/facebookresearch/demucs/blob/main/LICENSE`.
//! - Category: **source-separation** (music-stems source separation —
//!   sibling of `mossformer2_ss_16k` under the shared source-separation
//!   umbrella covering both speech and music separation families;
//!   distinct from `demucs_htdemucs` which owns the base 4-stem
//!   `facebook/demucs` release).
//!
//! # Arch tag distinctness
//!
//! `vokra.model.arch = "htdemucs_multi"` is intentionally distinct
//! from every sibling source-separation arch tag (`demucs` = base
//! `facebook/demucs` 4-stem; `sepformer`; `mossformer2_ss_16k`;
//! `bs_roformer`; `tiger_separator`; `mp_senet`; `conv_tasnet`).
//! Silently sharing an arch tag with the base `demucs` ModelKind would
//! mis-route the runtime dispatch (HT-Demucs adds the cross-domain
//! transformer branch on top of the base U-Net, and the 6-stem output
//! projection width differs from the 4-stem base).
//!
//! # No ONNX (permanent)
//!
//! The upstream HT-Demucs release ships PyTorch pickle files; this
//! converter **never** touches ONNX (FR-LD-05).
//!
//! # Wiring status
//!
//! This module exposes the source-config structural contract so a later binder
//! cannot accidentally reorder members or flatten the ensemble. Product
//! conversion remains disabled until every member tensor manifest, weight
//! license/provenance, and dependency/source review is complete. The former
//! pass-through skeleton is intentionally not a runtime contract.
//!
//! A native Hybrid Transformer forward and product converter remain
//! follow-up work, deferred to owner sign-off (see
//! `docs/license-audit.md` §3.1).

use std::path::Path;

use crate::ConvertError;

/// Exact source member ordering in `htdemucs_ft.yaml`, as recorded by the
/// pinned upstream configuration. This is metadata only; it does not
/// authenticate or load any checkpoint bytes.
pub const HTDEMUCS_FT_MEMBER_IDS: &[&str] = &["f7e0c4bc", "d12395a8", "92cfc3b6", "04573f0d"];

/// Exact member ordering in `htdemucs_6s.yaml`, as recorded by the pinned
/// upstream configuration. The six-source model is one member, not a
/// flattened or four-member ensemble.
pub const HTDEMUCS_6S_MEMBER_IDS: &[&str] = &["5c90dfd2"];

/// The two source-config variants that are proven by the pinned upstream
/// files. No other member set is accepted by the structural contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtdemucsMultiVariant {
    /// Four fine-tuned members combined with the declared 4x4 identity matrix.
    FineTuned4,
    /// One six-source member with the derived 1x1 identity matrix.
    SixSource,
}

impl HtdemucsMultiVariant {
    /// Returns the exact upstream configuration filename.
    #[must_use]
    pub const fn config_filename(self) -> &'static str {
        match self {
            Self::FineTuned4 => "htdemucs_ft.yaml",
            Self::SixSource => "htdemucs_6s.yaml",
        }
    }

    /// Returns the exact member IDs in upstream order.
    #[must_use]
    pub const fn member_ids(self) -> &'static [&'static str] {
        match self {
            Self::FineTuned4 => HTDEMUCS_FT_MEMBER_IDS,
            Self::SixSource => HTDEMUCS_6S_MEMBER_IDS,
        }
    }

    /// Returns the source count declared by the upstream model variant.
    #[must_use]
    pub const fn source_count(self) -> usize {
        match self {
            Self::FineTuned4 => 4,
            Self::SixSource => 6,
        }
    }

    /// Returns the row-major ensemble matrix and its dimensions.
    ///
    /// The matrix values are copied from the pinned YAML configuration: the
    /// fine-tuned variant declares a 4x4 identity, while the single 6-source
    /// member derives a 1x1 identity. No averaging or implicit reweighting is
    /// allowed here.
    #[must_use]
    pub const fn ensemble_matrix(self) -> (&'static [f32], usize, usize) {
        const FT: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        const SIX: [f32; 1] = [1.0];
        match self {
            Self::FineTuned4 => (&FT, 4, 4),
            Self::SixSource => (&SIX, 1, 1),
        }
    }
}

/// Validates an ensemble member list and matrix against a pinned source
/// configuration. This is deliberately a structural check only: it does not
/// claim that checkpoint bytes, licenses, tensor roles, or numerical parity
/// have been authenticated.
pub fn validate_htdemucs_multi_structure(
    variant: HtdemucsMultiVariant,
    member_ids: &[&str],
    matrix: &[f32],
    matrix_rows: usize,
    matrix_columns: usize,
) -> Result<(), ConvertError> {
    if member_ids != variant.member_ids() {
        return Err(ConvertError::Usage(format!(
            "HT-Demucs multi {} member order differs from pinned upstream configuration",
            variant.config_filename()
        )));
    }
    let (expected, rows, columns) = variant.ensemble_matrix();
    if matrix_rows != rows || matrix_columns != columns || matrix != expected {
        return Err(ConvertError::Usage(format!(
            "HT-Demucs multi {} ensemble matrix differs from pinned upstream configuration",
            variant.config_filename()
        )));
    }
    Ok(())
}

/// Outcome of an HT-Demucs Multi conversion.
///
/// Mirrors the sibling BF16-passthrough converters' counter shape —
/// the invariant `read == written + skipped_non_float` is auditable at
/// the report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HtdemucsMultiReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16.
    pub bf16_passthrough: usize,
}

/// HT-Demucs multi conversion is intentionally disabled until VAST
/// authenticates the official ensemble configuration, member ordering,
/// source-weight matrix, and every member tensor manifest.
pub fn convert_htdemucs_multi_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<HtdemucsMultiReport, ConvertError> {
    Err(ConvertError::Usage(
        "HT-Demucs multi conversion is INSPECTION_ONLY until VAST authenticates the official ensemble config, member ordering, and tensor manifests".to_owned(),
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn htdemucs_multi_conversion_is_inspection_only() {
        let error = convert_htdemucs_multi_file(
            Path::new("/does/not/exist.safetensors"),
            Path::new("/tmp/htdemucs-multi.gguf"),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
        assert!(error.contains("ensemble"), "{error}");
    }

    #[test]
    fn source_configs_have_distinct_strict_member_contracts() {
        let (matrix, rows, columns) = HtdemucsMultiVariant::FineTuned4.ensemble_matrix();
        validate_htdemucs_multi_structure(
            HtdemucsMultiVariant::FineTuned4,
            HTDEMUCS_FT_MEMBER_IDS,
            matrix,
            rows,
            columns,
        )
        .expect("pinned fine-tuned contract");

        let (matrix, rows, columns) = HtdemucsMultiVariant::SixSource.ensemble_matrix();
        validate_htdemucs_multi_structure(
            HtdemucsMultiVariant::SixSource,
            HTDEMUCS_6S_MEMBER_IDS,
            matrix,
            rows,
            columns,
        )
        .expect("pinned six-source contract");
        assert_ne!(
            HtdemucsMultiVariant::FineTuned4.member_ids(),
            HtdemucsMultiVariant::SixSource.member_ids()
        );
        assert_eq!(HtdemucsMultiVariant::FineTuned4.source_count(), 4);
        assert_eq!(HtdemucsMultiVariant::SixSource.source_count(), 6);
    }

    #[test]
    fn structure_rejects_reordered_or_flattened_members() {
        let (matrix, rows, columns) = HtdemucsMultiVariant::FineTuned4.ensemble_matrix();
        let mut reordered = HTDEMUCS_FT_MEMBER_IDS.to_vec();
        reordered.swap(0, 1);
        assert!(
            validate_htdemucs_multi_structure(
                HtdemucsMultiVariant::FineTuned4,
                &reordered,
                matrix,
                rows,
                columns,
            )
            .is_err()
        );
        assert!(
            validate_htdemucs_multi_structure(
                HtdemucsMultiVariant::SixSource,
                &["flattened-2132-tensors"],
                &[1.0],
                1,
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn official_ensemble_contract_is_not_silently_flattened() {
        let error = convert_htdemucs_multi_file(
            Path::new("/does/not/exist.safetensors"),
            Path::new("/tmp/htdemucs-multi.gguf"),
            Some("mit"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
        assert!(error.contains("tensor manifests"), "{error}");
    }
}
