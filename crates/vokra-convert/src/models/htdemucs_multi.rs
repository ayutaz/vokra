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
//! Product GGUF output is disabled while the live 2,132-tensor bag's ensemble
//! contract is unauthenticated. Inspection records member manifests, source
//! hashes, and the official ordering/weight matrix only.
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
//! This is an inspection-only boundary; product conversion is disabled until
//! the exact ensemble manifest is audited. The former pass-through skeleton
//! is intentionally not a runtime contract.
//!
//! A native Hybrid Transformer forward and product converter remain
//! follow-up work, deferred to owner sign-off (see
//! `docs/license-audit.md` §3.1).

use std::path::Path;

use crate::ConvertError;

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
