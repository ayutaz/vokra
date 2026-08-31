//! SGMSE-VoiceBank inspection-only conversion boundary.
//!
//! The public 647-tensor artifact and arbitrary safetensors are not accepted
//! as a runtime checkpoint. A VAST inspection must first authenticate the
//! fixed SpeechBrain checkpoint, safe-load container, EMA extraction, full
//! manifest, and upstream implementation contract.

use std::collections::BTreeSet;
use std::path::Path;

use crate::ConvertError;

/// Compatibility report retained for callers while product conversion is
/// disabled. No instance is returned until the inspection gate is cleared.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SgmseReport {
    /// Number of tensors observed by a future authenticated converter.
    pub read: usize,
    /// Number of floating-point tensors written by a future authenticated converter.
    pub written: usize,
    /// Number of non-floating-point tensors skipped by a future authenticated converter.
    pub skipped_non_float: usize,
    /// Number of BF16 tensors preserved by a future authenticated converter.
    pub bf16_passthrough: usize,
}

/// Dependency-free representation of one row in the checkpoint-specific
/// SGMSE contract.  The converter does not infer names: an authenticated VAST
/// inspection must provide these exact rows and the required-role set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SgmseManifestRow {
    pub name: String,
    pub role: String,
    pub dtype_tag: u32,
    pub dimensions: Vec<u64>,
}

/// Validate the typed role/name/shape declaration before a future writer is
/// allowed to emit GGUF. This deliberately accepts no catch-all/pass-through
/// role and does not coerce unsupported dtypes.
pub fn validate_typed_manifest(
    rows: &[SgmseManifestRow],
    required_roles: &[String],
) -> Result<(), String> {
    let expected: BTreeSet<_> = required_roles.iter().map(String::as_str).collect();
    if expected.is_empty() || expected.len() != required_roles.len() || rows.len() != expected.len()
    {
        return Err(
            "sgmse: typed manifest required-role set is empty, duplicate, or incomplete".to_owned(),
        );
    }
    let mut names = BTreeSet::new();
    let mut roles = BTreeSet::new();
    for row in rows {
        if row.name.is_empty()
            || row
                .name
                .chars()
                .any(|character| character == '|' || character.is_control())
            || !names.insert(row.name.as_str())
            || !roles.insert(row.role.as_str())
            || !expected.contains(row.role.as_str())
            || !valid_role(&row.role)
            || !matches!(row.dtype_tag, 0 | 1 | 30)
            || row.dimensions.is_empty()
            || row.dimensions.contains(&0)
        {
            return Err(
                "sgmse: typed manifest has duplicate, unknown, or unsupported row".to_owned(),
            );
        }
    }
    if roles != expected {
        return Err("sgmse: typed manifest is missing or has extra roles".to_owned());
    }
    Ok(())
}

fn valid_role(role: &str) -> bool {
    matches!(
        role,
        "fourier_frequencies"
            | "sigma_first_projection"
            | "sigma_first_bias"
            | "sigma_second_projection"
            | "sigma_second_bias"
    ) || role
        .strip_prefix("stage:")
        .and_then(|rest| {
            let mut fields = rest.split(':');
            let _index = fields.next()?.parse::<usize>().ok()?;
            let kind = fields.next()?;
            let _block = fields.next()?.parse::<usize>().ok()?;
            let slot = fields.next()?;
            if fields.next().is_some() || kind.is_empty() || slot.is_empty() {
                return None;
            }
            let valid_kind = matches!(
                kind,
                "input"
                    | "residual"
                    | "attention"
                    | "downsample"
                    | "upsample"
                    | "progressive_output"
                    | "progressive_input"
                    | "middle"
                    | "output"
            );
            let valid_slot = matches!(
                slot,
                "weight"
                    | "bias"
                    | "norm_gamma"
                    | "norm_beta"
                    | "time_embedding"
                    | "query"
                    | "key"
                    | "value"
                    | "output"
                    | "fir_kernel"
            );
            if valid_kind && valid_slot {
                Some(())
            } else {
                None
            }
        })
        .is_some()
}

/// Reject every candidate until the fixed upstream checkpoint and complete
/// tensor contract have been authenticated on VAST. In particular, this
/// prevents empty inputs and permissive license relabels from producing a
/// product-facing GGUF.
pub fn convert_sgmse_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<SgmseReport, ConvertError> {
    Err(ConvertError::Usage(
        "SGMSE-VoiceBank conversion is AUTHENTICATED_MANIFEST_REQUIRED: VAST must authenticate the fixed checkpoint, safe-load container, EMA extraction, typed role/name/dtype/shape manifest, and complete NCSN++ assignment before conversion".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_checkpoint_requires_authenticated_manifest() {
        let error = convert_sgmse_file(
            Path::new("/does/not/exist.safetensors"),
            Path::new("/tmp/sgmse-voicebank.gguf"),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("AUTHENTICATED_MANIFEST_REQUIRED"), "{error}");
        assert!(
            error.contains("typed role/name/dtype/shape manifest")
                && error.contains("complete NCSN++ assignment"),
            "{error}"
        );
    }

    #[test]
    fn permissive_relabel_cannot_bypass_checkpoint_gate() {
        let error = convert_sgmse_file(
            Path::new("/does/not/exist.safetensors"),
            Path::new("/tmp/sgmse-voicebank.gguf"),
            Some("mit"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("AUTHENTICATED_MANIFEST_REQUIRED"), "{error}");
    }

    #[test]
    fn typed_manifest_rejects_missing_extra_and_catch_all_roles() {
        let required = vec!["fourier_frequencies".to_owned()];
        let valid = vec![SgmseManifestRow {
            name: "source.exact.frequencies".to_owned(),
            role: required[0].clone(),
            dtype_tag: 0,
            dimensions: vec![128],
        }];
        validate_typed_manifest(&valid, &required).unwrap();

        let mut extra = valid.clone();
        extra[0].role = "arbitrary_passthrough".to_owned();
        assert!(validate_typed_manifest(&extra, &required).is_err());
        assert!(validate_typed_manifest(&[], &required).is_err());
    }
}
