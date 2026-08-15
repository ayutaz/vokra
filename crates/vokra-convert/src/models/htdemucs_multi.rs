//! **HT-Demucs Multi** (`facebook/htdemucs` + `facebook/htdemucs_6s`,
//! MIT): safetensors → GGUF conversion (coverage-audit-2026-08-03 Wave A
//! permissive continuation, 2026-08-04).
//!
//! Input: the upstream Meta HT-Demucs release — Hybrid Transformer
//! Demucs (Rouard et al. 2023 ICASSP arXiv:2211.08553, "Hybrid
//! Transformers for Music Source Separation") — as either the 4-source
//! fine-tuned variant (`htdemucs_ft`: drums / bass / other / vocals) or
//! the 6-source variant (`htdemucs_6s`: 4 + piano + guitar). Both
//! variants share the same encoder / cross-domain transformer trunk
//! and differ only in the terminal output projection width. This
//! converter is a **variant-agnostic BF16 pass-through skeleton** —
//! the source count rides in the tensor shapes verbatim so a single
//! ModelKind covers both without a variant enum. The upstream release
//! ships torch pickles distributed via `torch.hub` /
//! `demucs.pretrained.get_model()`; callers pre-flatten to safetensors
//! offline via a future `tools/parity/htdemucs_prepare_checkpoint.py`
//! (not yet written — the DFN3 / DAC / CSM pickle-bridge pattern, so no
//! pickle enters the runtime, FR-LD-05).
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`
//! and `vokra.provenance.*` metadata chunks the runtime source-
//! separation path binds against.
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
//! # BF16 pass-through (mirror of sensevoicesmall / neucodec /
//! # ecapa_tdnn / speaker_3d)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm on the same
//! match arm — no convert-time widening. BF16 is emitted as GGUF type
//! 30 ([`GgmlType::BF16`]); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream `demucs.htdemucs` state-dict
//! keys verbatim** (`encoder.*` / `decoder.*` / `tblocks.*` / `mask.*`
//! per the HT-Demucs class layout). Real-weight parity binding to a
//! future `vokra-models::htdemucs_multi` runtime module (native
//! Hybrid Transformer forward with source-count parametrization) is
//! deferred to owner sign-off per `docs/license-audit.md §3.1`.
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
//! This is the TDD skeleton (BF16 / F16 / F32 pass-through plus
//! provenance / category stamps). The runtime native Hybrid
//! Transformer + cross-domain attention forward is a follow-up wave,
//! deferred to owner sign-off (see `docs/license-audit.md` §3.1).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for HT-Demucs Multi GGUFs. Intentionally
/// distinct from every sibling source-separation arch tag (`demucs`,
/// `sepformer`, `mossformer2_ss_16k`, `bs_roformer`, `tiger_separator`,
/// `mp_senet`, `conv_tasnet`) — HT-Demucs adds the cross-domain
/// transformer on top of the U-Net trunk and the 4-vs-6 source variants
/// differ in output projection width, so silently sharing an arch tag
/// with the base `demucs` ModelKind would mis-route the runtime
/// dispatch.
pub const ARCH: &str = "htdemucs_multi";

/// `vokra.model.name` value written for the canonical HT-Demucs
/// multi-variant family.
pub const NAME: &str = "htdemucs_multi";

/// `vokra.model.category` value written for every HT-Demucs Multi
/// GGUF. Sibling of `mossformer2_ss_16k` / `sepformer` /
/// `bs_roformer` (`source-separation` umbrella covering both music
/// and speech separation families).
pub const CATEGORY: &str = "source-separation";

/// Upstream HF repository slug (`org/name`) — canonical HF mirror of
/// the HT-Demucs family. Both 4-source and 6-source variants publish
/// under the same `facebook/htdemucs` prefix (the 6-source is
/// `facebook/htdemucs_6s`); the canonical slug stamped here is the
/// 4-source root, and the alias table admits both.
pub const UPSTREAM_HF: &str = "facebook/htdemucs";

/// Default upstream weight licence (SPDX). Verified against
/// `github.com/facebookresearch/demucs/blob/main/LICENSE` (standard
/// MIT).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// convention the sibling BF16-passthrough converters use applies).

/// `vokra.model.category` metadata key. Local per the established
/// sensevoicesmall / nkf_aec / funcodec convention.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` metadata key — the primary
/// redistribution source HF slug. Local per the same convention as
/// [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

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

/// Converts an HT-Demucs Multi safetensors checkpoint at `input`
/// (pre-flattened from either `htdemucs_ft` 4-source or `htdemucs_6s`
/// 6-source torch pickle — a future
/// `tools/parity/htdemucs_prepare_checkpoint.py` is not yet written, so
/// that flattening is an owner-side step today) into a Vokra-native
/// GGUF at `output`, returning an [`HtdemucsMultiReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict key; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` chunks are stamped for the runtime compliance
/// gate (FR-CP-03). The source-count axis (4 vs 6) rides in the
/// terminal projection tensor shapes verbatim — a single ModelKind
/// covers both without a variant enum.
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX` (`"mit"`,
/// `Permissive`).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_htdemucs_multi_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<HtdemucsMultiReport, ConvertError> {
    // Load the whole checkpoint into memory — each HT-Demucs variant is
    // ~86 MB (well below the streaming-mandated Moshi 14 GiB tier).
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let effective_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some(
            "facebook/htdemucs family (Hybrid Transformer Demucs music source separation, \
             4-source htdemucs_ft or 6-source htdemucs_6s variant, Rouard et al. 2023 \
             arXiv:2211.08553, MIT)",
        ),
    );

    let mut report = HtdemucsMultiReport::default();
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
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
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-htdemucs-multi-{tag}-{}-{}.{ext}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        p
    }

    struct TempFileGuard(PathBuf);
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Pins the BF16 pass-through end-to-end.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload = bf16_bytes(&values);
        assert_eq!(payload.len(), 12);
        // HT-Demucs cross-domain transformer attention Q-projection
        // weight — the upstream `demucs.htdemucs` state-dict key
        // convention preserved verbatim through the
        // `htdemucs_prepare_checkpoint.py` bridge.
        let header = r#"{"encoder.tblocks.0.attn.wq.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report =
            convert_htdemucs_multi_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("encoder.tblocks.0.attn.wq.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());
    }

    /// Pins F32 and F16 pass-through. MIT default → Permissive.
    #[test]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_permissive() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();

        let header = format!(
            r#"{{"encoder.layers.0.norm.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"decoder.mask.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
        );
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16_bytes);

        let input_path = scratch_path("mixed-in", "safetensors");
        let output_path = scratch_path("mixed-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_htdemucs_multi_file(&input_path, &output_path, None)
            .expect("convert F32 + F16 mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let f32_info = file
            .tensor_info("encoder.layers.0.norm.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file.tensor_info("decoder.mask.bias").expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
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
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "mit must resolve to Permissive (T1 tier)"
        );
    }

    /// Pins the license override boundary.
    #[test]
    fn license_override_replaces_default() {
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = r#"{"encoder.embed.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("lic-in", "safetensors");
        let output_path = scratch_path("lic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_htdemucs_multi_file(&input_path, &output_path, Some("apache-2.0"))
            .expect("convert with override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );
    }
}
