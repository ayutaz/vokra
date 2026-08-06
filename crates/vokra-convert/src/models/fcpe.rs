//! **FCPE** — Fast Context-based Pitch Estimator: safetensors → GGUF
//! conversion (M5-16 / FR-OP-83).
//!
//! Upstream: `CNChTu/FCPE` (MIT). This converter is the offline
//! `.safetensors → Vokra GGUF` half; the upstream release is a
//! torch-pickle `.pt`, so callers pre-flatten it to safetensors via
//! `tools/parity/fcpe_prepare_checkpoint.py` (the DFN3 / DAC / CSM
//! bridge pattern — no pickle ever enters the runtime, FR-LD-05).
//!
//! # Category / arch / provenance
//!
//! - `vokra.model.arch = "fcpe"`
//! - `vokra.model.name = "fcpe"`
//! - `vokra.model.category = "f0"` — pitch / F0 extractor family
//!   (distinct taxonomy from `codec` / `tts` / `asr` / `s2s`; the
//!   runtime dispatches on `arch`, this is a catalog tag).
//! - `vokra.provenance.upstream_hf = "CNChTu/FCPE"` — GitHub-only
//!   release; the string preserves the CC-verified upstream anchor
//!   even though there is no HF mirror.
//!
//! # License posture — MIT (Permissive)
//!
//! Default `LicenseClass::Permissive` (SPDX `mit`), CC-verified on
//! 2026-07-30 against GitHub `CNChTu/FCPE` `LICENSE` (`docs/license-
//! audit.md` §3.1). Callers who ship the weights under a distinct
//! SPDX id (e.g. a fine-tune redistribution) can override at the outer
//! `convert_file --license <spdx>` boundary (the Whisper / kokoro /
//! xcodec2 pattern).
//!
//! # BF16 pass-through (mirror of `neucodec` / `xcodec2`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`) — the same posture as the sibling codec / TTS
//! converters. No convert-time widening; the runtime widens BF16 →
//! f32 losslessly via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Every F32 /
//! F16 tensor passes through under its upstream safetensors name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **prep-script's canonical output names**
//! (Vokra-defined for FCPE — the offline
//! `fcpe_prepare_checkpoint.py` remaps upstream `torchfcpe.model.
//! CFNaiveMelPE` state-dict keys to the flat layout the runtime binds).
//! The runtime layout is documented in
//! `crates/vokra-models/src/f0/fcpe.rs` module docs.
//!
//! # Wiring
//!
//! CLI dispatch (`vokra-cli convert --model fcpe`) resolves to this
//! module through [`crate::ModelKind::Fcpe`]; the file-based public
//! entry point is [`convert_fcpe_file`].

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for FCPE GGUFs. Distinct from every other F0
/// extractor family (`rmvpe` / `crepe`) — silently sharing an arch
/// would mis-route the runtime binder (they load different tensor
/// name sets + different topologies).
pub(crate) const ARCH: &str = "fcpe";
/// `vokra.model.name` value for the canonical FCPE GGUF.
pub(crate) const NAME: &str = "fcpe";

/// `vokra.model.category` value — FCPE is an **F0 / pitch extractor**
/// (FR-OP-83). Orthogonal to `arch` (the runtime dispatches on arch);
/// the category tag is a machine-readable catalog surface (see
/// `docs/license-audit.md` §3.1 for the tier-and-tag registry).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const MODEL_CATEGORY: &str = "f0";

/// Upstream release slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the
/// artifact back to its serving location without parsing the free-text
/// `vokra.provenance.source` string. FCPE ships on GitHub only — the
/// slug preserves the CC-verified upstream anchor even though there is
/// no HF mirror.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const UPSTREAM_HF: &str = "CNChTu/FCPE";

/// Default upstream weight license — `mit`, per the CNChTu/FCPE LICENSE
/// file (CC-verified 2026-07-30; sign-off 2026-07-30 yousan =
/// ☑ Commercial, `docs/license-audit.md` §3.1).
const DEFAULT_LICENSE_SPDX: &str = "mit";

/// Human-readable upstream source note stored in
/// `vokra.provenance.source`. Short — the license machine class is
/// carried separately in the `vokra.provenance.weight_license` chunk.
const UPSTREAM_SOURCE: &str = "CNChTu/FCPE (Fast Context-based Pitch Estimator, Conformer + 360-bin log-freq classifier, MIT)";

/// Outcome of an FCPE conversion (additive counters — a non-zero value
/// on any field is a positive report; a zero `written` value means the
/// input safetensors carried no float tensors and the runtime will
/// refuse to bind any weights, FR-EX-08).
#[derive(Debug, Default)]
pub struct FcpeReport {
    /// Total tensors observed in the input safetensors.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time, so any
    /// tensor reaching this counter would signal a reader change
    /// upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16
    /// (observability counter — the ADR pattern shared with neucodec /
    /// xcodec2 so a latent silent-widen cannot slip in undetected).
    pub bf16_passthrough: usize,
}

/// Internal shared conversion: reads a parsed safetensors buffer,
/// writes every F32 / F16 / BF16 tensor verbatim under its upstream
/// name, and stamps the `vokra.model.*` + `vokra.provenance.*`
/// metadata chunks.
///
/// The caller handles the `license` override at the outer boundary —
/// this function always stamps the built-in default (`mit`,
/// [`LicenseClass::Permissive`]). The [`crate::convert_file_licensed`]
/// outer wrapper re-stamps the `vokra.provenance.{license,
/// weight_license,source}` chunks when the caller supplied a non-
/// default SPDX id.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, FcpeReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Category / upstream-HF stamps — not covered by `stamp_provenance`
    // (which handles the SPDX + class + model_id + source group only),
    // so written directly. Consumers pick a decode path by category and
    // trace the artifact back to its serving location by upstream_hf.
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Built-in stamp = mit Permissive. The outer `convert_file_licensed`
    // layer overrides these three chunks if the caller passed a distinct
    // `--license <spdx>`.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        DEFAULT_LICENSE_SPDX,
        Some(NAME),
        Some(UPSTREAM_SOURCE),
    );

    let mut report = FcpeReport::default();
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
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
    Ok((b, report))
}

/// File-based FCPE converter (standalone entry — mirror of
/// `convert_neucodec_file` / `convert_xcodec2_file`).
///
/// Reads `input` (prep-script-flattened FCPE safetensors), writes a
/// Vokra GGUF to `output`. `license` overrides the default `mit`
/// provenance stamp (the Whisper / kokoro override pattern); pass
/// `None` to keep the built-in `mit` stamp.
///
/// # Errors
///
/// - [`ConvertError::Io`] if the input cannot be read or the output
///   cannot be written.
/// - [`ConvertError::Parse`] if the safetensors header is malformed.
/// - [`ConvertError::Gguf`] if the GGUF cannot be assembled.
pub fn convert_fcpe_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FcpeReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let (mut b, report) = convert(bytes)?;

    // Standalone-entry license override: mirror the outer
    // `convert_file_licensed` logic so a caller invoking this function
    // directly (bypassing `ModelKind` dispatch) still gets the same
    // license-override semantics.
    if let Some(spdx) = license.filter(|s| !s.is_empty()) {
        let class = LicenseClass::from_license_str(spdx);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, class.as_str());
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, spdx);
        b.add_string(
            chunks::KEY_PROVENANCE_SOURCE,
            &format!("upstream distribution source (licence {spdx} per source)"),
        );
    }

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    /// A unique temp path — per-process id **plus** a monotonic counter so
    /// two tests in the same process never race on the same file.
    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-fcpe-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    /// Encodes an f32 array as little-endian BF16 bytes (top 16 bits of
    /// the f32 pattern — the exact inverse of the runtime's
    /// `decode_bf16 : bits << 16`).
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Builds a synthetic single-tensor safetensors buffer with a
    /// caller-declared dtype and raw payload.
    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// A BF16 safetensors payload converts to a BF16 GGUF tensor with the
    /// canonical arch / name / category / provenance chunks stamped.
    #[test]
    fn convert_bf16_pass_through_stamps_metadata() {
        let values = vec![1.0f32, -2.0, 3.5, 0.25];
        let payload = bf16_bytes(&values);
        let st = safetensors_one("stem.weight", "BF16", &[2, 2], &payload);

        let in_path = tmp_path("bf16-in");
        let out_path = tmp_path("bf16-out");
        std::fs::write(&in_path, &st).unwrap();

        let report = convert_fcpe_file(&in_path, &out_path, None)
            .expect("well-formed BF16 checkpoint must convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.skipped_non_float, 0);

        let bytes = std::fs::read(&out_path).unwrap();
        let file = GgufFile::parse(bytes).expect("parse GGUF");
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
            Some(MODEL_CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );

        // BF16 tensor was preserved verbatim (top-16 f32 bits == our payload).
        let info = file.tensor_info("stem.weight").expect("tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions.iter().product::<u64>(), 4);
        let out_bytes = file.tensor_bytes(info);
        assert_eq!(out_bytes, &payload[..]);

        let _ = std::fs::remove_file(&in_path);
        let _ = std::fs::remove_file(&out_path);
    }

    /// A caller-supplied `--license` override rewrites the provenance
    /// chunks (the Whisper / kokoro / xcodec2 pattern) — the callable
    /// path preserves it without touching the tensor bytes.
    #[test]
    fn convert_honors_license_override() {
        let value = 1.0f32;
        let st = safetensors_one("head.weight", "F32", &[1], &value.to_le_bytes());

        let in_path = tmp_path("license-in");
        let out_path = tmp_path("license-out");
        std::fs::write(&in_path, &st).unwrap();
        let _ = convert_fcpe_file(&in_path, &out_path, Some("apache-2.0"))
            .expect("licence override must be accepted");

        let file = GgufFile::parse(std::fs::read(&out_path).unwrap()).unwrap();
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

        let _ = std::fs::remove_file(&in_path);
        let _ = std::fs::remove_file(&out_path);
    }
}
