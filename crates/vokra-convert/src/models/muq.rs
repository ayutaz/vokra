#![allow(clippy::doc_lazy_continuation)]
//! **MuQ** (`OpenMuQ/MuQ-large-msd-iter`, **license unknown**):
//! safetensors → GGUF conversion (music-understanding wave,
//! 2026-08-13).
//!
//! Input: the upstream `OpenMuQ/MuQ-large-msd-iter` release — MuQ is
//! a self-supervised music representation learner that uses **Mel-
//! Residual Vector Quantization** targets and a **BEATs (Bidirectional
//! Encoder representation from Audio Transformers)** acoustic teacher
//! (Zhu et al. 2025, arXiv:2501.01108 "MuQ: Self-Supervised Music
//! Representation Learning with Mel Residual Vector Quantization").
//! Positioned as a direct MERT alternative — trained on the Million
//! Song Dataset with iterative refinement, ~500M-parameter class,
//! produces frame-level and utterance-level music embeddings for
//! downstream music-tagging / genre / MIR tasks.
//!
//! # Vokra scope — music understanding (2026-07-30 scope expansion)
//!
//! Sibling of `mert` (music-understanding, HuBERT-derived) and
//! `dasheng` (speech+music+environmental universal). Distinct arch
//! tag `muq` because the encoder training target (Mel-RVQ vs
//! MERT's residual-VQ-VAE token + CQT teacher, vs Dasheng's masked
//! ViT reconstruction) is a different topology axis — silently
//! sharing an arch tag would misroute the runtime dispatch and
//! e.g. try to bind an RVQ decoder over a HuBERT-derived checkpoint.
//! Category `music-embedding`.
//!
//! # License posture — **Unknown** (fail-closed)
//!
//! Upstream `OpenMuQ/MuQ-large-msd-iter` HF cardData does not
//! declare a `license:` tag as of task input 2026-08-13, and no
//! primary source has been CC-verified. Provenance stamp defaults
//! to [`LicenseClass::Unknown`] (fail-closed under M2-13). The
//! upstream repo may add a license later; a caller who has verified
//! a specific SPDX out-of-band can override at the outer boundary
//! via `--license <spdx>`. §3.1 sign-off stays blank fail-closed
//! until owner completes primary-source confirmation (owner ADR
//! on license resolution or Rejected posture).
//!
//! # Scale — local convert OK (~0.5 GB)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe). No vast.ai
//! handoff required.
//!
//! # No ONNX (permanent)
//!
//! MuQ ships as safetensors + PyTorch pickle; this converter
//! **never** touches ONNX (FR-LD-05). Callers pre-flatten to
//! safetensors offline via a future
//! `tools/parity/muq_prepare_checkpoint.py` if needed.
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm. BF16 is
//! emitted as GGUF type 30 ([`GgmlType::BF16`]); the runtime widens
//! BF16 → f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for MuQ GGUFs. Distinct from sibling
/// music-understanding arch tags (`mert` / `dasheng`) because the
/// Mel-RVQ + BEATs-teacher training target is a distinct topology
/// axis.
pub const ARCH: &str = "muq";

/// `vokra.model.name` — canonical `OpenMuQ/MuQ-large-msd-iter`
/// release.
pub const NAME: &str = "muq-large-msd-iter";

/// `vokra.model.category` — sibling of `mert` under
/// music-embedding.
pub const CATEGORY: &str = "music-embedding";

/// Upstream HF slug — recorded on `vokra.provenance.upstream_hf`.
pub const UPSTREAM_HF: &str = "OpenMuQ/MuQ-large-msd-iter";

/// Default SPDX. Upstream `OpenMuQ/MuQ-large-msd-iter` HF cardData
/// does not declare a `license:` tag as of task input 2026-08-13,
/// so the classifier `from_license_str("unknown")` correctly resolves
/// to [`LicenseClass::Unknown`] (fail-closed under M2-13). A caller
/// who has verified a specific SPDX out-of-band overrides via
/// `--license <spdx>` at the outer boundary.
pub const DEFAULT_LICENSE_SPDX: &str = "unknown";

const UPSTREAM_SOURCE: &str = "OpenMuQ/MuQ-large-msd-iter (self-supervised music representation learner, \
     Mel-Residual VQ + BEATs teacher, trained on Million Song Dataset with iterative \
     refinement, Zhu et al. arXiv:2501.01108, license unknown — fail-closed)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a MuQ conversion. Mirrors the counter shape of the
/// sibling BF16 pass-through converters (`mert` / `yamnet`) — the
/// invariant `read == written + skipped_non_float` is auditable at
/// the report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MuqReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16, so any tensor reaching
    /// this counter would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens BF16
    /// → f32 losslessly via the single choke point
    /// `vokra_core::gguf::quant::decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Converts an `OpenMuQ/MuQ-large-msd-iter` safetensors checkpoint at
/// `input` into a Vokra-native GGUF at `output`, returning a
/// [`MuqReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` chunks
/// are stamped for the runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX` (`"unknown"`,
/// `Unknown`) — fail-closed under M2-13, publish refused until a
/// caller supplies a real SPDX + §3.1 sign-off completes.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_muq_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MuqReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Unknown fail-closed default: if the caller passes no license
    // string, the classifier resolves `"unknown"` to `LicenseClass::Unknown`
    // and the M2-13 runtime gate refuses to load without a research
    // flag. Any caller who has resolved the license out-of-band supplies
    // `--license <spdx>` at the outer boundary.
    let spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(&mut b, class, spdx, Some(NAME), Some(UPSTREAM_SOURCE));

    let mut report = MuqReport::default();
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
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-muq-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

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

    #[test]
    fn f32_tensor_passes_through_and_default_license_is_unknown_fail_closed() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let st = safetensors_one("encoder.rvq.codebook.weight", "F32", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_muq_file(&inp, &outp, None).expect("convert F32");
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);

        let g = GgufFile::open(&outp).unwrap();
        let read_str = |k: &str| -> String {
            g.get(k)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{k}: missing"))
                .to_owned()
        };
        assert_eq!(read_str(chunks::KEY_MODEL_ARCH), ARCH);
        assert_eq!(read_str(chunks::KEY_MODEL_NAME), NAME);
        assert_eq!(read_str(KEY_MODEL_CATEGORY), CATEGORY);
        assert_eq!(read_str(KEY_PROVENANCE_UPSTREAM_HF), UPSTREAM_HF);
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_LICENSE),
            DEFAULT_LICENSE_SPDX
        );
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            LicenseClass::Unknown.as_str(),
            "unknown must resolve to Unknown (fail-closed under M2-13)"
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let values: [f32; 4] = [1.0, -2.5, 0.15625, 3.5];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one("encoder.blocks.0.attn.weight", "BF16", &[2, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_muq_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("encoder.blocks.0.attn.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn license_override_apache_flips_to_permissive() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        convert_muq_file(&inp, &outp, Some("apache-2.0")).expect("convert with override");

        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
        );
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
