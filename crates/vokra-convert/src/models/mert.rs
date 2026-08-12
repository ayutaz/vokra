#![allow(clippy::doc_lazy_continuation)]
//! **MERT-v1-330M** (`m-a-p/MERT-v1-330M`, **cc-by-nc-4.0**):
//! safetensors → GGUF conversion (music-understanding wave,
//! 2026-08-13).
//!
//! Input: the upstream `m-a-p/MERT-v1-330M` release — Music
//! undERstanding model with large-scale self-supervised Training
//! (Li et al. 2023, arXiv:2306.00107). MERT is an audio-MPM
//! (masked-prediction) self-supervised music encoder built on the
//! HuBERT architecture (Conv1D feature extractor + 24-layer
//! Transformer, ~330M params, 24 kHz mono waveform in) with a music-
//! specific reconstruction target (residual VQ-VAE token) and
//! acoustic teacher (CQT). Serves as the canonical
//! music-understanding embedding target — its 24-layer hidden states
//! feed downstream music-tagging, music-similarity, and cover-song-
//! identification tasks (SOTA on MIREX-benchmark tasks in 2023).
//!
//! # Vokra scope — music understanding (2026-07-30 scope expansion)
//!
//! First member of the music-understanding embedding family under the
//! `music-embedding` category. Sibling of `muq` (music-understanding
//! alternative), `dasheng` (speech+music+environmental universal
//! encoder), and downstream tag classifiers like `panns` / `yamnet`.
//! Distinct arch tag `mert` from every sibling because the encoder
//! topology (HuBERT-derived Conv1D + Transformer + music-specific
//! reconstruction heads) differs from Dasheng (masked-autoencoder
//! ConvNeXt/ViT) and MuQ (Mel-Residual VQ + BEATs teacher). Silently
//! sharing an arch tag would misroute the runtime dispatch and route
//! e.g. an MPM checkpoint through a MAE loader.
//!
//! # License posture — CC-BY-NC 4.0 (**NonCommercial**)
//!
//! Upstream HF cardData `license: cc-by-nc-4.0` (m-a-p/MERT-v1-330M
//! model card, primary source referenced by task input 2026-08-13).
//! Same T4 tier as X-Codec 2 (2026-07-28) / MusicGen family
//! (2026-08-01) — `publish-one.sh --allow-noncommercial` gate + the
//! M2-13 runtime research-flag gate refuse to load in commercial
//! mode unless overridden via `--license <spdx>`. §3.1 sign-off stays
//! blank fail-closed until owner completes primary-source
//! confirmation (owner ADR on Research-only tier acceptance).
//!
//! # Scale — local convert OK (~0.3 GB / ~330M params)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe). No vast.ai
//! handoff required for conversion.
//!
//! # No ONNX (permanent)
//!
//! MERT ships as safetensors + PyTorch pickle; this converter
//! **never** touches ONNX (FR-LD-05). Callers who receive the
//! upstream `pytorch_model.bin` pickle pre-flatten to safetensors
//! offline via a future `tools/parity/mert_prepare_checkpoint.py`
//! (the DAC / Kokoro / UTMOSv2 bridge pattern — no pickle in the
//! runtime, NFR-DS-02).
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

/// `vokra.model.arch` for MERT GGUFs. Distinct from sibling
/// music-understanding arch tags (`muq` / `dasheng`) — MERT's
/// HuBERT-derived encoder + music-specific reconstruction heads
/// (residual VQ-VAE token + CQT teacher) is a distinct topology
/// from Dasheng (MAE ConvNeXt/ViT) and MuQ (Mel-RVQ + BEATs).
pub const ARCH: &str = "mert";

/// `vokra.model.name` — canonical `m-a-p/MERT-v1-330M` release.
pub const NAME: &str = "mert-v1-330m";

/// `vokra.model.category` — music understanding embedding (24-layer
/// hidden-state consumer surface for downstream music tagging /
/// similarity / cover-song ID).
pub const CATEGORY: &str = "music-embedding";

/// Upstream HF slug — recorded on `vokra.provenance.upstream_hf`.
pub const UPSTREAM_HF: &str = "m-a-p/MERT-v1-330M";

/// Default SPDX. HF cardData primary source `license: cc-by-nc-4.0`
/// (music-understanding wave task input, 2026-08-13). A caller with a
/// different attestation may override at the outer boundary
/// (`--license <spdx>`); the M2-13 runtime gate then reclassifies.
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-4.0";

const UPSTREAM_SOURCE: &str = "m-a-p/MERT-v1-330M (Music undERstanding model with large-scale self-supervised \
     Training — HuBERT-derived Conv1D + 24-layer Transformer + RVQ-VAE token / CQT \
     teacher, ~330M params, Li et al. arXiv:2306.00107, cc-by-nc-4.0)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of an MERT conversion. Mirrors the counter shape of the
/// sibling BF16 pass-through converters (`yamnet` / `utmosv2`) — the
/// invariant `read == written + skipped_non_float` is auditable at the
/// report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MertReport {
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

/// Converts an `m-a-p/MERT-v1-330M` safetensors checkpoint at `input`
/// into a Vokra-native GGUF at `output`, returning a [`MertReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` chunks
/// are stamped for the runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX` (`"cc-by-nc-4.0"`,
/// `NonCommercial`) — publish requires
/// `publish-one.sh --allow-noncommercial` per the X-Codec 2 T4
/// precedent.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_mert_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MertReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // NonCommercial fail-closed default per audiogen_medium / musicgen_medium
    // pattern: unless the caller supplies an explicit SPDX, force the
    // classifier to `NonCommercial` (the classifier `from_license_str` also
    // resolves `cc-by-nc-4.0` to `NonCommercial` — the explicit tuple below
    // is a defense-in-depth pin so a future classifier refactor cannot
    // accidentally weaken the default posture).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::NonCommercial),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_SOURCE));

    let mut report = MertReport::default();
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
            "vokra-convert-mert-{tag}-{}-{n}",
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
    fn f32_tensor_passes_through_and_default_license_is_noncommercial() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Realistic MERT encoder key — HuBERT-style prefix.
        let st = safetensors_one(
            "encoder.layers.0.self_attn.q_proj.weight",
            "F32",
            &[2, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        let r = convert_mert_file(&inp, &outp, None).expect("convert F32");
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
            LicenseClass::NonCommercial.as_str(),
            "cc-by-nc-4.0 must resolve to NonCommercial (T4 tier, fail-closed)"
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
        let st = safetensors_one("encoder.layers.1.fc1.weight", "BF16", &[2, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_mert_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.written, 1);
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("encoder.layers.1.fc1.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn license_override_to_permissive_flips_class() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        // A caller with a different license attestation escapes the
        // NonCommercial default (mirror of the audiogen_medium /
        // musicgen_medium escape hatch).
        convert_mert_file(&inp, &outp, Some("apache-2.0")).expect("convert with override");

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
