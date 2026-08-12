#![allow(clippy::doc_lazy_continuation)]
//! **EAT** (`cwx-worst-one/EAT`, **mit**): safetensors → GGUF
//! conversion (SSL audio-encoder wave, 2026-08-13).
//!
//! Input: the upstream `cwx-worst-one/EAT` release — EAT
//! ("Effective Audio Transformer") is a self-supervised audio
//! encoder that combines an **utterance-level Transformer** with
//! **inverse block masking** and a self-distillation objective
//! (Chen et al. 2024, arXiv:2401.03497). Trained on AudioSet-2M
//! with MAE-style masked reconstruction, positioned as an
//! efficient alternative to BEATs / AST for downstream audio
//! tagging and general audio-embedding tasks. ~86M parameters
//! base variant (~350 MB PyTorch checkpoint).
//!
//! # Vokra scope — SSL audio encoder (2026-07-30 scope expansion)
//!
//! Sibling of `beats` (iterative-tokenizer SSL), `dasheng`
//! (universal MAE), `atst` (teacher-student patchout), `m2d`
//! (masked-modeling-duo). Distinct arch tag `eat` because the
//! utterance-level Transformer + inverse-block-masking topology
//! is a distinct axis from every sibling SSL encoder — silently
//! sharing would misroute the runtime dispatch and try to bind
//! e.g. a MAE decoder over an utterance-level checkpoint
//! (FR-EX-08). Category `audio-embedding`.
//!
//! # License posture — mit (**Permissive**)
//!
//! Upstream `github.com/cwx-worst-one/EAT` LICENSE reports
//! `spdx_id: MIT` via GitHub API `/repos/cwx-worst-one/EAT/license`
//! (task input 2026-08-13). No HuggingFace mirror exists as of
//! 2026-08-13 (search of `EAT` audio-tagged models returned no
//! matches beyond unrelated finetunes). §3.1 sign-off stays
//! blank fail-closed until owner completes primary-source
//! confirmation (memory `[[feedback-license-signoff-primary-source]]`
//! — no CC pre-fill).
//!
//! # Scale — local convert OK (~0.35 GB)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe). No vast.ai
//! handoff required.
//!
//! # No ONNX / no pickle (permanent)
//!
//! EAT ships as PyTorch `.pt` pickle from the upstream GitHub
//! release; this converter **never** touches ONNX or pickle
//! (FR-LD-05 / NFR-DS-02). Callers pre-flatten via a future
//! `tools/parity/eat_prepare_checkpoint.py` uv-managed Python
//! 3.12 sidecar (memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`) mirroring the DAC / Kokoro /
//! UTMOSv2 bridge pattern.
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm. BF16
//! is emitted as GGUF type 30 ([`GgmlType::BF16`]); the runtime
//! widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for EAT GGUFs. Distinct from sibling SSL
/// audio-encoder arch tags (`beats` / `dasheng` / `atst` / `m2d` /
/// `mert` / `muq`) — EAT's utterance-level Transformer +
/// inverse-block-masking training target is a distinct topology
/// axis from every sibling.
pub const ARCH: &str = "eat";

/// `vokra.model.name` — canonical `eat-base` size point.
/// Sibling `eat-large` release exists in the upstream releases
/// page but is a distinct arch variant published as its own
/// `NAME` following the snac_24khz / snac_44khz pattern (added
/// via a separate future ModelKind).
pub const NAME: &str = "eat-base";

/// `vokra.model.category` — general audio-embedding (sibling of
/// `dasheng` / `beats` / `atst` / `m2d`; downstream music-tagging
/// / audio-classification / sound-event heads feed from the
/// encoder's hidden states).
pub const CATEGORY: &str = "audio-embedding";

/// `vokra.provenance.upstream_url` value — the GitHub tree the
/// release ships from. EAT is not hosted on HuggingFace, so this
/// uses `upstream_url` rather than `upstream_hf`; the model-card
/// generator picks up either. Sibling of `nsnet2::UPSTREAM_URL` /
/// `emotion2vec` / `beats::UPSTREAM_URL` posture.
pub const UPSTREAM_URL: &str = "github.com/cwx-worst-one/EAT";

/// Default SPDX. Upstream `cwx-worst-one/EAT` LICENSE via GitHub
/// API `/repos/cwx-worst-one/EAT/license` returns
/// `spdx_id: MIT` (task input 2026-08-13). A caller with a
/// different attestation may override at the outer boundary
/// (`--license <spdx>`).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

const UPSTREAM_SOURCE: &str = "cwx-worst-one/EAT (Effective Audio Transformer, utterance-level Transformer + \
     inverse block masking self-supervised audio encoder, ~86M params base, Chen et al. \
     arXiv:2401.03497, mit)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Outcome of an EAT conversion. Mirrors the counter shape of the
/// sibling BF16 pass-through converters (`beats` / `dasheng` /
/// `mert` / `muq` / `yamnet`) — the invariant
/// `read == written + skipped_non_float` is auditable at the
/// report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EatReport {
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

/// Converts an EAT safetensors checkpoint at `input` into a
/// Vokra-native GGUF at `output`, returning an [`EatReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` chunks
/// are stamped for the runtime compliance gate (FR-CP-03).
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
pub fn convert_eat_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<EatReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    let spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(&mut b, class, spdx, Some(NAME), Some(UPSTREAM_SOURCE));

    let mut report = EatReport::default();
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
            "vokra-convert-eat-{tag}-{}-{n}",
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
    fn f32_tensor_passes_through_and_default_license_is_permissive() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // EAT uses a `patch_embed` conv + Transformer encoder blocks
        // — realistic upstream state-dict name from the utterance-level
        // Transformer body.
        let st = safetensors_one("patch_embed.proj.weight", "F32", &[1, 3], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_eat_file(&inp, &outp, None).expect("convert F32");
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
        assert_eq!(read_str(KEY_PROVENANCE_UPSTREAM_URL), UPSTREAM_URL);
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_LICENSE),
            DEFAULT_LICENSE_SPDX
        );
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            LicenseClass::Permissive.as_str(),
            "mit must resolve to Permissive"
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let values: [f32; 4] = [1.0, -0.5, 0.25, 8.0];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one("blocks.0.attn.qkv.weight", "BF16", &[2, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_eat_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("blocks.0.attn.qkv.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn license_override_swaps_stamp() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        convert_eat_file(&inp, &outp, Some("apache-2.0")).expect("convert with override");
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
