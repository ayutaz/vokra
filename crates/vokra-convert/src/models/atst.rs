#![allow(clippy::doc_lazy_continuation)]
//! **ATST** (`Audio-WestlakeU/audiossl/tree/main/audiossl/methods/atst`,
//! **cc-by-4.0 weight** / mit code): safetensors → GGUF conversion
//! (SSL audio-encoder wave, 2026-08-13).
//!
//! Input: the upstream `Audio-WestlakeU/audiossl` release — ATST
//! ("Audio Teacher-Student Transformer") is a self-supervised
//! audio encoder trained via a **BYOL-style EMA teacher +
//! student patchout** objective over log-mel spectrogram
//! (Li et al. 2022, INTERSPEECH, arXiv:2204.12076; frame-level
//! extension "atstframe" Li et al. 2023, TASLP, arXiv:2306.04186).
//! Positioned as an efficient audio-embedding backbone for
//! downstream sound-event detection / audio-tagging / speaker
//! tasks. ~86M parameter class base variant (~200 MB checkpoint).
//!
//! # Vokra scope — SSL audio encoder (2026-07-30 scope expansion)
//!
//! Sibling of `beats` (iterative-tokenizer SSL), `eat`
//! (utterance-level Transformer + inverse block masking),
//! `dasheng` (universal MAE), `m2d` (masked-modeling-duo).
//! Distinct arch tag `atst` because the BYOL-style teacher-
//! student patchout topology is a distinct axis from every sibling
//! SSL encoder (contrastive / masked / dual-branch objectives all
//! differ) — silently sharing would misroute the runtime
//! dispatch and try to bind e.g. a MAE decoder over a
//! teacher-student checkpoint (FR-EX-08). Category
//! `audio-embedding`.
//!
//! # License posture — **cc-by-4.0 (weight) / mit (code)** — split
//!
//! **The upstream README explicitly separates code and weight
//! licenses**:
//!
//! > "The pretrained checkpoints hyper-linked in this repo are
//! > licensed under CC BY 4.0. To view a copy of this license,
//! > visit http://creativecommons.org/licenses/by/4.0/
//! >
//! > audiossl is licenced under MIT Licence."
//!
//! (`raw.githubusercontent.com/Audio-WestlakeU/audiossl/main/LICENSE`,
//! primary source task input 2026-08-13). GitHub API
//! `/repos/Audio-WestlakeU/audiossl/license` returns
//! `spdx_id: NOASSERTION` — GitHub's classifier does not know how
//! to combine "code MIT / weight CC-BY-4.0" into a single SPDX,
//! so the primary source is the LICENSE file text itself.
//!
//! **Vokra records the WEIGHT license (`cc-by-4.0`,
//! `AttributionRequired`) since `vokra.provenance.weight_license`
//! is a weight-tracking stamp**, not a code-tracking stamp. The
//! code SPDX (`mit`) applies to the ATST training code but is
//! not what a weight-provenance stamp records. Downstream
//! distributors of the weight must comply with CC BY 4.0
//! attribution requirements. §3.1 sign-off stays blank fail-closed
//! until owner completes primary-source confirmation.
//!
//! # Scale — local convert OK (~0.2 GB)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe). No vast.ai
//! handoff required.
//!
//! # No ONNX / no pickle (permanent)
//!
//! ATST ships as PyTorch `.ckpt` pickle from the upstream repo
//! release; this converter **never** touches ONNX or pickle
//! (FR-LD-05 / NFR-DS-02). Callers pre-flatten via a future
//! `tools/parity/atst_prepare_checkpoint.py` uv-managed Python
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

/// `vokra.model.arch` for ATST GGUFs. Distinct from sibling SSL
/// audio-encoder arch tags (`beats` / `eat` / `dasheng` / `m2d` /
/// `mert` / `muq`) — ATST's BYOL-style teacher-student patchout
/// training target is a distinct topology axis from every sibling.
pub const ARCH: &str = "atst";

/// `vokra.model.name` — canonical `atst-base` size point (the
/// INTERSPEECH 2022 release; sibling `atst-frame` is the
/// frame-level TASLP 2023 extension published as its own future
/// `NAME` following the snac_24khz / snac_44khz pattern).
pub const NAME: &str = "atst-base";

/// `vokra.model.category` — general audio-embedding (sibling of
/// `dasheng` / `beats` / `eat` / `m2d`; downstream sound-event
/// detection / audio-tagging / speaker heads feed from the
/// encoder's hidden states).
pub const CATEGORY: &str = "audio-embedding";

/// `vokra.provenance.upstream_url` value — the GitHub tree the
/// release ships from. ATST is not hosted on HuggingFace, so this
/// uses `upstream_url` rather than `upstream_hf`; the model-card
/// generator picks up either. Sibling of `beats::UPSTREAM_URL` /
/// `eat::UPSTREAM_URL` / `nsnet2::UPSTREAM_URL` posture.
pub const UPSTREAM_URL: &str =
    "github.com/Audio-WestlakeU/audiossl/tree/main/audiossl/methods/atst";

/// Default SPDX. **Weight license** = `cc-by-4.0` per upstream
/// LICENSE file text primary source (task input 2026-08-13):
///
/// > "The pretrained checkpoints hyper-linked in this repo are
/// > licensed under CC BY 4.0."
///
/// The code SPDX would be `mit` but `vokra.provenance.weight_license`
/// tracks the weight tier, not the code tier — CC-BY-4.0 is the
/// enforceable posture for weight redistribution
/// (`AttributionRequired` — downstream distributors must credit
/// the ATST authors per CC BY 4.0). A caller with a different
/// attestation may override at the outer boundary
/// (`--license <spdx>`).
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-4.0";

const UPSTREAM_SOURCE: &str = "Audio-WestlakeU/audiossl/methods/atst (Audio Teacher-Student Transformer, BYOL-style \
     EMA teacher + student patchout SSL audio encoder, ~86M params base, Li et al. \
     arXiv:2204.12076 INTERSPEECH 2022 + arXiv:2306.04186 TASLP 2023, code mit / weight cc-by-4.0)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Outcome of an ATST conversion. Mirrors the counter shape of
/// the sibling BF16 pass-through converters (`beats` / `eat` /
/// `dasheng` / `mert` / `muq` / `yamnet`) — the invariant
/// `read == written + skipped_non_float` is auditable at the
/// report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AtstReport {
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

/// Converts an ATST safetensors checkpoint at `input` into a
/// Vokra-native GGUF at `output`, returning an [`AtstReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` chunks
/// are stamped for the runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX` (`"cc-by-4.0"`,
/// `AttributionRequired`) since the weight is the tracked artifact.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_atst_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<AtstReport, ConvertError> {
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

    let mut report = AtstReport::default();
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
            "vokra-convert-atst-{tag}-{}-{n}",
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
    fn f32_tensor_passes_through_and_default_license_is_attribution_required() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // ATST uses a teacher/student duo — realistic upstream state-dict
        // name from the BYOL-style objective.
        let st = safetensors_one(
            "student.encoder.blocks.0.norm1.weight",
            "F32",
            &[3],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        let r = convert_atst_file(&inp, &outp, None).expect("convert F32");
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
            LicenseClass::AttributionRequired.as_str(),
            "cc-by-4.0 must resolve to AttributionRequired"
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
        let st = safetensors_one(
            "teacher.encoder.blocks.0.attn.qkv.weight",
            "BF16",
            &[2, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        let r = convert_atst_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("teacher.encoder.blocks.0.attn.qkv.weight")
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
        // AttributionRequired default — mirror of the audiogen_medium /
        // musicgen_medium escape hatch.
        convert_atst_file(&inp, &outp, Some("mit")).expect("convert with override");
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
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
