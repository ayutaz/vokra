#![allow(clippy::doc_lazy_continuation)]
//! **M2D** (`nttcslab/m2d`, **license unknown**): safetensors → GGUF
//! conversion (SSL audio-encoder wave, 2026-08-13).
//!
//! Input: the upstream `nttcslab/m2d` release — M2D
//! ("Masked Modeling Duo") is a self-supervised audio encoder
//! from NTT Communication Science Laboratories that jointly
//! predicts masked patches from a **target** online branch AND
//! its **predictive representation** via a dual-branch objective
//! (Niizumi et al. 2023, ICASSP arXiv:2210.14648, "Masked Modeling
//! Duo: Learning Representations by Encouraging Both Networks to
//! Model the Input"; TASLP 2024 extension for sound event detection
//! and speech). Positioned as an efficient audio-embedding backbone
//! for downstream sound-event detection / audio-tagging / speaker
//! tasks. ~86M parameter class base variant (~200 MB).
//!
//! # Vokra scope — SSL audio encoder (2026-07-30 scope expansion)
//!
//! Sibling of `beats` (iterative-tokenizer SSL), `eat`
//! (utterance-level Transformer + inverse block masking),
//! `atst` (teacher-student patchout), `dasheng` (universal MAE).
//! Distinct arch tag `m2d` because the masked-modeling-**duo**
//! (dual online + target branch, joint prediction of masked
//! patches AND their online-branch representation) topology is a
//! distinct axis from every sibling SSL encoder (single-branch
//! MAE = Dasheng / EAT, teacher-student patchout = ATST,
//! iterative tokenizer = BEATs). Silently sharing would misroute
//! the runtime dispatch and try to bind e.g. a single-branch MAE
//! decoder over a dual-branch checkpoint (FR-EX-08). Category
//! `audio-embedding`.
//!
//! # License posture — **Unknown** (fail-closed)
//!
//! Upstream `github.com/nttcslab/m2d` LICENSE is a **PDF file**
//! (`LICENSE.pdf`) that GitHub's classifier cannot machine-read —
//! GitHub API `/repos/nttcslab/m2d/license` returns
//! `spdx_id: NOASSERTION` with body decoding to:
//!
//! > "Please find the LICENSE at
//! > `https://github.com/nttcslab/m2d/blob/master/LICENSE.pdf`"
//!
//! (verified via GitHub API primary source task input 2026-08-13).
//! **No HuggingFace mirror exists as of 2026-08-13** (search of
//! `nttcslab/m2d` and `m2d` audio-tagged returned no matches).
//! Provenance stamp defaults to [`LicenseClass::Unknown`]
//! (fail-closed under M2-13). Owner must:
//!
//! 1. Download `LICENSE.pdf` and read it,
//! 2. Complete primary-source confirmation on the SPDX tier,
//! 3. Override via `--license <spdx>` at the outer boundary
//!    (`convert_m2d_file`'s `license` parameter → the CLI
//!    `--license` flag propagates through `convert_file_licensed`).
//!
//! §3.1 sign-off stays blank fail-closed until this owner ADR
//! completes (memory `[[feedback-license-signoff-primary-source]]`
//! — no CC pre-fill).
//!
//! # Scale — local convert OK (~0.2 GB)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe). No vast.ai
//! handoff required.
//!
//! # No ONNX / no pickle (permanent)
//!
//! M2D ships as PyTorch `.pth` pickle from the upstream release
//! (linked from README, hosted externally); this converter
//! **never** touches ONNX or pickle (FR-LD-05 / NFR-DS-02).
//! Callers pre-flatten via a future
//! `tools/parity/m2d_prepare_checkpoint.py` uv-managed Python
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

/// `vokra.model.arch` for M2D GGUFs. Distinct from sibling SSL
/// audio-encoder arch tags (`beats` / `eat` / `atst` / `dasheng` /
/// `mert` / `muq`) — M2D's dual-branch (online + target) masked-
/// modeling-duo training target is a distinct topology axis from
/// every sibling.
pub const ARCH: &str = "m2d";

/// `vokra.model.name` — canonical `m2d-base` size point. Sibling
/// variants (`m2d-eat` sound-event-detection specialization,
/// speech-specific fine-tunes etc.) are distinct release
/// identities published as their own future `NAME` following the
/// snac_24khz / snac_44khz pattern (added via separate future
/// ModelKind).
pub const NAME: &str = "m2d-base";

/// `vokra.model.category` — general audio-embedding (sibling of
/// `dasheng` / `beats` / `eat` / `atst`; downstream sound-event
/// detection / audio-tagging / speaker heads feed from the
/// encoder's hidden states).
pub const CATEGORY: &str = "audio-embedding";

/// `vokra.provenance.upstream_url` value — the GitHub tree the
/// release ships from. M2D is not hosted on HuggingFace, so this
/// uses `upstream_url` rather than `upstream_hf`; the model-card
/// generator picks up either. Sibling of `beats::UPSTREAM_URL` /
/// `eat::UPSTREAM_URL` / `atst::UPSTREAM_URL` /
/// `nsnet2::UPSTREAM_URL` posture.
pub const UPSTREAM_URL: &str = "github.com/nttcslab/m2d";

/// Default SPDX. Upstream `nttcslab/m2d` LICENSE is a **PDF file**
/// (`LICENSE.pdf`); GitHub API `/repos/nttcslab/m2d/license`
/// returns `spdx_id: NOASSERTION` (task input 2026-08-13). The
/// classifier `from_license_str("unknown")` correctly resolves to
/// [`LicenseClass::Unknown`] (fail-closed under M2-13, runtime
/// gate refuses to load without a research flag). Owner must
/// download `LICENSE.pdf`, read it, and override via
/// `--license <spdx>` at the outer boundary once the SPDX tier
/// is confirmed.
pub const DEFAULT_LICENSE_SPDX: &str = "unknown";

const UPSTREAM_SOURCE: &str = "nttcslab/m2d (Masked Modeling Duo — dual-branch SSL audio encoder joint-predicting \
     masked patches AND online-branch representation, ~86M params base, Niizumi et al. \
     arXiv:2210.14648 ICASSP 2023 + TASLP 2024 extension, LICENSE.pdf non-machine-readable \
     — fail-closed unknown)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Outcome of an M2D conversion. Mirrors the counter shape of the
/// sibling BF16 pass-through converters (`beats` / `eat` / `atst`
/// / `dasheng` / `mert` / `muq` / `yamnet`) — the invariant
/// `read == written + skipped_non_float` is auditable at the
/// report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct M2dReport {
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

/// Converts an M2D safetensors checkpoint at `input` into a
/// Vokra-native GGUF at `output`, returning an [`M2dReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` chunks
/// are stamped for the runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX` (`"unknown"`,
/// `Unknown`) — fail-closed under M2-13, publish refused until a
/// caller supplies a real SPDX + §3.1 sign-off completes after owner
/// reads upstream `LICENSE.pdf`.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_m2d_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<M2dReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // Unknown fail-closed default: if the caller passes no license
    // string, the classifier resolves `"unknown"` to
    // `LicenseClass::Unknown` and the M2-13 runtime gate refuses to
    // load without a research flag. Any caller who has resolved the
    // license out-of-band (by downloading LICENSE.pdf and reading
    // it) supplies `--license <spdx>` at the outer boundary.
    let spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(&mut b, class, spdx, Some(NAME), Some(UPSTREAM_SOURCE));

    let mut report = M2dReport::default();
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
            "vokra-convert-m2d-{tag}-{}-{n}",
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
        // M2D uses an online + target duo — realistic upstream
        // state-dict name from the dual-branch objective.
        let st = safetensors_one("online.blocks.0.attn.qkv.weight", "F32", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_m2d_file(&inp, &outp, None).expect("convert F32");
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
        let st = safetensors_one("target.blocks.0.attn.qkv.weight", "BF16", &[2, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_m2d_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("target.blocks.0.attn.qkv.weight")
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

        convert_m2d_file(&inp, &outp, Some("apache-2.0")).expect("convert with override");

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
