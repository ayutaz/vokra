#![allow(clippy::doc_lazy_continuation)]
//! **BEATs** (`microsoft/unilm/tree/master/beats`, **mit**): safetensors →
//! GGUF conversion (SSL audio-encoder wave, 2026-08-13).
//!
//! Input: the upstream `microsoft/unilm` release — BEATs
//! ("Bidirectional Encoder representation from Audio Transformers",
//! Chen et al. 2023 ICML arXiv:2212.09058) is a foundational
//! **self-supervised audio encoder** trained via **iterative acoustic
//! tokenizer + mask acoustic modeling**. The iter3-plus checkpoint is
//! the third refinement round (tokenizer at iter3 → mask acoustic
//! modeling → new tokenizer at iter3+ → mask acoustic modeling).
//! Weights ship as `.pt` checkpoints (`BEATs_iter3_plus_AS2M.pt` etc.)
//! from the upstream `github.com/microsoft/unilm/tree/master/beats`
//! release page (no first-party HuggingFace mirror), and are used as
//! (a) a general audio-embedding backbone for downstream tagging /
//! classification, and (b) the **acoustic teacher** for other SSL
//! releases (notably MuQ — Zhu et al. 2025 arXiv:2501.01108 uses
//! BEATs as its teacher). ~90M parameters (~340 MB `.pt`).
//!
//! # Vokra scope — SSL audio encoder (2026-07-30 scope expansion)
//!
//! Sibling of `dasheng` (universal MAE, speech + music + env), `mert`
//! (HuBERT-derived music-only MPM), `muq` (Mel-RVQ + BEATs teacher —
//! **downstream consumer of BEATs**), `eat` / `atst` / `m2d` (SSL
//! wave siblings). Distinct arch tag `beats` because the iterative
//! acoustic tokenizer + mask acoustic modeling training target is a
//! distinct topology from every sibling — silently sharing would
//! misroute the runtime dispatch and try to bind e.g. a HuBERT
//! decoder over an iterative-tokenizer checkpoint (FR-EX-08).
//! Category `audio-embedding` (sibling of `dasheng` / `eat` / `atst`
//! / `m2d`; downstream music-tagging / audio-classification / sound-
//! event heads feed from the encoder's hidden states).
//!
//! # License posture — mit (**Permissive**)
//!
//! Upstream `microsoft/unilm` root LICENSE `spdx_id: MIT` (GitHub
//! API `/repos/microsoft/unilm/license` primary source, task input
//! 2026-08-13). The `beats/` subdirectory carries no separate LICENSE
//! file — it inherits the root MIT umbrella. **HF community mirrors
//! (`mooneyko/BEATs`, `camenduru/beats`, `lpepino/beats_ckpts`) do
//! not carry `license:` tags in their cardData** ge no first-party
//! Microsoft HF org release exists as of 2026-08-13, so a `--license`
//! override to a stricter posture is a legitimate CC / owner path if
//! primary-source confirmation surfaces a per-checkpoint restriction.
//! §3.1 sign-off stays blank fail-closed until owner completes
//! primary-source confirmation (memory `[[feedback-license-signoff-
//! primary-source]]` — no CC pre-fill).
//!
//! # Scale — local convert OK (~0.34 GB / ~90M params base)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe). No vast.ai
//! handoff required.
//!
//! # No ONNX / no pickle (permanent)
//!
//! BEATs ships as PyTorch `.pt` pickle from the unilm release page;
//! this converter **never** touches ONNX or pickle (FR-LD-05 /
//! NFR-DS-02). Callers pre-flatten `BEATs_iter3_plus_AS2M.pt` →
//! `.safetensors` offline via a future
//! `tools/parity/beats_prepare_checkpoint.py` uv-managed Python 3.12
//! sidecar (memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`) mirroring the DAC / Kokoro / UTMOSv2
//! bridge pattern.
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

/// `vokra.model.arch` for BEATs GGUFs. Distinct from sibling SSL
/// audio-encoder arch tags (`dasheng` / `eat` / `atst` / `m2d` /
/// `mert` / `muq`) — BEATs' iterative acoustic tokenizer + mask
/// acoustic modeling target is a distinct topology axis from MAE
/// ViT/ConvNeXt (Dasheng), utterance-level Transformer (EAT),
/// teacher-student patchout (ATST), masked-modeling-duo (M2D),
/// HuBERT-derived (MERT), and Mel-RVQ + BEATs teacher (MuQ).
pub const ARCH: &str = "beats";

/// `vokra.model.name` — canonical BEATs `iter3_plus_AS2M` checkpoint
/// (the "third-plus" refinement fine-tuned on AudioSet-2M, the
/// primary-source release named on the unilm README). Sibling
/// checkpoints (`iter3_plus_AS2M_finetuned_on_AS2M`, iter1 / iter2 /
/// iter3, `_finetuned_on_AS20K` etc.) are variants of this same
/// arch; the canonical NAME anchors the arch-shared publish slug and
/// downstream heads can be swapped via converter args in a future
/// wave if needed.
pub const NAME: &str = "beats-iter3-plus-as2m";

/// `vokra.model.category` — general audio-embedding (sibling of
/// `dasheng` / `eat` / `atst` / `m2d`; downstream music-tagging /
/// audio-classification / sound-event heads feed from the encoder's
/// hidden states).
pub const CATEGORY: &str = "audio-embedding";

/// `vokra.provenance.upstream_url` value — the GitHub tree the
/// release ships from. BEATs is not hosted on HuggingFace as a
/// first-party Microsoft release (community mirrors on HF carry no
/// `license:` tag as of 2026-08-13), so we record `upstream_url`
/// rather than `upstream_hf`; the model-card generator picks up
/// either. Sibling of `nsnet2::UPSTREAM_URL` / `emotion2vec`
/// posture.
pub const UPSTREAM_URL: &str = "github.com/microsoft/unilm/tree/master/beats";

/// Default SPDX. Upstream `microsoft/unilm` root LICENSE
/// `spdx_id: MIT` (GitHub API `/repos/microsoft/unilm/license` per
/// task input 2026-08-13). The `beats/` subdirectory inherits the
/// root umbrella. A caller with a different attestation may override
/// at the outer boundary (`--license <spdx>`).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

const UPSTREAM_SOURCE: &str = "microsoft/unilm/tree/master/beats (BEATs: Bidirectional Encoder representation \
     from Audio Transformers, iterative acoustic tokenizer + mask acoustic modeling, ~90M \
     params iter3_plus_AS2M, Chen et al. arXiv:2212.09058 ICML 2023, mit)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Outcome of a BEATs conversion. Mirrors the counter shape of the
/// sibling BF16 pass-through converters (`dasheng` / `mert` / `muq`
/// / `yamnet`) — the invariant `read == written + skipped_non_float`
/// is auditable at the report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BeatsReport {
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

/// Converts a BEATs safetensors checkpoint at `input` into a
/// Vokra-native GGUF at `output`, returning a [`BeatsReport`].
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
pub fn convert_beats_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<BeatsReport, ConvertError> {
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

    let mut report = BeatsReport::default();
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
            "vokra-convert-beats-{tag}-{}-{n}",
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
        // BEATs uses a `patch_embed` conv stem and stacked Transformer
        // encoder blocks (`encoder.layers.N.self_attn.q_proj.weight` etc.).
        let st = safetensors_one("patch_embedding.proj.weight", "F32", &[1, 3], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_beats_file(&inp, &outp, None).expect("convert F32");
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
        let st = safetensors_one(
            "encoder.layers.0.self_attn.q_proj.weight",
            "BF16",
            &[2, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        let r = convert_beats_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("encoder.layers.0.self_attn.q_proj.weight")
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

        // A caller with a stricter attestation may downgrade off the
        // default MIT (e.g. a per-checkpoint NC posture surfacing during
        // primary-source verification).
        convert_beats_file(&inp, &outp, Some("cc-by-nc-4.0")).expect("convert with override");
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("cc-by-nc-4.0"),
        );
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercial.as_str()),
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
