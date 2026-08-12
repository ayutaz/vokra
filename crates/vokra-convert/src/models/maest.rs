#![allow(clippy::doc_lazy_continuation)]
//! **MAEST** (`mtg-upf/discogs-maest-30s-pw-129e`,
//! **cc-by-nc-sa-4.0**): safetensors → GGUF conversion
//! (SSL audio-encoder wave, 2026-08-13).
//!
//! Input: the upstream `mtg-upf/discogs-maest-30s-pw-129e` release —
//! MAEST ("Music **A**udio **E**fficient **S**pectrogram
//! **T**ransformer", Alonso-Jiménez et al. 2023 ISMIR
//! arXiv:2309.16418) is a self-supervised **music-tagger** built on
//! the **Audio Spectrogram Transformer (AST)** backbone (HF
//! `config`: `model_type: audio-spectrogram-transformer`,
//! `architectures: ["ASTForAudioClassification"]`, verified via HF
//! cardData API 2026-08-13). The `30s-pw-129e` variant is 30-second
//! patch-wise pretrained for 129 epochs on the MTG Discogs4All
//! music-tagger dataset. ~87M F32 parameters (safetensors
//! `parameters.F32: 86,858,128` per HF API primary source).
//!
//! # Vokra scope — music understanding via AST-backbone SSL
//!
//! Sibling of `mert` (HuBERT-derived Conv1D + Transformer MPM),
//! `muq` (Mel-RVQ + BEATs teacher), `dasheng` (universal MAE).
//! Distinct arch tag `maest` because the AST-backbone (patch-wise
//! Transformer over log-mel spectrogram) + Discogs-tagger SSL
//! pretraining objective is a distinct topology axis from every
//! sibling music-embedding model — silently sharing would
//! misroute the runtime dispatch (FR-EX-08). Category
//! `music-embedding` (sibling of `mert` / `muq`; downstream
//! music-tagging heads consume the encoder's hidden states).
//!
//! # License posture — CC-BY-NC-SA 4.0 (**NonCommercialShareAlike**)
//!
//! Upstream HF cardData `license: cc-by-nc-sa-4.0` (verified via
//! `https://huggingface.co/api/models/mtg-upf/discogs-maest-30s-pw-129e`
//! primary source task input 2026-08-13; HF tag
//! `license:cc-by-nc-sa-4.0` also present). This is **T4 tier +
//! SA cascade** — the strictest CC family + share-alike:
//!
//! - **NonCommercial** — commercial use forbidden without a
//!   separate license from the MTG group.
//! - **ShareAlike** — any downstream distribution of the weight
//!   (or a derivative) must be under the same CC-BY-NC-SA 4.0
//!   license, effectively preventing re-licensing.
//! - **BY (Attribution)** — cascaded attribution requirement.
//!
//! **Publish path**: `publish-one.sh --allow-noncommercial` gate
//! + `fetch_license.sh --spdx cc-by-nc-sa-4.0` canonical LICENSE
//! bundled (nisqa_v2_weight / audioldm2 precedent). §3.1 sign-off
//! stays blank fail-closed until owner completes primary-source
//! confirmation (memory `[[feedback-license-signoff-primary-source]]`
//! — no CC pre-fill).
//!
//! # Scale — local convert OK (~0.15 GB / ~87M F32 params)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe). No vast.ai
//! handoff required.
//!
//! # No ONNX / no pickle (permanent)
//!
//! MAEST ships as single-file safetensors (`model.safetensors`)
//! per HF `siblings` inspection (also includes legacy
//! `pytorch_model.bin` pickle which Vokra never reads). This
//! converter **never** touches ONNX or pickle (FR-LD-05 /
//! NFR-DS-02) — the safetensors path is the primary source and
//! no bridge sidecar is needed.
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

/// `vokra.model.arch` for MAEST GGUFs. Distinct from sibling
/// music-embedding arch tags (`mert` = HuBERT-derived MPM /
/// `muq` = Mel-RVQ + BEATs teacher / `dasheng` = MAE
/// ViT/ConvNeXt) — MAEST's AST-backbone (patch-wise Transformer
/// over log-mel spectrogram) + Discogs-tagger SSL objective is a
/// distinct topology axis from every sibling.
pub const ARCH: &str = "maest";

/// `vokra.model.name` — canonical `mtg-upf/discogs-maest-30s-pw-129e`
/// release variant (30-second, patch-wise, 129 epochs — the
/// primary release variant per the upstream README). Sibling
/// variants (5s / 10s / 20s durations, 30s-pw-73e checkpoint
/// point etc.) are distinct release identities published as their
/// own future `NAME` following the snac_24khz / snac_44khz
/// pattern (added via separate future ModelKind).
pub const NAME: &str = "maest-30s-pw-129e";

/// `vokra.model.category` — music understanding embedding
/// (sibling of `mert` / `muq`; downstream music-tagging heads
/// consume the encoder's hidden states). Distinct from
/// `audio-tagging` (sibling of `yamnet` / `panns` / `ast` /
/// `clap`) because MAEST is trained specifically on the Discogs
/// music-tagger dataset — output is genre / mood / instrument /
/// era annotations over music, not the general AudioSet audio-
/// event ontology.
pub const CATEGORY: &str = "music-embedding";

/// Upstream HF slug — recorded on `vokra.provenance.upstream_hf`.
pub const UPSTREAM_HF: &str = "mtg-upf/discogs-maest-30s-pw-129e";

/// Default SPDX. HF cardData primary source `license: cc-by-nc-sa-4.0`
/// (verified via `https://huggingface.co/api/models/mtg-upf/
/// discogs-maest-30s-pw-129e` task input 2026-08-13). A caller
/// with a different attestation may override at the outer
/// boundary (`--license <spdx>`); the M2-13 runtime gate then
/// reclassifies.
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-sa-4.0";

const UPSTREAM_SOURCE: &str = "mtg-upf/discogs-maest-30s-pw-129e (Music AEST — Discogs-pretrained AST self-supervised \
     music-tagger, 30-second patch-wise 129-epoch pretraining, ~87M F32 params, \
     Alonso-Jiménez et al. arXiv:2309.16418 ISMIR 2023, cc-by-nc-sa-4.0)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a MAEST conversion. Mirrors the counter shape of
/// the sibling BF16 pass-through converters (`mert` / `muq` /
/// `dasheng` / `beats` / `eat` / `atst` / `yamnet`) — the
/// invariant `read == written + skipped_non_float` is auditable
/// at the report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MaestReport {
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

/// Converts a `mtg-upf/discogs-maest-30s-pw-129e` safetensors
/// checkpoint at `input` into a Vokra-native GGUF at `output`,
/// returning a [`MaestReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` chunks
/// are stamped for the runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX` (`"cc-by-nc-sa-4.0"`,
/// `NonCommercialShareAlike`) — fail-closed under M2-13, publish
/// requires `publish-one.sh --allow-noncommercial` + share-alike
/// obligation on any downstream distribution.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_maest_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MaestReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(&mut b, class, spdx, Some(NAME), Some(UPSTREAM_SOURCE));

    let mut report = MaestReport::default();
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
            "vokra-convert-maest-{tag}-{}-{n}",
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
    fn f32_tensor_passes_through_and_default_license_is_ncsa_fail_closed() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // MAEST uses HF's ASTForAudioClassification wrapper — realistic
        // upstream state-dict name from the AST-backbone body.
        let st = safetensors_one(
            "audio_spectrogram_transformer.encoder.layer.0.attention.attention.query.weight",
            "F32",
            &[3],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        let r = convert_maest_file(&inp, &outp, None).expect("convert F32");
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
            LicenseClass::NonCommercialShareAlike.as_str(),
            "cc-by-nc-sa-4.0 must resolve to NonCommercialShareAlike (T4 + SA cascade)"
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
            "audio_spectrogram_transformer.encoder.layer.0.output.dense.weight",
            "BF16",
            &[2, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        let r = convert_maest_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("audio_spectrogram_transformer.encoder.layer.0.output.dense.weight")
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
        // NonCommercialShareAlike default — mirror of the audioldm2 /
        // nisqa_v2_weight escape hatch (though for MAEST this would
        // require MTG group re-license).
        convert_maest_file(&inp, &outp, Some("apache-2.0")).expect("convert with override");
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
