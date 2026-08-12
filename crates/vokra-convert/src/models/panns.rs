#![allow(clippy::doc_lazy_continuation)]
//! **PANNs Cnn14** (`nicofarr/panns_Cnn14`, **license unknown**):
//! safetensors → GGUF conversion (music-understanding wave,
//! 2026-08-13).
//!
//! Input: the upstream `nicofarr/panns_Cnn14` HF mirror of the
//! **Pretrained Audio Neural Networks** (PANNs) `Cnn14` checkpoint
//! (Kong et al. 2020 arXiv:1912.10211 "PANNs: Large-Scale Pretrained
//! Audio Neural Networks for Audio Pattern Recognition"). Cnn14 is
//! a VGG-style 14-layer 2D-CNN over 64-mel log-spectrogram, trained
//! on AudioSet (2 M weakly-labelled 10 s YouTube clips) for a
//! **527-class** ontology output. Widely used as a music-tagging /
//! sound-event-detection backbone; produces frame-level embeddings
//! (2048-d for Cnn14) or clip-level 527-way probabilities.
//! `nicofarr/panns_Cnn14` HF mirror hosts the checkpoint under the
//! Cnn14 tag; the upstream reference is `qiuqiangkong/audioset_tagging_cnn`
//! (MIT), but the mirror repo does not declare a `license:` tag.
//!
//! # Vokra scope — audio tagging (2026-07-30 scope expansion)
//!
//! Sibling of `yamnet` (521-class edge classifier, MobileNetV1) and
//! `ast` / `clap` (audio-tagging family). Distinct arch tag `panns`
//! from `yamnet` because the residual Cnn14 backbone is a distinct
//! topology from MobileNetV1 depthwise-separable — silently sharing
//! an arch tag would misroute the runtime dispatch and try to bind
//! a depthwise-conv loader over a residual-conv checkpoint. Category
//! `audio-tagging`.
//!
//! # License posture — **Unknown** (fail-closed)
//!
//! Upstream reference `qiuqiangkong/audioset_tagging_cnn` is MIT
//! (well-known — Kong et al. 2020 ICASSP paper implementation), but
//! the HF mirror repo `nicofarr/panns_Cnn14` carries no cardData
//! `license:` tag as of task input 2026-08-13 and its LICENSE file
//! status is un-verified. Fail-closed default = `Unknown` under
//! M2-13. A caller who has verified the mirror's LICENSE
//! out-of-band overrides via `--license mit` at the outer boundary.
//! §3.1 sign-off stays blank until owner completes primary-source
//! confirmation.
//!
//! # Scale — local convert OK (~0.35 GB / ~80M params Cnn14)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe). No vast.ai
//! handoff required.
//!
//! # No ONNX (permanent)
//!
//! PANNs upstream ships as PyTorch pickle (`.pth`); this converter
//! **never** touches ONNX (FR-LD-05). Callers pre-flatten to
//! safetensors offline via a future
//! `tools/parity/panns_prepare_checkpoint.py` (the DAC / Kokoro /
//! UTMOSv2 bridge pattern — no pickle in the runtime, NFR-DS-02).
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

/// `vokra.model.arch` for PANNs GGUFs. Distinct from sibling
/// audio-tagging arch tags (`yamnet` / `ast` / `clap`) because the
/// residual VGG-flavour Cnn14 is a distinct topology.
pub const ARCH: &str = "panns";

/// `vokra.model.name` — canonical `nicofarr/panns_Cnn14` mirror
/// (Cnn14 variant of the PANNs family).
pub const NAME: &str = "panns-cnn14";

/// `vokra.model.category` — audio-tagging (527-class AudioSet
/// ontology).
pub const CATEGORY: &str = "audio-tagging";

/// Upstream HF slug — recorded on `vokra.provenance.upstream_hf`.
pub const UPSTREAM_HF: &str = "nicofarr/panns_Cnn14";

/// Default SPDX. HF mirror `nicofarr/panns_Cnn14` carries no
/// cardData `license:` tag as of task input 2026-08-13 (upstream
/// reference `qiuqiangkong/audioset_tagging_cnn` is MIT but the
/// mirror LICENSE is un-verified). Fail-closed default = `unknown`.
pub const DEFAULT_LICENSE_SPDX: &str = "unknown";

const UPSTREAM_SOURCE: &str = "nicofarr/panns_Cnn14 (HF mirror of Pretrained Audio Neural Networks Cnn14, \
     VGG-flavour 14-layer 2D-CNN over 64-mel log-spectrogram, 527-class AudioSet, \
     ~80M params, Kong et al. arXiv:1912.10211 / qiuqiangkong/audioset_tagging_cnn \
     upstream reference MIT, mirror LICENSE un-verified — fail-closed)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a PANNs conversion. Mirrors the counter shape of the
/// sibling BF16 pass-through converters (`yamnet` / `mert` / `muq` /
/// `dasheng`) — the invariant `read == written + skipped_non_float`
/// is auditable at the report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PannsReport {
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

/// Converts a `nicofarr/panns_Cnn14` safetensors checkpoint at
/// `input` into a Vokra-native GGUF at `output`, returning a
/// [`PannsReport`].
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
pub fn convert_panns_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<PannsReport, ConvertError> {
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

    let mut report = PannsReport::default();
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
            "vokra-convert-panns-{tag}-{}-{n}",
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
        // PANNs Cnn14 block key — the Kong et al. reference state-dict
        // uses `conv_block{n}.conv{i}.weight` names.
        let st = safetensors_one("conv_block1.conv1.weight", "F32", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_panns_file(&inp, &outp, None).expect("convert F32");
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
        let values: [f32; 4] = [1.0, -0.5, 0.25, 8.0];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one("fc_audioset.weight", "BF16", &[2, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_panns_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("fc_audioset.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn license_override_mit_flips_to_permissive() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        // Upstream reference qiuqiangkong/audioset_tagging_cnn is MIT.
        // A caller who verifies the mirror LICENSE out-of-band may
        // supply `--license mit` to flip the classifier out of
        // Unknown fail-closed.
        convert_panns_file(&inp, &outp, Some("mit")).expect("convert with override");

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
