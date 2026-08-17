#![allow(clippy::doc_lazy_continuation)]
//! **YAMNet** (`thelou1s/yamnet`, apache-2.0): safetensors → GGUF
//! conversion (music-understanding wave, 2026-08-13).
//!
//! Input: the upstream `thelou1s/yamnet` HF mirror of Google Research's
//! YAMNet audio-event classifier (521-class AudioSet). YAMNet is a
//! MobileNetV1-flavour depthwise-separable-CNN acoustic-scene classifier
//! trained on Google's AudioSet ontology (edge-friendly ~15 MB, 16 kHz
//! mono log-mel input at 96 mel bins × ~0.96 s frames). Serves as the
//! smallest music-understanding target in the wave — its 521-class
//! output covers instruments / music genres alongside general
//! environmental audio, so it complements the deeper MERT / MuQ /
//! Dasheng / PANNs / Basic-Pitch music-understanding stack with a
//! low-latency triage layer.
//!
//! # Vokra scope — music understanding (2026-07-30 scope expansion)
//!
//! Sibling of `panns` / `ast` / `clap` (audio-tagging family) — YAMNet
//! is the tiny 15 MB edge-class member of the same 521/527-class family.
//! Distinct arch tag `yamnet` from every sibling because the encoder
//! topology (MobileNetV1 depthwise-separable Conv2D) differs from AST
//! (patch-embed Transformer) / CLAP (contrastive text-audio) / PANNs
//! (Cnn10/Cnn14 residual Conv2D). Category `audio-tagging` — silently
//! sharing an arch tag would misroute the runtime dispatch and route
//! e.g. a MobileNet checkpoint through a residual Cnn14 loader.
//!
//! # License posture — apache-2.0 (**Permissive**)
//!
//! Upstream YAMNet reference (Google Research tensorflow/models) is
//! Apache-2.0. The `thelou1s/yamnet` HF mirror carries no
//! model-card `license:` tag in its front-matter as of 2026-08-13 —
//! provenance stamp defaults to `apache-2.0` (the Google Research
//! source license); a caller with a different attestation may override
//! via `--license <spdx>` at the outer boundary. §3.1 sign-off stays
//! blank fail-closed until owner completes primary-source
//! confirmation of the mirror repo's LICENSE.
//!
//! # Scale — local convert OK (~15 MB)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe). No vast.ai
//! handoff required.
//!
//! # No ONNX (permanent)
//!
//! YAMNet's upstream distribution is TensorFlow SavedModel + tfhub.dev;
//! HF mirrors typically ship it as `.h5` or converted safetensors.
//! Callers pre-flatten the checkpoint to safetensors offline via a
//! future `tools/parity/yamnet_prepare_checkpoint.py` (the DAC /
//! Kokoro / UTMOSv2 pattern — zero-dep, no TF/Keras/torch in the
//! runtime, NFR-DS-02 / FR-LD-05). This converter reads safetensors
//! only.
//!
//! # BF16 pass-through (mirror of utmosv2 / nkf_aec)
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

/// `vokra.model.arch` for YAMNet GGUFs. Distinct from sibling
/// audio-tagging arch tags (`panns` / `ast` / `clap`) because the
/// MobileNetV1 depthwise-separable-Conv2D backbone is a distinct
/// topology from Cnn10/Cnn14 residual (PANNs), patch-embed
/// Transformer (AST), or contrastive text-audio (CLAP).
pub const ARCH: &str = "yamnet";

/// `vokra.model.name` for the canonical `thelou1s/yamnet` release.
pub const NAME: &str = "yamnet";

/// `vokra.model.category` — audio-tagging (521-class AudioSet).
pub const CATEGORY: &str = "audio-tagging";

/// Upstream HF slug — recorded on `vokra.provenance.upstream_hf`.
pub const UPSTREAM_HF: &str = "thelou1s/yamnet";

/// Default SPDX. Reference implementation
/// (`github.com/tensorflow/models/tree/master/research/audioset/yamnet`)
/// is Apache-2.0 (Google Research standard); the HF mirror carries no
/// explicit license tag as of 2026-08-13, so a caller with a different
/// attestation should pass `--license <spdx>` at the outer boundary.
/// §3.1 sign-off stays blank until owner confirms.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

const UPSTREAM_SOURCE: &str = "thelou1s/yamnet (Google Research YAMNet MobileNetV1 audio-event classifier, \
     521-class AudioSet, ~15 MB edge model, apache-2.0)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a YAMNet conversion. Mirrors the counter shape of the
/// sibling BF16 pass-through converters (`utmosv2` / `nkf_aec`) — the
/// invariant `read == written + skipped_non_float` is auditable at the
/// report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct YamnetReport {
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

/// Converts a `thelou1s/yamnet` safetensors checkpoint at `input` into
/// a Vokra-native GGUF at `output`, returning a [`YamnetReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` chunks
/// are stamped for the runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX` (`"apache-2.0"`,
/// `Permissive`).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_yamnet_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<YamnetReport, ConvertError> {
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

    let mut report = YamnetReport::default();
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
            "vokra-convert-yamnet-{tag}-{}-{n}",
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
        let payload: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Realistic YAMNet backbone key: MobileNetV1 depthwise-separable
        // conv block. The state-dict name convention comes from Google
        // Research's tensorflow/models/research/audioset/yamnet.
        let st = safetensors_one("layer1.pointwise_conv.weight", "F32", &[2, 3], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_yamnet_file(&inp, &outp, None).expect("convert F32");
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);
        assert_eq!(r.skipped_non_float, 0);
        assert_eq!(r.bf16_passthrough, 0);

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
            LicenseClass::Permissive.as_str(),
            "apache-2.0 must resolve to Permissive (T1 tier)"
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(payload.len(), 12);
        let st = safetensors_one("classifier.weight", "BF16", &[2, 3], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_yamnet_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.written, 1);
        assert_eq!(r.bf16_passthrough, 1);

        // Byte-identity: BF16 must survive the round-trip verbatim
        // (no convert-time widen to F32).
        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("classifier.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
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

        convert_yamnet_file(&inp, &outp, Some("mit")).expect("convert with override");

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
