#![allow(clippy::doc_lazy_continuation)]
//! **Basic-Pitch** (`spotify/basic-pitch`, apache-2.0): safetensors →
//! GGUF conversion (music-understanding wave, 2026-08-13).
//!
//! Input: the upstream `spotify/basic-pitch` release — Spotify's
//! polyphonic audio-to-MIDI pitch-detection model (Bittner et al.
//! 2022 ICASSP "A Lightweight Instrument-Agnostic Model for
//! Polyphonic Note Transcription"). Basic-Pitch is a ~6 MB CNN over
//! CQT (constant-Q transform) input producing three output
//! "posteriorgrams": (1) frame-level note activation, (2) note onset
//! detection, and (3) contour (multi-pitch F0 estimate). Post-
//! processing converts these into MIDI notes. Instrument-agnostic,
//! covers polyphonic music (piano, guitar, vocals, etc.) and works
//! on 22.05 kHz mono.
//!
//! # Vokra scope — music understanding (2026-07-30 scope expansion)
//!
//! Complements the sibling F0 pitch-extractor family (`crepe` /
//! `fcpe` / `rmvpe`, monophonic) with a **polyphonic** pitch/note
//! extractor. Distinct arch tag `basic-pitch` because the output
//! surface (3-head posteriorgram over frames + onsets + contour) is
//! a distinct topology from the monophonic F0 siblings (single-head
//! 360-class or 384-class pitch grid). Category `pitch-transcription`
//! — distinct from `f0` (monophonic pitch extraction) because the
//! output is polyphonic MIDI notes rather than a single-voice F0
//! contour.
//!
//! # License posture — apache-2.0 (**Permissive**)
//!
//! Upstream `spotify/basic-pitch` HF cardData primary source
//! `license: apache-2.0` (task input 2026-08-13, Spotify Research
//! release; verified against the reference GitHub repo
//! `spotify/basic-pitch` which is Apache-2.0 with commercial-safe
//! posture). §3.1 sign-off stays blank fail-closed until owner
//! completes primary-source confirmation.
//!
//! # Scale — local convert OK (~0.006 GB / ~6 MB)
//!
//! Smallest converter in the music-understanding wave — well below
//! the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe).
//!
//! # No ONNX (permanent)
//!
//! Basic-Pitch upstream ships as TensorFlow SavedModel + CoreML
//! (`nmp.tflite` for edge) — this converter **never** touches ONNX
//! (FR-LD-05). Callers pre-flatten the TF SavedModel or torch port
//! to safetensors offline via a future
//! `tools/parity/basic_pitch_prepare_checkpoint.py` (the DAC /
//! Kokoro / UTMOSv2 bridge pattern).
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

/// `vokra.model.arch` for Basic-Pitch GGUFs. Distinct from every
/// sibling F0 extractor (`crepe` / `fcpe` / `rmvpe`) because the
/// 3-head polyphonic posteriorgram output is a distinct topology
/// from the single-head monophonic pitch grid.
pub const ARCH: &str = "basic-pitch";

/// `vokra.model.name` — canonical `spotify/basic-pitch` release.
pub const NAME: &str = "basic-pitch";

/// `vokra.model.category` — pitch-transcription (polyphonic MIDI-note
/// extraction from audio, distinct from the monophonic `f0` category
/// owned by `crepe` / `fcpe` / `rmvpe`).
pub const CATEGORY: &str = "pitch-transcription";

/// Upstream HF slug — recorded on `vokra.provenance.upstream_hf`.
pub const UPSTREAM_HF: &str = "spotify/basic-pitch";

/// Default SPDX. Upstream `spotify/basic-pitch` HF cardData primary
/// source `license: apache-2.0` (task input 2026-08-13, Spotify
/// Research release).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

const UPSTREAM_SOURCE: &str = "spotify/basic-pitch (Spotify Research polyphonic audio-to-MIDI pitch-detection \
     model, ~6 MB CNN over CQT with 3-head posteriorgram output (frame / onset / contour), \
     instrument-agnostic, Bittner et al. ICASSP 2022, apache-2.0)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a Basic-Pitch conversion. Mirrors the counter shape of
/// the sibling BF16 pass-through converters (`yamnet` / `mert` /
/// `muq` / `dasheng` / `panns`) — the invariant `read == written +
/// skipped_non_float` is auditable at the report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BasicPitchReport {
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

/// Converts a `spotify/basic-pitch` safetensors checkpoint at `input`
/// into a Vokra-native GGUF at `output`, returning a
/// [`BasicPitchReport`].
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
pub fn convert_basic_pitch_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<BasicPitchReport, ConvertError> {
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

    let mut report = BasicPitchReport::default();
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
            "vokra-convert-basic-pitch-{tag}-{}-{n}",
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
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Basic-Pitch head key — 3 output heads (contour / note / onset).
        let st = safetensors_one("contour_head.conv.weight", "F32", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_basic_pitch_file(&inp, &outp, None).expect("convert F32");
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
            LicenseClass::Permissive.as_str(),
            "apache-2.0 must resolve to Permissive"
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
        let st = safetensors_one("note_head.conv.weight", "BF16", &[2, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_basic_pitch_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("note_head.conv.weight")
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

        convert_basic_pitch_file(&inp, &outp, Some("mit")).expect("convert with override");
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
