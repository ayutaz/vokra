#![allow(clippy::doc_lazy_continuation)]
//! **AudioGen-Medium** (`facebook/audiogen-medium`, **cc-by-nc-4.0**):
//! safetensors → GGUF conversion (Wave 5 residual, 2026-08-01).
//!
//! Input: the upstream `facebook/audiogen-medium` release — Meta AudioCraft's
//! 1.5B-parameter text-to-audio autoregressive transformer LM (Kreuk et al.
//! 2023, arXiv:2209.15352 "AudioGen: Textually Guided Audio Generation").
//! AudioGen is a **MusicGen sibling** — identical topology (transformer LM
//! over EnCodec RVQ tokens conditioned on frozen T5 text encoder), tuned
//! on environmental sounds / SFX (dog barking, footsteps, glass breaking,
//! ambient noise) rather than music. Only the training data + optional
//! stereo head differ; the arch is shared under the same `musicgen` tag.
//!
//! # Vokra scope — audio generation (per 2026-07-30 scope expansion)
//!
//! Only OSS text-to-SFX generator with public weights (bark handles vocal
//! SFX, this covers non-vocal environmental audio). Shares the `music`
//! category tag with MusicGen family (audio-generation taxonomy tree =
//! text-to-audio 全般、silently sharing `tts` would misroute).
//!
//! # License posture — CC-BY-NC 4.0 (**NonCommercial**)
//!
//! Same license posture as sibling MusicGen family: `LicenseClass::
//! NonCommercial` fail-closed default, `publish-one.sh --allow-noncommercial`
//! required at publish time. X-Codec 2 (2026-07-28) + MusicGen-Medium
//! (2026-08-01) T4 precedent 継承.
//!
//! # Scale — local convert OK (~3.7 GB)
//!
//! AudioGen-Medium ships as ~3.7 GB on HF — well below the M1 iMac 16 GB
//! local-convert threshold (memory [[feedback-large-models-on-vast-ai]]:
//! ≥8 GB safe threshold), so conversion + publish can happen locally.
//! Contrast MusicGen-Medium 11.4 GB / MusicGen-Large 19.5 GB which require
//! vast.ai handoff.
//!
//! # No ONNX (permanent)
//!
//! AudioGen ships safetensors + PyTorch pickle; this converter **never**
//! touches ONNX (FR-LD-05).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for AudioGen GGUFs.
///
/// Shared with MusicGen family — the topology is byte-parallel identical
/// (transformer LM over EnCodec RVQ tokens + T5 text encoder). Only the
/// training corpus differs (music vs environmental sounds), which does
/// not change the runtime dispatch surface.
pub const ARCH: &str = "musicgen";

/// `vokra.model.name` — distinct spelling within the shared arch (mirror
/// of snac_24khz / snac_44khz sibling posture).
pub const NAME: &str = "audiogen-medium";

pub const CATEGORY: &str = "music";

pub const UPSTREAM_HF: &str = "facebook/audiogen-medium";

pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-4.0";

const UPSTREAM_SOURCE: &str =
    "facebook/audiogen-medium (Meta AudioCraft 1.5B text-to-audio LM, cc-by-nc-4.0)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of an AudioGen-Medium conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AudioGenMediumReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

/// Converts a `facebook/audiogen-medium` safetensors checkpoint at
/// `input` into a Vokra-native GGUF at `output`, returning a
/// [`AudioGenMediumReport`].
///
/// Same conversion posture as sibling `convert_musicgen_medium_file`:
/// F32 / F16 / BF16 tensors pass through verbatim; non-float tensors are
/// skipped defensively.
pub fn convert_audiogen_medium_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<AudioGenMediumReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::NonCommercial),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_SOURCE));
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = AudioGenMediumReport::default();
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
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

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
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
            "vokra-convert-audiogen-medium-{tag}-{}-{n}",
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
        let st = safetensors_one("audio_encoder.embed", "F32", &[2, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_audiogen_medium_file(&inp, &outp, None).unwrap();
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);
        assert_eq!(r.bf16_passthrough, 0);

        let g = GgufFile::open(&outp).unwrap();
        // Mirror of styletts2 `get_string` helper — the sibling metadata
        // read pattern (`GgufMetadataValue::as_str()` returns
        // `Option<&str>` for string-typed keys).
        let read_str = |key: &str| -> String {
            g.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{key}: missing"))
                .to_owned()
        };
        assert_eq!(read_str(chunks::KEY_MODEL_ARCH), ARCH);
        assert_eq!(read_str(chunks::KEY_MODEL_NAME), NAME);
        assert_eq!(read_str(KEY_MODEL_CATEGORY), CATEGORY);
        assert_eq!(read_str(KEY_PROVENANCE_UPSTREAM_HF), UPSTREAM_HF);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let payload: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one("lm.attn.qkv", "BF16", &[2, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_audiogen_medium_file(&inp, &outp, None).unwrap();
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);
        assert_eq!(r.bf16_passthrough, 1);
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
        convert_audiogen_medium_file(&inp, &outp, Some("mit")).unwrap();
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
