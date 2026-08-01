#![allow(clippy::doc_lazy_continuation)]
//! **openWakeWord** (`dscripka/openWakeWord`, **apache-2.0**):
//! safetensors → GGUF conversion (Wave residual, 2026-08-02).
//!
//! openWakeWord (`github.com/dscripka/openWakeWord`, Apache-2.0) is a
//! small custom-KWS (keyword-spotting / wake-word) family — a shallow
//! MLP / CNN over pre-computed melspectrogram features from a shared
//! (Google speech_embedding TFLite) frontend. Each wake-word is a
//! separate tiny checkpoint (~1–5 MB) reusing the same feature
//! extractor. This converter is the audio-dialect `kws` op entry
//! (FR-OP `kws`) — a distinct arch tag `openwakeword`, category `kws`.
//!
//! # License posture — Apache-2.0 (**Permissive**)
//!
//! `dscripka/openWakeWord` (GitHub) ships Apache-2.0 for both code and
//! bundled model checkpoints; the HF mirror at `dscripka/openWakeWord`
//! rate-limits the API (401) but the primary source (GitHub) is
//! Apache-2.0 verified. Default license is `apache-2.0` +
//! `LicenseClass::Permissive` (per the sibling Silero / CAM++ / piper-
//! plus first-party first-party Permissive posture). Override via
//! `--license <spdx>` at the outer boundary if a caller ships a
//! checkpoint under a different SPDX id.
//!
//! # Scale — local convert safe (~0.01 GB)
//!
//! Each openWakeWord checkpoint is 1–5 MB; the whole bundle
//! (~10 wake-words + speech_embedding frontend) is a handful of MB.
//! Well below the M1 iMac 16 GB safe local threshold per memory
//! `[[feedback-large-models-on-vast-ai]]` (≥8 GB is the vast.ai
//! cutoff). No vast.ai handoff required.
//!
//! # Runtime port
//!
//! Deferred. The converter lands the safetensors → GGUF bridge (BF16 /
//! F16 / F32 pass-through skeleton) so a Vokra runtime `kws` op can
//! consume the artifact. Upstream ships `.tflite` / `.onnx`
//! checkpoints; callers pre-flatten with an owner-side tool per the
//! sibling snac / bicodec / focalcodec pattern (a
//! `tools/parity/openwakeword_prepare_checkpoint.py` bridge is a
//! future WP).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "openwakeword";
pub const NAME: &str = "openwakeword";
pub const CATEGORY: &str = "kws";
pub const UPSTREAM_HF: &str = "dscripka/openWakeWord";
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

const UPSTREAM_SOURCE: &str =
    "dscripka/openWakeWord (custom-KWS MLP/CNN over precomputed melspec, apache-2.0)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct OpenWakeWordReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_openwakeword_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<OpenWakeWordReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_SOURCE));
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = OpenWakeWordReport::default();
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
            "vokra-convert-openwakeword-{tag}-{}-{n}",
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
    fn openwakeword_f32_tensor_passes_through_and_default_license_is_permissive() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let st = safetensors_one("mlp.dense.weight", "F32", &[2, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_openwakeword_file(&inp, &outp, None).unwrap();
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);
        assert_eq!(r.bf16_passthrough, 0);
        assert_eq!(r.skipped_non_float, 0);

        let g = GgufFile::open(&outp).unwrap();
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
        assert_eq!(read_str("vokra.provenance.license"), DEFAULT_LICENSE_SPDX);
        assert_eq!(g.tensors().len(), 1);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn openwakeword_bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one("mlp.dense.weight", "BF16", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_openwakeword_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);
        assert_eq!(r.written, 1);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn openwakeword_license_override_swaps_stamp() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();
        convert_openwakeword_file(&inp, &outp, Some("mit")).unwrap();
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get("vokra.provenance.license")
                .and_then(|v| v.as_str())
                .unwrap(),
            "mit"
        );
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
