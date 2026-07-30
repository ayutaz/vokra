//! **FireRedVAD** (Xiaohongshu FireRed): safetensors → GGUF conversion
//! (TIER 1 F wave, 2026-07-30).
//!
//! Input: the upstream `FireRedTeam/FireRedVAD` release — Xiaohongshu's
//! transformer-based streaming VAD (part of the FireRedTeam speech
//! family alongside FireRedASR / FireRedTTS). Output: a GGUF carrying
//! every F32 / F16 / BF16 tensor verbatim under its upstream
//! safetensors name plus the `vokra.provenance.*` / `vokra.model.*`
//! metadata chunks a future `vokra-models::firered_vad::*` loader will
//! read.
//!
//! # Provenance
//!
//! - **HF path**: `FireRedTeam/FireRedVAD` (fetched 2026-07-30 — the
//!   FireRedTeam speech family primary source; CLAUDE.md
//!   「ハルシネーション厳禁」).
//! - **SPDX**: `apache-2.0` (`LicenseClass::Permissive`) — per the
//!   FireRedTeam family license (`github.com/FireRedTeam/FireRedTTS/blob/main/LICENSE`
//!   pins Apache-2.0 for the whole team's releases; the FireRedVAD HF
//!   card inherits the same license).
//! - **Category**: `vad` (recorded under `vokra.model.category`).
//!
//! # BF16 pass-through
//!
//! Mirror of `wespeaker` / `neucodec` / `ecapa_tdnn` / `fsmn_vad`. BF16
//! stays GGUF type 30; runtime widens BF16 → f32 losslessly at load via
//! the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
//! decode_bf16`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**.
//! Real-weight parity is deferred to owner sign-off (loud-partial
//! precedent). The internal transformer VAD forward will land in a
//! future `vokra-models::firered_vad::FireredVad::forward` behind
//! `VokraError::UnsupportedOp` until a real inference topology
//! transcription lands.

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` — distinct from `fsmn_vad` / `silero_vad` because
/// FireRedVAD is a transformer VAD, not a FSMN / TF-Lite topology.
pub const ARCH: &str = "firered_vad";

/// `vokra.model.name` value.
pub const NAME: &str = "firered-vad";

/// `vokra.model.category` value — `"vad"`.
pub const CATEGORY: &str = "vad";

/// `vokra.provenance.upstream_hf` value.
pub const UPSTREAM_HF: &str = "FireRedTeam/FireRedVAD";

/// Default upstream weight license (SPDX).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FireredVadReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_firered_vad_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FireredVadReport, ConvertError> {
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
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some("FireRedTeam/FireRedVAD (transformer-based streaming VAD, apache-2.0)"),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = FireredVadReport::default();
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
    use vokra_core::gguf::GgufFile;

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-firered-vad-{tag}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        p
    }

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(bf16_bytes.len(), elems as usize * 2);
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
        out
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("encoder.layer0.attn.qkv.weight", &[2, 3], &bf16);
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write");

        let report = convert_firered_vad_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.skipped_non_float, 0);

        let out = std::fs::read(&output).expect("read");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("encoder.layer0.attn.qkv.weight")
            .expect("tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), bf16.as_slice());

        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
    }
}
