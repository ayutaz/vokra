//! **3D-Speaker ERes2Net** speaker encoder: safetensors checkpoint →
//! GGUF conversion (2026-07-25).
//!
//! Input: the upstream `iic/speech_eres2net_sv_zh-cn_16k-common`
//! release on Hugging Face (mirror of the ModelScope 3D-Speaker
//! project). Output: a GGUF carrying every float tensor plus the
//! `vokra.model.*` / `vokra.provenance.*` metadata chunks the
//! runtime speaker path binds against.
//!
//! # License
//!
//! - SPDX: **apache-2.0** ([`vokra_core::LicenseClass::Permissive`]).
//! - Category: **speaker** (encoder-only, speaker verification /
//!   diarization; sibling of CAM++ in the Vokra speaker family).
//! - Notes: 3D-Speaker ERes2Net (ModelScope).
//!
//! # BF16 posture
//!
//! F32 / F16 / BF16 all pass through **verbatim** on the same match arm
//! — the qwen3-tts / vibevoice / voxcpm2 / moshi pattern. BF16 is
//! emitted as GGUF type 30 (`GgmlType::BF16`) with no convert-time
//! widening; the runtime widens BF16 → f32 losslessly at load via the
//! single choke point `crates/vokra-core/src/gguf/quant/mod.rs
//! decode_bf16` (BF16 is the top 16 bits of an f32 — `bits << 16` is
//! exact). The counter [`Speaker3dReport::bf16_passthrough`] records
//! how many BF16 tensors landed on this arm, mirror of
//! `moshi::MoshiReport::bf16_passthrough` /
//! `qwen3_tts::Qwen3TtsReport::bf16_passthrough` /
//! `vibevoice::VibeVoiceReport::bf16_passthrough`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VibeVoice
//! contract). Real-weight parity binding is deferred to owner sign-off
//! per `docs/license-audit.md` §3.1.
//!
//! # No ONNX (permanent)
//!
//! The upstream 3D-Speaker release ships PyTorch / safetensors + a
//! Python pipeline; this converter **never** touches ONNX (FR-LD-05).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for 3D-Speaker ERes2Net GGUFs.
pub(crate) const ARCH: &str = "speaker_3d";

/// `vokra.model.name` value for the canonical
/// `iic/speech_eres2net_sv_zh-cn_16k-common` release.
pub(crate) const NAME: &str = "speech_eres2net_sv_zh-cn_16k-common";

/// Category marker written as `vokra.model.category`. Vokra's runtime
/// uses this string to route the artifact to its speaker-encoder
/// pipeline (siblings: CAM++).
pub(crate) const CATEGORY: &str = "speaker";

/// Upstream Hugging Face repo id (self-describing provenance).
pub(crate) const UPSTREAM_HF: &str = "iic/speech_eres2net_sv_zh-cn_16k-common";

/// Default weight license SPDX for the canonical release.
pub(crate) const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

/// `vokra.model.category` metadata key.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` — the Hugging Face repo the weight
/// came from (informational; readable by the model card / catalog
/// tooling).
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a 3D-Speaker conversion.
#[derive(Debug, Default)]
pub struct Speaker3dReport {
    /// Total tensors seen in the input safetensors (before filtering).
    /// Equal to `written + skipped_non_float` on any well-formed
    /// checkpoint the safetensors reader accepts.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling converters).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — mirrors
    /// `qwen3_tts::Qwen3TtsReport::bf16_passthrough` /
    /// `vibevoice::VibeVoiceReport::bf16_passthrough` /
    /// `voxcpm2::VoxCpm2Report::bf16_passthrough`. Guards against a
    /// silent BF16 → F32 widen regression.
    pub bf16_passthrough: usize,
}

/// Converts a 3D-Speaker ERes2Net safetensors checkpoint at `input`
/// into a GGUF written to `output`.
///
/// - Every F32 / F16 / BF16 tensor passes through **verbatim** under
///   its upstream safetensors name.
/// - The `vokra.model.*` + `vokra.provenance.*` chunks stamp the arch
///   (`speaker_3d`), name, category (`speaker`), upstream HF repo, and
///   weight license.
/// - `license` = `None` uses the canonical `apache-2.0`
///   ([`DEFAULT_LICENSE_SPDX`]). Passing `Some(spdx)` overrides both
///   the raw SPDX string and the re-derived [`LicenseClass`], keeping
///   the GGUF the single source of truth the model card is generated
///   from (no card / artifact drift — mirrors the
///   `convert_file_licensed` override at the top-level lib.rs
///   boundary).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_speaker_3d_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Speaker3dReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Category marker so the runtime + model-card tooling can route the
    // GGUF to the speaker-encoder pipeline (siblings: CAM++).
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    // Informational provenance: the upstream Hugging Face repo the
    // weight came from. Read by the catalog / publishability tooling.
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // License stamp: default is the canonical apache-2.0 (Permissive)
    // grant on `iic/speech_eres2net_sv_zh-cn_16k-common`. `license =
    // Some(spdx)` overrides both the raw SPDX string and the re-derived
    // `LicenseClass` — same override contract as
    // `convert_file_licensed` at the top-level lib.rs boundary
    // (the artifact IS the single source of truth the model card is
    // generated from, so a caller redistributing under a different
    // upstream SPDX must override here rather than post-hoc mutate).
    let effective_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some("iic/speech_eres2net_sv_zh-cn_16k-common (3D-Speaker ERes2Net, ModelScope)"),
    );

    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30); the runtime widens BF16 → f32
    // exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // qwen3-tts / vibevoice / voxcpm2 / moshi.
    let mut report = Speaker3dReport::default();
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
    use vokra_core::gguf::GgufFile;

    /// Writes `bytes` to a fresh unique file under [`std::env::temp_dir`]
    /// (mirrors the moshi streaming-test pattern). The caller is
    /// responsible for deleting the file when done — tests use a
    /// [`TempFileGuard`] to clean up on Drop.
    fn temp_path(tag: &str, ext: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-speaker3d-{}-{}-{}.{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
            ext
        ));
        p
    }

    /// RAII cleanup so failing tests do not leak temp files on disk
    /// (best-effort — a panic mid-cleanup is fine).
    struct TempFileGuard(std::path::PathBuf);
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Builds a minimal safetensors buffer carrying a single BF16 tensor
    /// with a deterministic non-zero payload (so a silent widen /
    /// downcast trips a byte-identity assert later).
    fn bf16_tensor_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Six BF16 elements, shape [2, 3] → 12 bytes payload. Non-zero
        // bit patterns catch any silent widen (values would still be
        // "correct" after a widen, so we also assert on dtype and raw
        // bytes below — three concentric fences).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload = bf16_tensor_bytes(&values);
        assert_eq!(payload.len(), 12);
        let header = r#"{"encoder.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = temp_path("bf16-in", "safetensors");
        let output_path = temp_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report =
            convert_speaker_3d_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of qwen3-tts / vibevoice)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        // Round-trip through the emitted GGUF: dtype preserved, payload
        // byte-identical (no convert-time widening).
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("encoder.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            payload.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );
    }

    #[test]
    fn f32_and_f16_tensors_pass_through() {
        // Two tensors in one safetensors file:
        //   encoder.weight — F32, [1, 2] →  8 bytes @ [0..8)
        //   encoder.bias   — F16, [2]    →  4 bytes @ [8..12)
        // Both dtypes must reach the pass-through arm and neither must
        // increment `bf16_passthrough`.
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000]; // 1.0, 2.0 in IEEE half.
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);

        let header = format!(
            r#"{{"encoder.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"encoder.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
        );
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16_bytes);

        let input_path = temp_path("mixed-in", "safetensors");
        let output_path = temp_path("mixed-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_speaker_3d_file(&input_path, &output_path, None)
            .expect("convert F32 + F16 mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2, "F32 and F16 must both pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 / F16 must NOT increment the BF16 counter"
        );

        // Both tensors survive the round trip with dtype + bytes intact.
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let f32_info = file.tensor_info("encoder.weight").expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(f32_info.dimensions, vec![1, 2]);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file.tensor_info("encoder.bias").expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(f16_info.dimensions, vec![2]);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Provenance stamped through the default (apache-2.0 Permissive).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
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
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
    }
}
