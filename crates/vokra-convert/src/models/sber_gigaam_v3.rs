//! **Sber GigaAM v3** (`ai-sage/GigaAM-v3` family): safetensors → GGUF
//! conversion (coverage-audit-2026-08-03 Wave B).
//!
//! Input: the upstream `ai-sage/GigaAM-v3` release on Hugging Face — a
//! Sberbank AI Russian ASR model with a Conformer encoder + CTC or
//! RNN-T head. Sber ships the release as both a NeMo bundle and
//! flattened safetensors; callers can:
//!
//! - Pass a safetensors checkpoint from the HF mirror directly, or
//! - Pre-flatten the `.nemo` / `.ckpt` via
//!   `tools/parity/sber_gigaam_v3_prepare_checkpoint.py` (the DFN3 /
//!   Canary / Parakeet-family pickle-bridge pattern — pickles never
//!   enter the runtime, FR-LD-05).
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`
//! and `vokra.provenance.*` metadata chunks a future
//! `vokra-models::sber_gigaam_v3` native forward will read.
//!
//! # License / provenance
//!
//! - SPDX: **MIT** ([`vokra_core::LicenseClass::Permissive`]) — the
//!   Sberbank / SberDevices GigaAM release ships under an MIT LICENSE
//!   (permissive; no runtime-side attribution obligation, unlike the
//!   NVIDIA CC-BY 4.0 Parakeet / Canary tier).
//! - Upstream: recorded under
//!   `vokra.provenance.upstream_hf = "ai-sage/GigaAM-v3"` (HF mirror
//!   in the ai-sage collection).
//! - Category: **asr** — a Russian-focused ASR model. The narrower
//!   `asr-russian` label from the coverage-audit ticket is captured in
//!   the module doc + the model name ("gigaam-v3"); the
//!   `vokra.model.category` chunk uses the first-word convention
//!   (`asr`) so downstream category filters do not multiply into
//!   language-specific labels the arch tag already carries.
//! - Sign-off: real-weight parity + docs/license-audit.md §3.1 row is
//!   an owner follow-up (Wave B fast-track — the converter provides
//!   the byte-parallel GGUF surface only, and the runtime forward
//!   binding reuses the shared `vokra_ops::conformer` +
//!   `vokra_ops::ctc_decode` / `vokra_ops::rnnt_decode` primitives per
//!   the ticket's "Conformer + CTC/RNN-T seam 流用可" note).
//!
//! # BF16 pass-through (mirror of emotion2vec / wespeaker / neucodec /
//! # qwen3_tts / voxcpm2 / vibevoice)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm on the same
//! match arm — no convert-time widening. BF16 is emitted as GGUF type
//! 30 (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`SberGigaamV3Report::bf16_passthrough`] guards against a
//! silent widen / downcast regression.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream state-dict keys verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Parakeet / Canary
//! contract). Real-weight binding to a future
//! `vokra-models::sber_gigaam_v3::Weights::from_gguf` is a follow-up
//! wave gated on the upstream tensor-name manifest fetch + license
//! sign-off; this converter passes every F32 / F16 / BF16 tensor
//! through unchanged so the future loader can walk the same names.
//!
//! # No ONNX (permanent)
//!
//! Sber GigaAM is distributed as safetensors / a Python (NeMo)
//! pipeline; this converter **never** touches ONNX (FR-LD-05); the
//! pipeline will be re-implemented natively in a future
//! `crates/vokra-models/src/sber_gigaam_v3/` module (whisper.cpp 型
//! self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Sber GigaAM v3 GGUFs. Distinct from every
/// sibling ASR arch tag (parakeet-ctc / parakeet-tdt / canary /
/// omniasr-ctc / whisper / distil-whisper / kotoba-whisper / kyutai-stt)
/// because the Sber GigaAM training + tokenizer are Russian-first and
/// the runtime dispatch has to route to the Russian CTC / RNN-T decoder
/// stack; silently sharing an arch tag would misroute the runtime.
pub const ARCH: &str = "sber_gigaam_v3";

/// `vokra.model.name` value written for the canonical
/// `ai-sage/GigaAM-v3` release.
pub const NAME: &str = "gigaam-v3";

/// `vokra.model.category` value written for every Sber GigaAM v3 GGUF.
///
/// The audit's category label is "asr-russian"; the shorter `asr` is
/// used here so runtime dispatch and model-card grouping stay uniform
/// with the sibling ASR family (parakeet-ctc / canary / whisper /
/// distil-whisper / kotoba-whisper / kyutai-stt) and do not multiply
/// category labels by per-language distinctions the arch tag already
/// carries. The Russian focus is captured in the module doc + the
/// canonical model name.
pub const CATEGORY: &str = "asr";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf`. The ai-sage collection is the
/// canonical HF redistribution surface for the Sberbank / SberDevices
/// GigaAM family.
pub const UPSTREAM_HF: &str = "ai-sage/GigaAM-v3";

/// Default upstream weight licence (SPDX). The Sberbank / SberDevices
/// GigaAM release ships MIT (permissive, no runtime-side attribution).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the funcodec / wespeaker / speaker_3d /
// ecapa_tdnn / emotion2vec convention).

/// `vokra.model.category` metadata key. Local per the established
/// sibling convention (not yet centralized in `vokra-core::gguf::chunks`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` — the HF path the release ships from.
/// Parallel to the sibling non-HF key `vokra.provenance.upstream_url`
/// used by GitHub-only releases (nkf-aec / ten-vad / etc); this
/// converter stamps the HF variant because the ai-sage collection is
/// the primary redistribution surface.
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a Sber GigaAM v3 conversion.
///
/// Mirrors the sibling BF16-passthrough converters' counter shape
/// (`super::emotion2vec::Emotion2vecReport`,
/// `super::wespeaker::WespeakerReport`). Tracks every tensor the
/// safetensors reader surfaced so the invariant
/// `read == written + skipped_non_float` is auditable at the report
/// level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SberGigaamV3Report {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy path).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling reports).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; runtime widens BF16 →
    /// f32 losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. A silent
    /// widen / downcast regression would surface as this counter
    /// drifting away from the input BF16 count.
    pub bf16_passthrough: usize,
}

/// Converts a Sber GigaAM v3 safetensors checkpoint at `input` into a
/// Vokra-native GGUF at `output`, returning a
/// [`SberGigaamV3Report`].
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// state-dict key; the `vokra.model.*` (arch / name / category) +
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_hf) chunks are stamped for the runtime compliance gate
/// (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"mit"`, `Permissive`) — the standing
/// override mechanism for "implementation is clean-room but the
/// redistributed checkpoint carries a different SPDX" scenarios
/// (mirror of `convert_file_licensed` in `lib.rs`).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_sber_gigaam_v3_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SberGigaamV3Report, ConvertError> {
    // Whole-file read: the GigaAM v3 release is ~500 MB - 1.2 GB
    // (per coverage-audit ticket), well below the streaming-mandated
    // Moshi 14 GiB tier, so the simple `std::fs::read` posture the
    // sibling non-streaming converters (emotion2vec / wespeaker /
    // ecapa_tdnn / qwen3_tts) use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = MIT (upstream Sberbank / SberDevices GigaAM
    // release). The `license` override lets a downstream repackager
    // stamp a different SPDX if they redistribute under stricter terms
    // (the same knob `convert_file_licensed` exposes in `lib.rs`).
    let effective_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some(
            "ai-sage/GigaAM-v3 (Sberbank AI Russian ASR, Conformer encoder + CTC/RNN-T head, MIT)",
        ),
    );

    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30), same posture as emotion2vec /
    // wespeaker / qwen3_tts / vibevoice / voxcpm2 / neucodec; runtime
    // widens BF16 → f32 exactly at load via
    // `vokra-core::gguf::quant::decode_bf16` (`bits << 16` is exact).
    let mut report = SberGigaamV3Report::default();
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

    // Serialize and land the emitted GGUF at `output`. `to_bytes()`
    // stamps `vokra.schema.version` + `vokra.schema.producer` on its own
    // via the writer's built-in schema stamper — no per-converter
    // duplication needed.
    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch path (PID + nanos + a suffix derived
    /// from the caller — every test in this module uses a distinct
    /// `tag` so concurrent runs do not collide).
    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-sber-gigaam-v3-{tag}-{}-{}.{ext}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        p
    }

    /// RAII cleanup so failing tests do not leak temp files on disk
    /// (best-effort — a panic mid-cleanup is fine).
    struct TempFileGuard(PathBuf);
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Encodes `values` as BF16 (top 16 bits of each `f32`) little-endian.
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Pins the BF16 pass-through end-to-end: the tensor survives the
    /// converter's `convert_sber_gigaam_v3_file` file → file round-trip
    /// with its dtype preserved (`GgmlType::BF16`, GGUF type 30) and
    /// its payload byte-identical. Mirrors
    /// `emotion2vec::tests::bf16_tensor_passes_through_verbatim` /
    /// `wespeaker::tests::bf16_tensor_passes_through_verbatim`. A
    /// silent widen at convert time would still round-trip _values_
    /// (BF16 → f32 widen is exact), so this test asserts on the dtype
    /// AND the raw bytes — two concentric fences.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero bit patterns so a silent widen / downcast cannot
        // round-trip trivially.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload = bf16_bytes(&values);
        assert_eq!(payload.len(), 12, "6 elements × 2 bytes BF16");
        let header = r#"{"encoder.layers.0.self_attn.qkv_proj.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report =
            convert_sber_gigaam_v3_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of emotion2vec / wespeaker)"
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
            .tensor_info("encoder.layers.0.self_attn.qkv_proj.weight")
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

        // Provenance + category chunks pinned on the artifact itself.
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
            Some(CATEGORY),
            "category chunk records the first-word ASR tag"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF),
            "upstream_hf chunk pins the ai-sage HF redistribution surface"
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
        // Schema stamp is written unconditionally by the GGUF writer.
        assert!(
            file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
            "vokra.schema.version must be stamped"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
            "vokra.schema.producer must be stamped"
        );
    }

    /// Pins that F32 and F16 tensors both ride the pass-through arm in
    /// the same conversion (mixed-dtype loops don't collapse to one
    /// arm), and that the BF16 counter stays at its `Default 0` when
    /// no BF16 tensor is present (additive-field regression guard).
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        // Two tensors in one safetensors file:
        //   encoder.layers.0.linear.weight — F32, [1, 2] →  8 bytes
        //   decoder.head.bias              — F16, [2]    →  4 bytes
        // Both dtypes must reach the pass-through arm and neither must
        // increment `bf16_passthrough`.
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000]; // 1.0, 2.0 in IEEE half.
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);

        let header = format!(
            r#"{{"encoder.layers.0.linear.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"decoder.head.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
        );
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16_bytes);

        let input_path = scratch_path("mixed-in", "safetensors");
        let output_path = scratch_path("mixed-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_sber_gigaam_v3_file(&input_path, &output_path, None)
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
        let f32_info = file
            .tensor_info("encoder.layers.0.linear.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(f32_info.dimensions, vec![1, 2]);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file.tensor_info("decoder.head.bias").expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(f16_info.dimensions, vec![2]);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());
    }

    /// Pins the license override boundary: passing `Some(spdx)` replaces
    /// both the raw SPDX string and the re-derived `LicenseClass`,
    /// keeping the GGUF the single source of truth the model card is
    /// generated from (no card / artifact drift). Mirrors the outer
    /// `convert_file_licensed` override contract at the top-level lib.rs
    /// boundary.
    #[test]
    fn license_override_replaces_default() {
        // Minimal single-F32-tensor safetensors buffer — the license
        // override contract is independent of tensor shape / count.
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = r#"{"encoder.layers.0.linear.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("license-in", "safetensors");
        let output_path = scratch_path("license-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        // Override with `apache-2.0`: the raw SPDX string flips AND the
        // re-derived `LicenseClass` stays `Permissive` (both MIT and
        // Apache-2.0 map to Permissive), so the class is unchanged but
        // the SPDX string reflects the override. This is enough to pin
        // the override path — a stricter override (`cc-by-nc-4.0` →
        // `NonCommercial`) would additionally flip the class, but a
        // fixture that runs on the compliance gate has been intentionally
        // avoided here so the test does not fight the gate.
        let report = convert_sber_gigaam_v3_file(&input_path, &output_path, Some("apache-2.0"))
            .expect("convert with license override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "raw SPDX string must reflect the override"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "class stays Permissive (both MIT and Apache-2.0 map to Permissive)"
        );
    }
}
