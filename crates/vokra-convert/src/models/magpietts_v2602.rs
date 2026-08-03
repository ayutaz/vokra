//! **NVIDIA MagpieTTS v2 (v2602)**: safetensors checkpoint → GGUF
//! conversion (coverage-audit-2026-08-03 Wave B).
//!
//! Input: the upstream `nvidia/magpietts-v2602` release on HuggingFace —
//! a **multilingual TTS** (9 languages) release from NVIDIA NeMo. The
//! upstream distribution ships as a `.nemo` NGC container (tar / tar.gz
//! of a YAML config + a torch pickle `.ckpt`); callers pre-flatten it
//! to safetensors offline through
//! `tools/parity/magpietts_v2602_prepare_checkpoint.py` (the DAC / DFN3
//! / CSM / Canary / Parakeet pickle-bridge pattern — pickles never
//! enter the runtime, FR-LD-05).
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`
//! and `vokra.provenance.*` metadata chunks the runtime TTS path binds
//! against.
//!
//! # License
//!
//! - SPDX: **apache-2.0** ([`vokra_core::LicenseClass::Permissive`]).
//! - Category: **tts** — the model is a TTS release (the audit ticket
//!   groups it as `tts-multilingual-9lang`; the shorter `tts` variant
//!   is used here so runtime dispatch and model-card grouping stay
//!   uniform with the existing TTS family and do not multiply category
//!   labels by language-count distinctions the arch tag already
//!   carries).
//! - Notes: audit ticket `docs/tickets/coverage-audit-2026-08-03/
//!   wave-b/magpietts-v2602.md` cites the ~700 MB size, Apache-2.0
//!   license, and 9-language multilingual coverage; the upstream
//!   distribution path (`nvidia/magpietts-v2602` on Hugging Face; a
//!   parallel NGC mirror may also exist) is recorded verbatim in the
//!   `vokra.provenance.upstream_hf` chunk so a downstream consumer
//!   can trace the artifact back to its serving location.
//!
//! # BF16 pass-through (mirror of speaker_3d / ecapa_tdnn / qwen3_tts /
//! # voxcpm2 / vibevoice / moshi / emotion2vec / wespeaker / frcrn)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm on the same
//! match arm — no convert-time widening. BF16 is emitted as GGUF type
//! 30 (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`MagpiettsV2602Report::bf16_passthrough`] guards against
//! a silent widen / downcast regression.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / WeSpeaker / emotion2vec / FRCRN contract; the
//! prepare-checkpoint sidecar preserves the dotted state-dict keys the
//! NeMo `.nemo` container exposes). Real-weight parity binding to the
//! runtime `vokra-models::magpietts_v2602` module (native TTS forward)
//! is deferred to owner sign-off per `docs/license-audit.md §3.1`.
//!
//! # No ONNX (permanent)
//!
//! The upstream MagpieTTS release ships NeMo torch-pickle checkpoints;
//! this converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in a future
//! `crates/vokra-models/src/magpietts_v2602/` module (whisper.cpp 型
//! self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for MagpieTTS-v2602 GGUFs. Intentionally
/// distinct from every sibling TTS arch tag — silently sharing (e.g.)
/// `cosyvoice2` / `piper_plus` / `sbv2` / `qwen3_tts` would misroute
/// the runtime dispatch (a HiFTChain-based loader would try to
/// interpret a MagpieTTS checkpoint with a completely different
/// topology). Snake-case per the local convention (the CLI slug uses
/// the hyphenated form via [`crate::ModelKind::as_arg`]).
pub const ARCH: &str = "magpietts_v2602";

/// `vokra.model.name` value written for the canonical
/// `nvidia/magpietts-v2602` release.
pub const NAME: &str = "magpietts-v2602";

/// `vokra.model.category` value written for every MagpieTTS-v2602 GGUF.
///
/// The audit ticket's category label is "tts-multilingual-9lang"; the
/// shorter `tts` variant is used here so runtime dispatch and model-
/// card grouping stay uniform with the existing `tts` family (Kokoro /
/// piper-plus / CosyVoice2/3 / Chatterbox / Qwen3-TTS / VoxCPM /
/// VibeVoice / Irodori / VITS-JA / SBV2 / Dia / Zonos) and do not
/// multiply category labels by language-count distinctions the arch
/// tag already carries.
pub const CATEGORY: &str = "tts";

/// Upstream distribution slug on Hugging Face, recorded under
/// `vokra.provenance.upstream_hf` so a downstream consumer can trace
/// the artifact back to its serving location without parsing the
/// free-text `vokra.provenance.source` blob.
pub const UPSTREAM_HF: &str = "nvidia/magpietts-v2602";

/// Default upstream weight licence (SPDX). Apache-2.0 per the audit
/// ticket's Publish gate `redistributable` entry.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// convention the sibling BF16-passthrough converters use applies).

/// `vokra.model.category` metadata key. Local per the established
/// funcodec / wespeaker / speaker_3d / ecapa_tdnn / emotion2vec /
/// frcrn / nkf_aec precedent (not yet centralized in
/// `vokra-core::gguf::chunks`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` — the Hugging Face repository slug
/// the release ships from. Local per the wespeaker / frcrn / speaker_3d
/// convention (parallel to `vokra.provenance.upstream_url` for
/// non-HF sources).
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a MagpieTTS-v2602 conversion.
///
/// Mirrors the sibling BF16-passthrough converters' counter shape
/// ([`super::frcrn::FrcrnReport`], [`super::emotion2vec::Emotion2vecReport`],
/// [`super::wespeaker::WespeakerReport`]) — adds `read` tracking every
/// tensor the safetensors reader surfaced so the invariant
/// `read == written + skipped_non_float` is auditable at the report
/// level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MagpiettsV2602Report {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for parity with the sibling converters).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens BF16
    /// → f32 losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. A silent
    /// widen / downcast regression would surface as this counter
    /// drifting away from the input BF16 count.
    pub bf16_passthrough: usize,
}

/// Converts a MagpieTTS-v2602 safetensors checkpoint at `input` (as
/// emitted by `tools/parity/magpietts_v2602_prepare_checkpoint.py`)
/// into a Vokra-native GGUF at `output`, returning a
/// [`MagpiettsV2602Report`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict key; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_hf) chunks are stamped for the runtime compliance gate
/// (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// [`DEFAULT_LICENSE_SPDX`] (`"apache-2.0"`, `Permissive`) — the
/// upstream release is Apache-2.0 per the audit ticket. A downstream
/// repackager may pass e.g. `Some("apache-2.0")` verbatim to make the
/// stamp explicit even without a licence change (mirror of the
/// `wespeaker` / `emotion2vec` / `frcrn` override convention).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_magpietts_v2602_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MagpiettsV2602Report, ConvertError> {
    // Whole-file read: MagpieTTS-v2602 is ~700 MB per the audit ticket
    // — comfortably below the streaming threshold the Moshi 15 GB /
    // Voxtral 8.7 GB converters run. Any future 2B+ MagpieTTS sibling
    // would swap this call for `SafetensorsFileReader::open` +
    // `GgufStreamWriter::begin` per the moshi.rs / qwen3_tts.rs ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough).
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = apache-2.0 (upstream `nvidia/magpietts-v2602`
    // Apache-2.0 per the audit ticket's Publish gate). The optional
    // `license` argument overrides via the same restated-source
    // convention as the sibling converters (`wespeaker` / `frcrn` /
    // `emotion2vec` / `speaker_3d`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some("nvidia/magpietts-v2602 (NeMo multilingual TTS — 9 languages, apache-2.0)"),
    );

    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted BF16-passthrough
    // ADR the sibling non-streaming converters (speaker_3d / ecapa_tdnn /
    // qwen3_tts / vibevoice / voxcpm2 / moshi / wespeaker / emotion2vec /
    // frcrn / nkf_aec) share; the runtime widens BF16 → f32 exactly at
    // load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    let mut report = MagpiettsV2602Report::default();
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

    // Serialize and land the emitted GGUF at `output`. `to_bytes()`
    // stamps `vokra.schema.version` + `vokra.schema.producer` on its own
    // via the writer's built-in schema stamper — no per-converter
    // duplication needed.
    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch path (PID + nanos + a suffix derived from
    /// the caller — every test in this module uses a distinct `name` so
    /// concurrent runs do not collide).
    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-magpietts-v2602-{tag}-{}-{}.{ext}",
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
    /// converter's `convert_magpietts_v2602_file` file → file round-trip
    /// with its dtype preserved (`GgmlType::BF16`, GGUF type 30) and
    /// its payload byte-identical. Mirrors
    /// `frcrn::tests::bf16_tensor_passes_through_verbatim` /
    /// `nkf_aec::tests::bf16_tensor_passes_through_verbatim` /
    /// `emotion2vec::tests::bf16_tensor_passes_through_verbatim`. A
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
        let header = r#"{"encoder.embed_tokens.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
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
            convert_magpietts_v2602_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of frcrn / nkf_aec / emotion2vec)"
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
            .tensor_info("encoder.embed_tokens.weight")
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

    /// Pins that F32 and F16 tensors both ride the pass-through arm in
    /// the same conversion (mixed-dtype loops don't collapse to one arm),
    /// and that the BF16 counter stays at its `Default 0` when no BF16
    /// tensor is present (additive-field regression guard).
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        // Two tensors in one safetensors file:
        //   text_encoder.linear.weight — F32, [1, 2] →  8 bytes @ [0..8)
        //   text_encoder.linear.bias   — F16, [2]    →  4 bytes @ [8..12)
        // Both dtypes must reach the pass-through arm and neither must
        // increment `bf16_passthrough`.
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000]; // 1.0, 2.0 in IEEE half.
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);

        let header = format!(
            r#"{{"text_encoder.linear.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"text_encoder.linear.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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

        let report = convert_magpietts_v2602_file(&input_path, &output_path, None)
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
            .tensor_info("text_encoder.linear.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(f32_info.dimensions, vec![1, 2]);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file
            .tensor_info("text_encoder.linear.bias")
            .expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(f16_info.dimensions, vec![2]);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Provenance stamped through the default (apache-2.0 / Permissive).
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
        let header = r#"{"decoder.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("lic-in", "safetensors");
        let output_path = scratch_path("lic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        // Override the apache-2.0 default with mit — both remain
        // Permissive, so the LicenseClass rederivation is a no-op; the
        // SPDX string is what changes.
        let report = convert_magpietts_v2602_file(&input_path, &output_path, Some("mit"))
            .expect("convert with override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "override replaces the raw SPDX string"
        );
        // Both apache-2.0 and mit map to Permissive, so this stays
        // Permissive — asserting explicitly guards against a rederivation
        // regression that dropped the license → class step.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );
    }
}
