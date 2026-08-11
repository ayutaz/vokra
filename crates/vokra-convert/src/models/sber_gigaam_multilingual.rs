//! **sber-gigaam-multilingual** (Sber GigaAM multilingual ASR):
//! safetensors checkpoint → GGUF conversion
//! (coverage-audit-2026-08-03 Wave B).
//!
//! Input: the upstream `salute-developers/GigaAM` release on GitHub
//! (multilingual variant; a mirror `ai-sage/GigaAM-Multilingual` on
//! Hugging Face is documented in the audit ticket as
//! "要 mirror URL 確認", so the converter treats the GitHub release as
//! the primary redistribution source and stamps
//! `vokra.provenance.upstream_url` — the sibling nkf-aec / htdemucs /
//! frcrn / nsnet2 / rnnoise convention for GitHub-only releases).
//! The multilingual variant is the 2026 SberDevices release covering
//! 70+ languages via a shared Conformer + char-wise CTC head — the
//! architecture is the same Conformer + CTC family the sibling
//! `sber-gigaam-v3` (Russian-specific fine-tune, planned as its own
//! ticket in the same wave) targets; the *vocabulary* is what
//! differs (70+ language char space vs Russian-only). This converter
//! is therefore a **standalone `ModelKind::SberGigaamMultilingual`
//! today**; a future refactor MAY absorb both siblings under a
//! single `ModelKind::GigaAm` with a variant enum (`_v3` /
//! `_multilingual` — the sibling ticket's audit note), but that
//! refactor is deliberately deferred so this Wave B ticket can land
//! independently of the sibling's implementation (worktree
//! isolation, per the coverage-audit workflow).
//!
//! Output: a GGUF carrying every float tensor plus the
//! `vokra.model.*` and `vokra.provenance.*` metadata chunks the
//! runtime ASR path will bind against once the runtime side of the
//! family lands (Conformer encoder + CTC decode via the shared
//! `vokra_ops::conformer` + `vokra_ops::ctc_decode` primitives the
//! Parakeet-CTC / omniASR-CTC / Canary siblings already exercise —
//! no new op is introduced by this converter).
//!
//! # License
//!
//! - SPDX: **MIT** ([`vokra_core::LicenseClass::Permissive`]) — verified
//!   against the upstream `github.com/salute-developers/GigaAM/LICENSE`
//!   per the audit ticket (`docs/tickets/coverage-audit-2026-08-03/
//!   wave-b/sber-gigaam-multilingual.md`).
//! - Category: **asr** (multilingual ASR, 70+ language char-wise CTC —
//!   the audit's short-form category name; the "70+lang" refinement
//!   lives in the ticket / model card, not in the runtime dispatch tag).
//! - Notes: the audit ticket flags a `training-data-litigation-risk:
//!   medium-high` (70+ language corpus provenance = Common Voice / MLS /
//!   VoxPopuli / FLEURS mixture is not yet chain-audited) — that
//!   sign-off is the owner's job in `docs/license-audit.md §3.1`, not
//!   the converter's; the converter stamps the SPDX id it was told and
//!   nothing more.
//!
//! # BF16 pass-through (mirror of nkf_aec / speaker_3d / ecapa_tdnn /
//! # qwen3_tts / vibevoice / voxcpm2 / moshi)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm on the same
//! match arm — no convert-time widening. BF16 is emitted as GGUF type
//! 30 (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`SberGigaamMultilingualReport::bf16_passthrough`] guards
//! against a silent widen / downcast regression.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream state-dict keys verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / speaker_3d /
//! ecapa_tdnn / nkf_aec contract; the prepare-checkpoint sidecar
//! preserves the dotted state-dict keys). Real-weight parity binding
//! to the runtime `vokra-models` module is deferred to owner sign-off
//! per `docs/license-audit.md §3.1`.
//!
//! # Provenance key choice: `vokra.provenance.upstream_url`
//!
//! Unlike the majority of sibling converters which stamp
//! `vokra.provenance.upstream_hf = <org>/<repo>` (the HF hub is the
//! primary redistribution source for `speechbrain/…`, `openbmb/…`,
//! `nvidia/…`, `microsoft/…`, `ResembleAI/…`, `iic/…`), sber-gigaam-
//! multilingual's primary redistribution source per the audit ticket
//! is the **GitHub release** at `github.com/salute-developers/GigaAM`
//! — the ticket flags the HF mirror `ai-sage/GigaAM-Multilingual` as
//! "要 mirror URL 確認" (proprietary-mirror-risk medium: Sber has
//! maintained HF mirrors independently in the past). The converter
//! therefore stamps [`KEY_PROVENANCE_UPSTREAM_URL`] (following the
//! Wave A convention for non-HF-verified sources: `nkf-aec` /
//! `ten-vad` / `htdemucs-4s-6s` / `frcrn` / `nsnet2` /
//! `dnsmos-p808-p835` / `openwakeword-op` / `torchaudio-squim` /
//! `rnnoise-v0.2`). Once the mirror is owner-verified, a follow-up
//! commit may add `vokra.provenance.upstream_hf` alongside the URL.
//!
//! # No ONNX (permanent)
//!
//! The upstream Sber GigaAM release ships PyTorch pickle files
//! only; this converter **never** touches ONNX (FR-LD-05). The
//! `tools/parity/sber_gigaam_multilingual_prepare_checkpoint.py`
//! sidecar bridges the upstream `.pt` to safetensors offline.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for sber-gigaam-multilingual GGUFs.
/// Intentionally **distinct** from a hypothetical future
/// `sber_gigaam_v3` (the sibling Russian-only fine-tune) — the two
/// share the same Conformer + CTC family topology but differ in
/// vocabulary size (70+ language char space vs Russian-only), which
/// the runtime CTC bind will need to distinguish; silently sharing an
/// arch tag would mis-route the runtime dispatch once either half's
/// runtime binding lands. A future refactor that absorbs both under
/// `gigaam` + a variant hparam would first need a runtime-side
/// binder that reads the variant.
pub const ARCH: &str = "gigaam_multilingual";

/// `vokra.model.name` value written for the canonical
/// `salute-developers/GigaAM` (multilingual variant) release.
pub const NAME: &str = "sber-gigaam-multilingual";

/// `vokra.model.category` value written for every sber-gigaam-
/// multilingual GGUF.
///
/// The audit's category label is "asr/70+lang" (multilingual ASR);
/// the shorter `asr` variant is used here so runtime dispatch and
/// model-card grouping stay uniform with the existing `asr` family
/// (whisper / kotoba_whisper / parakeet-ctc / canary / omniasr-ctc /
/// distil-whisper) and do not multiply category labels by
/// vocabulary-size distinctions the arch tag already carries.
pub const CATEGORY: &str = "asr";

/// Primary redistribution source (SberDevices' GitHub repository —
/// the HF mirror `ai-sage/GigaAM-Multilingual` is documented in the
/// audit ticket as "要 mirror URL 確認"). Written under
/// [`KEY_PROVENANCE_UPSTREAM_URL`].
pub const UPSTREAM_URL: &str = "github.com/salute-developers/GigaAM";

/// Default upstream weight licence (SPDX). Verified against
/// `github.com/salute-developers/GigaAM/blob/main/LICENSE`
/// (standard MIT) per the audit ticket.
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// convention the sibling BF16-passthrough converters use applies).

/// `vokra.model.category` metadata key. Local per the established
/// funcodec / wespeaker / speaker_3d / ecapa_tdnn / nkf_aec precedent
/// (not yet centralized in `vokra-core::gguf::chunks`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_url` — the primary redistribution source
/// URL for models whose canonical release is NOT on the Hugging Face
/// hub. Parallel to `vokra.provenance.upstream_hf` (the HF-hosted
/// sibling key); the Wave A tickets established the split so the
/// model-card generator can distinguish "there is an HF mirror" from
/// "the source is a raw URL". Local per the same convention as
/// [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Outcome of a sber-gigaam-multilingual conversion.
///
/// Mirrors the sibling BF16-passthrough converters' counter shape
/// (`super::nkf_aec::NkfAecReport`,
/// `super::ecapa_tdnn::EcapaTdnnReport`,
/// `super::speaker_3d::Speaker3dReport`) — adds `read` tracking
/// every tensor the safetensors reader surfaced so the invariant
/// `read == written + skipped_non_float` is auditable at the report
/// level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SberGigaamMultilingualReport {
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

/// Converts a sber-gigaam-multilingual safetensors checkpoint at
/// `input` (as emitted by
/// `tools/parity/sber_gigaam_multilingual_prepare_checkpoint.py`)
/// into a Vokra-native GGUF at `output`, returning a
/// [`SberGigaamMultilingualReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict key; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id /
/// source / upstream_url) chunks are stamped for the runtime
/// compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw
/// SPDX string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"mit"`, `Permissive`) — the upstream
/// GitHub release ships MIT.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_sber_gigaam_multilingual_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SberGigaamMultilingualReport, ConvertError> {
    // Load the whole checkpoint into memory: the ticket's size
    // estimate is ~600 M params / ~1.2 GB safetensors — well below
    // the streaming-mandated tier (Moshi 14 GiB), so the simple
    // `std::fs::read` posture the sibling non-streaming converters
    // (nkf_aec / ecapa_tdnn / speaker_3d / qwen3_tts) use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // Default provenance stamp — Permissive MIT (upstream
    // `github.com/salute-developers/GigaAM/LICENSE`). The optional
    // `license` argument overrides below via the same restated-source
    // convention as the sibling converters.
    let effective_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some(
            "salute-developers/GigaAM (multilingual variant, char-wise CTC, \
             70+ languages, MIT)",
        ),
    );

    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted
    // BF16-passthrough ADR the sibling non-streaming converters
    // (nkf_aec / speaker_3d / ecapa_tdnn / qwen3_tts / vibevoice /
    // voxcpm2 / moshi) share; the runtime widens BF16 → f32 exactly
    // at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    let mut report = SberGigaamMultilingualReport::default();
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
    /// `name` so concurrent runs do not collide).
    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-sber-gigaam-multilingual-{tag}-{}-{}.{ext}",
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
    /// converter's `convert_sber_gigaam_multilingual_file` file → file
    /// round-trip with its dtype preserved (`GgmlType::BF16`, GGUF
    /// type 30) and its payload byte-identical. Mirrors
    /// `nkf_aec::tests::bf16_tensor_passes_through_verbatim` /
    /// `ecapa_tdnn::tests::bf16_tensor_passes_through_verbatim`. A
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
        let header = r#"{"encoder.layers.0.conv_module.pointwise_conv1.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_sber_gigaam_multilingual_file(&input_path, &output_path, None)
            .expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of nkf_aec / ecapa_tdnn / speaker_3d)"
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
            .tensor_info("encoder.layers.0.conv_module.pointwise_conv1.weight")
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

    /// Pins that F32 and F16 tensors both ride the pass-through arm
    /// in the same conversion (mixed-dtype loops don't collapse to
    /// one arm), and that the BF16 counter stays at its `Default 0`
    /// when no BF16 tensor is present (additive-field regression
    /// guard). Also confirms the provenance chunks and the
    /// `upstream_url` key land at the top level as strings the
    /// model-card generator can lift verbatim.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        // Two tensors in one safetensors file:
        //   encoder.linear.weight — F32, [1, 2] →  8 bytes @ [0..8)
        //   encoder.linear.bias   — F16, [2]    →  4 bytes @ [8..12)
        // Both dtypes must reach the pass-through arm and neither
        // must increment `bf16_passthrough`.
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000]; // 1.0, 2.0 in IEEE half.
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);

        let header = format!(
            r#"{{"encoder.linear.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"encoder.linear.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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

        let report = convert_sber_gigaam_multilingual_file(&input_path, &output_path, None)
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
            .tensor_info("encoder.linear.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(f32_info.dimensions, vec![1, 2]);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file.tensor_info("encoder.linear.bias").expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(f16_info.dimensions, vec![2]);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Provenance stamped through the default (MIT / Permissive).
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
            file.get(KEY_PROVENANCE_UPSTREAM_URL)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_URL)
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

    /// Pins the license override boundary: passing `Some(spdx)`
    /// replaces both the raw SPDX string and the re-derived
    /// `LicenseClass`, keeping the GGUF the single source of truth
    /// the model card is generated from (no card / artifact drift).
    /// Mirrors the outer `convert_file_licensed` override contract at
    /// the top-level lib.rs boundary.
    #[test]
    fn license_override_replaces_default() {
        // Minimal single-F32-tensor safetensors buffer — the license
        // override contract is independent of tensor shape / count.
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = r#"{"encoder.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("lic-in", "safetensors");
        let output_path = scratch_path("lic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        // Override the MIT default with apache-2.0 — both remain
        // Permissive, so the LicenseClass rederivation is a no-op;
        // the SPDX string is what changes.
        let report =
            convert_sber_gigaam_multilingual_file(&input_path, &output_path, Some("apache-2.0"))
                .expect("convert with override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "override replaces the raw SPDX string"
        );
        // Both MIT and apache-2.0 map to Permissive, so this stays
        // Permissive — asserting explicitly guards against a
        // rederivation regression that dropped the license → class
        // step.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );
    }
}
