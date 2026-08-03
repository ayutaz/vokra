//! **NKF-AEC** (Neural Kalman Filter Acoustic Echo Canceller): safetensors
//! checkpoint → GGUF conversion (coverage-audit-2026-08-03 Wave A).
//!
//! Input: the upstream `fjiang9/NKF-AEC` release on GitHub — a Neural
//! Kalman Filter AEC (Yang et al. ICASSP 2023, arXiv:2207.11388,
//! "Low-complexity Acoustic Echo Cancellation with Neural Kalman
//! Filtering"), 5.3 KB `.pt` at
//! `github.com/fjiang9/NKF-AEC/blob/main/pretrained/nkf.pt`. The upstream
//! release ships as a torch-pickle `.pt`; callers pre-flatten it to
//! safetensors offline through
//! `tools/parity/nkf_aec_prepare_checkpoint.py` (the DFN3 / DAC /
//! Kokoro / CSM pickle-bridge pattern — pickles never enter the
//! runtime, FR-LD-05).
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`
//! and `vokra.provenance.*` metadata chunks the runtime AEC path binds
//! against.
//!
//! # License
//!
//! - SPDX: **MIT** ([`vokra_core::LicenseClass::Permissive`]).
//! - Category: **aec** (Neural Echo Cancellation; sibling of the
//!   algorithmic AEC op family — [M4-03 `vokra_aec_*`] SpeexDSP /
//!   WebRTC AEC3 Rust port — so `nkf-aec` is a *neural* alternative
//!   placed alongside the algorithmic baseline, not a replacement).
//! - Notes: the audit ticket (`docs/tickets/coverage-audit-2026-08-03/
//!   wave-a/nkf-aec.md`) cites Yang et al. 2023 ICASSP; upstream repo
//!   LICENSE = standard MIT with `Copyright (c) 2022 Fei Jiang`.
//!
//! # BF16 pass-through (mirror of speaker_3d / ecapa_tdnn / qwen3_tts /
//! # voxcpm2 / vibevoice / moshi)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm on the same
//! match arm — no convert-time widening. BF16 is emitted as GGUF type
//! 30 (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`NkfAecReport::bf16_passthrough`] guards against a silent
//! widen / downcast regression.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream torch-pickle keys verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / speaker_3d /
//! ecapa_tdnn contract; the prepare-checkpoint sidecar preserves the
//! dotted state-dict keys). Real-weight parity binding to the runtime
//! `vokra-models::nkf_aec` module (native GEMV forward — the ticket's
//! "5.3 KB ゆえ single-pass GEMV で完結" note) is deferred to owner
//! sign-off per `docs/license-audit.md §3.1`.
//!
//! # Provenance key choice: `vokra.provenance.upstream_url`
//!
//! Unlike the vast majority of sibling converters which stamp
//! `vokra.provenance.upstream_hf = <org>/<repo>` (the HF hub is the
//! primary redistribution source for `speechbrain/…`, `openbmb/…`,
//! `nvidia/…`, `microsoft/…`, `ResembleAI/…`, `iic/…`), NKF-AEC's
//! primary redistribution source is the **GitHub release** at
//! `github.com/fjiang9/NKF-AEC` — the author ships no HF mirror. The
//! converter therefore stamps [`KEY_PROVENANCE_UPSTREAM_URL`] instead,
//! following the parallel key naming convention the Wave A tickets
//! established for non-HF sources (`ten-vad` /
//! `htdemucs-4s-6s` / `frcrn` / `nsnet2` / `dnsmos-p808-p835` /
//! `openwakeword-op` / `torchaudio-squim` / `rnnoise-v0.2`).
//!
//! # No ONNX (permanent)
//!
//! The upstream NKF-AEC release ships PyTorch pickle files only; this
//! converter **never** touches ONNX (FR-LD-05).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for NKF-AEC GGUFs. Intentionally distinct
/// from every sibling AEC / denoise family — the ticket's audit places
/// NKF-AEC as a first-class **neural** AEC alternative to the
/// algorithmic M4-03 `vokra_aec_*` (SpeexDSP / WebRTC AEC3 Rust port),
/// so silently sharing an arch tag with any denoise / algorithmic-AEC
/// sibling would mis-route the runtime dispatch.
pub const ARCH: &str = "nkf_aec";

/// `vokra.model.name` value written for the canonical
/// `fjiang9/NKF-AEC` release.
pub const NAME: &str = "nkf-aec";

/// `vokra.model.category` value written for every NKF-AEC GGUF.
///
/// The audit's category label is "aec / NEC" (acoustic echo cancellation
/// / neural echo cancellation); the shorter `aec` variant is used here
/// so runtime dispatch and model-card grouping stay uniform with the
/// existing `aec` family and do not multiply category labels by
/// neural-vs-algorithmic distinctions the arch tag already carries.
pub const CATEGORY: &str = "aec";

/// Primary redistribution source (author's GitHub repository — there is
/// no HF mirror). Written under [`KEY_PROVENANCE_UPSTREAM_URL`].
pub const UPSTREAM_URL: &str = "github.com/fjiang9/NKF-AEC";

/// Default upstream weight licence (SPDX). Verified against
/// `github.com/fjiang9/NKF-AEC/blob/main/LICENSE` (standard MIT).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// convention the sibling BF16-passthrough converters use applies).

/// `vokra.model.category` metadata key. Local per the established
/// funcodec / wespeaker / speaker_3d / ecapa_tdnn precedent (not yet
/// centralized in `vokra-core::gguf::chunks`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_url` — the primary redistribution source
/// URL for models whose canonical release is NOT on the Hugging Face
/// hub. Parallel to `vokra.provenance.upstream_hf` (the HF-hosted
/// sibling key); the Wave A tickets established the split so the
/// model-card generator can distinguish "there is an HF mirror" from
/// "the source is a raw URL". Local per the same convention as
/// [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Outcome of an NKF-AEC conversion.
///
/// Mirrors the sibling BF16-passthrough converters' counter shape
/// ([`super::ecapa_tdnn::EcapaTdnnReport`],
/// [`super::speaker_3d::Speaker3dReport`]) — adds `read` tracking every
/// tensor the safetensors reader surfaced so the invariant
/// `read == written + skipped_non_float` is auditable at the report
/// level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NkfAecReport {
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

/// Converts an NKF-AEC safetensors checkpoint at `input` (as emitted by
/// `tools/parity/nkf_aec_prepare_checkpoint.py`) into a Vokra-native
/// GGUF at `output`, returning an [`NkfAecReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// pickle key; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_url) chunks are stamped for the runtime compliance gate
/// (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// [`DEFAULT_LICENSE_SPDX`] (`"mit"`, `Permissive`) — the upstream
/// GitHub release ships MIT with `Copyright (c) 2022 Fei Jiang`.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_nkf_aec_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<NkfAecReport, ConvertError> {
    // Load the whole checkpoint into memory: the NKF-AEC release is 5.3
    // KB (the entire pretrained weight fits in a single L1 cache line
    // budget) — 6 orders of magnitude smaller than the streaming-mandated
    // Moshi 14 GiB tier, so the simple `std::fs::read` posture the sibling
    // non-streaming converters (ecapa_tdnn / speaker_3d / qwen3_tts) use
    // applies trivially.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // Default provenance stamp — Permissive MIT (upstream
    // `github.com/fjiang9/NKF-AEC/LICENSE`, `Copyright (c) 2022 Fei
    // Jiang`). The optional `license` argument overrides below via the
    // same restated-source convention as the sibling converters.
    let effective_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some("github.com/fjiang9/NKF-AEC (Neural Kalman Filter AEC, Yang et al. ICASSP 2023)"),
    );

    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted BF16-passthrough
    // ADR the sibling non-streaming converters (speaker_3d / ecapa_tdnn /
    // qwen3_tts / vibevoice / voxcpm2 / moshi) share; the runtime widens
    // BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    let mut report = NkfAecReport::default();
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

    /// Per-test unique scratch path (PID + nanos + a suffix derived from
    /// the caller — every test in this module uses a distinct `name` so
    /// concurrent runs do not collide).
    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-nkf-aec-{tag}-{}-{}.{ext}",
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
    /// converter's `convert_nkf_aec_file` file → file round-trip with
    /// its dtype preserved (`GgmlType::BF16`, GGUF type 30) and its
    /// payload byte-identical. Mirrors
    /// `ecapa_tdnn::tests::bf16_tensor_passes_through_verbatim` /
    /// `speaker_3d::tests::bf16_tensor_passes_through_verbatim`. A
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
        let header =
            r#"{"kf.gru.weight_ih_l0":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_nkf_aec_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of ecapa_tdnn / speaker_3d)"
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
            .tensor_info("kf.gru.weight_ih_l0")
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
        //   kf.linear.weight — F32, [1, 2] →  8 bytes @ [0..8)
        //   kf.linear.bias   — F16, [2]    →  4 bytes @ [8..12)
        // Both dtypes must reach the pass-through arm and neither must
        // increment `bf16_passthrough`.
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000]; // 1.0, 2.0 in IEEE half.
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);

        let header = format!(
            r#"{{"kf.linear.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"kf.linear.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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

        let report =
            convert_nkf_aec_file(&input_path, &output_path, None).expect("convert F32 + F16 mixed");
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
        let f32_info = file.tensor_info("kf.linear.weight").expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(f32_info.dimensions, vec![1, 2]);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file.tensor_info("kf.linear.bias").expect("F16 tensor");
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
        let header = r#"{"kf.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
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
        // Permissive, so the LicenseClass rederivation is a no-op; the
        // SPDX string is what changes.
        let report = convert_nkf_aec_file(&input_path, &output_path, Some("apache-2.0"))
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
        // Permissive — asserting explicitly guards against a rederivation
        // regression that dropped the license → class step.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );
    }
}
