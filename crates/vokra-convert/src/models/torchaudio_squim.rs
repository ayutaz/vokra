//! **torchaudio Squim** (`pytorch/audio`, BSD-2-Clause):
//! safetensors → GGUF conversion (coverage-audit-2026-08-03 Wave A permissive
//! continuation, 2026-08-04).
//!
//! Input: the upstream `torchaudio.prototype.pipelines.SQUIM_OBJECTIVE` +
//! `SQUIM_SUBJECTIVE` bundle from PyTorch torchaudio (Kumar et al. 2023
//! ICASSP arXiv:2304.01448, "TorchAudio-Squim: Reference-less Speech
//! Quality and Intelligibility Measures in TorchAudio"). SQUIM is a
//! **reference-free** multi-metric speech quality estimator — single-pass
//! STOI + PESQ + SI-SDR (`Objective`) and MOS (`Subjective`) prediction
//! from a degraded waveform alone (no clean reference). The upstream
//! release is distributed via `torch.hub` (no HF mirror) — this converter
//! consumes safetensors pre-flattened offline by
//! `tools/parity/torchaudio_squim_prepare_checkpoint.py` (the DFN3 / DAC /
//! Kokoro / CSM pickle-bridge pattern — pickles never enter the runtime,
//! FR-LD-05).
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`
//! and `vokra.provenance.*` metadata chunks the runtime eval path binds
//! against.
//!
//! # License
//!
//! - SPDX: **BSD-2-Clause** ([`vokra_core::LicenseClass::Permissive`]).
//! - Category: **eval** (reference-free speech quality — sibling of the
//!   existing `utmos` / `dnsmos` families, complementary because SQUIM
//!   emits 4 metrics in a single pass and UTMOS / DNSMOS each emit one).
//! - Notes: upstream repo LICENSE = standard BSD-2-Clause covering the
//!   torchaudio project (`github.com/pytorch/audio/blob/main/LICENSE`).
//!
//! # BF16 pass-through (mirror of nkf_aec / facebook_denoiser /
//! # sensevoicesmall / neucodec)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm on the same
//! match arm — no convert-time widening. BF16 is emitted as GGUF type
//! 30 ([`GgmlType::BF16`]); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`TorchaudioSquimReport::bf16_passthrough`] guards against a
//! silent widen / downcast regression.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream torch state-dict keys verbatim**
//! (`ssl_encoder.*` / `head.*` — the sibling BF16-passthrough contract;
//! the prepare-checkpoint sidecar preserves the dotted state-dict keys).
//! Real-weight parity binding to a future `vokra-eval::squim` module
//! (native SSL encoder + multi-head regression forward) is deferred to
//! owner sign-off per `docs/license-audit.md §3.1`.
//!
//! # Provenance key choice: `vokra.provenance.upstream_url`
//!
//! Unlike the majority of sibling converters which stamp
//! `vokra.provenance.upstream_hf = <org>/<repo>` (the HF hub is the
//! primary redistribution source for `speechbrain/…`, `openbmb/…`,
//! `nvidia/…`, `microsoft/…`, `ResembleAI/…`, `iic/…`), torchaudio
//! Squim's primary redistribution source is the **PyTorch torchaudio
//! GitHub repository** at `github.com/pytorch/audio` (the SQUIM
//! pipeline itself is distributed via `torch.hub` — a raw URL system,
//! not HF). The converter therefore stamps
//! [`KEY_PROVENANCE_UPSTREAM_URL`] instead, following the parallel key
//! naming convention the Wave A tickets established for non-HF sources
//! (`nkf-aec` / `ten-vad` / `facebook-denoiser`).
//!
//! # No ONNX (permanent)
//!
//! The upstream torchaudio release ships PyTorch pickle files
//! (`.pt` / `.pth`); this converter **never** touches ONNX
//! (FR-LD-05).
//!
//! # Wiring status
//!
//! This is the TDD skeleton (BF16 / F16 / F32 pass-through plus
//! provenance / category stamps). The runtime native SSL encoder + 4-
//! metric multi-head regression head forward is a follow-up wave,
//! deferred to owner sign-off (see `docs/license-audit.md` §3.1).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for torchaudio Squim GGUFs. Intentionally
/// distinct from every sibling eval / quality-metric arch tag
/// (`utmos` / `dnsmos` / `nisqa_v2_weight` / `utmosv2`) — SQUIM's
/// reference-free 4-metric multi-head architecture is a distinct
/// topology, so silently sharing would mis-route the runtime dispatch.
pub const ARCH: &str = "torchaudio_squim";

/// `vokra.model.name` value written for the canonical
/// `pytorch/audio` SQUIM release.
pub const NAME: &str = "torchaudio_squim";

/// `vokra.model.category` value written for every torchaudio Squim GGUF.
/// Sibling of `utmos` / `dnsmos` / `nisqa_v2_weight` (`eval` family).
pub const CATEGORY: &str = "eval";

/// Primary redistribution source (PyTorch torchaudio GitHub repository —
/// SQUIM ships via `torch.hub`, no HF mirror). Written under
/// [`KEY_PROVENANCE_UPSTREAM_URL`].
pub const UPSTREAM_URL: &str = "github.com/pytorch/audio";

/// Default upstream weight licence (SPDX). Verified against the
/// upstream `github.com/pytorch/audio/blob/main/LICENSE`
/// (standard BSD-2-Clause).
pub const DEFAULT_LICENSE_SPDX: &str = "bsd-2-clause";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// convention the sibling BF16-passthrough converters use applies).

/// `vokra.model.category` metadata key. Local per the established
/// sensevoicesmall / nkf_aec / funcodec convention (not yet centralized
/// in `vokra-core::gguf::chunks`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_url` — the primary redistribution source
/// URL for models whose canonical release is NOT on the Hugging Face
/// hub. Parallel to `vokra.provenance.upstream_hf` (the HF-hosted
/// sibling key); the Wave A tickets established the split so the
/// model-card generator can distinguish "there is an HF mirror" from
/// "the source is a raw URL". Local per the same convention as
/// [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Outcome of a torchaudio Squim conversion.
///
/// Mirrors the sibling BF16-passthrough converters' counter shape
/// (`super::nkf_aec::NkfAecReport`,
/// `super::facebook_denoiser::FacebookDenoiserReport`) — the
/// invariant `read == written + skipped_non_float` is auditable at the
/// report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TorchaudioSquimReport {
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

/// Converts a torchaudio Squim safetensors checkpoint at `input`
/// (pre-flattened from the upstream torch bundles by
/// `tools/parity/torchaudio_squim_prepare_checkpoint.py`) into a
/// Vokra-native GGUF at `output`, returning a
/// [`TorchaudioSquimReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict key; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_url) chunks are stamped for the runtime compliance gate
/// (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"bsd-2-clause"`, `Permissive`) — the
/// upstream torchaudio repo ships BSD-2-Clause.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_torchaudio_squim_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<TorchaudioSquimReport, ConvertError> {
    // Load the whole checkpoint into memory — the SQUIM bundles are
    // ~60 MB each (~120 MB combined), well below the streaming-mandated
    // Moshi 14 GiB tier, so the simple `std::fs::read` posture the
    // sibling non-streaming converters use applies trivially.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    let effective_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some(
            "github.com/pytorch/audio (torchaudio SQUIM reference-free speech quality \
             estimator: SQUIM_OBJECTIVE = STOI + PESQ + SI-SDR + SQUIM_SUBJECTIVE = MOS, \
             Kumar et al. 2023 ICASSP arXiv:2304.01448, BSD-2-Clause)",
        ),
    );

    let mut report = TorchaudioSquimReport::default();
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
    use vokra_core::gguf::GgufFile;

    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-torchaudio-squim-{tag}-{}-{}.{ext}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        p
    }

    struct TempFileGuard(PathBuf);
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Pins the BF16 pass-through end-to-end: the tensor survives the
    /// converter's `convert_torchaudio_squim_file` file → file round-trip
    /// with its dtype preserved ([`GgmlType::BF16`], GGUF type 30) and
    /// its payload byte-identical. Mirrors
    /// `nkf_aec::tests::bf16_tensor_passes_through_verbatim`. A silent
    /// widen at convert time would still round-trip _values_ (BF16 →
    /// f32 widen is exact), so this test asserts on the dtype AND the
    /// raw bytes — two concentric fences.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload = bf16_bytes(&values);
        assert_eq!(payload.len(), 12, "6 elements × 2 bytes BF16");
        // SQUIM Objective SSL encoder attention Q-projection weight —
        // the upstream torch state-dict convention preserved verbatim
        // through the `torchaudio_squim_prepare_checkpoint.py` bridge.
        let header = r#"{"ssl_encoder.layers.0.attn.wq.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
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
            convert_torchaudio_squim_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of nkf_aec / facebook_denoiser)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("ssl_encoder.layers.0.attn.wq.weight")
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
    /// the same conversion (mixed-dtype loops don't collapse to one
    /// arm), and that the BF16 counter stays at its `Default 0` when no
    /// BF16 tensor is present (additive-field regression guard). Also
    /// asserts the arch / name / category / provenance stamps land
    /// through the default (bsd-2-clause → [`LicenseClass::Permissive`])
    /// code path.
    #[test]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_permissive() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000]; // 1.0, 2.0 in IEEE half.
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);

        let header = format!(
            r#"{{"ssl_encoder.layers.0.norm.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"objective_head.stoi.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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

        let report = convert_torchaudio_squim_file(&input_path, &output_path, None)
            .expect("convert F32 + F16 mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2, "F32 and F16 must both pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 / F16 must NOT increment the BF16 counter"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let f32_info = file
            .tensor_info("ssl_encoder.layers.0.norm.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(f32_info.dimensions, vec![1, 2]);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file
            .tensor_info("objective_head.stoi.bias")
            .expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(f16_info.dimensions, vec![2]);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

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
            Some(LicenseClass::Permissive.as_str()),
            "bsd-2-clause must resolve to Permissive (T1 tier)"
        );
    }

    /// Pins the license override boundary: passing `Some(spdx)` replaces
    /// both the raw SPDX string and the re-derived [`LicenseClass`],
    /// keeping the GGUF the single source of truth the model card is
    /// generated from (no card / artifact drift). Mirrors the outer
    /// `convert_file_licensed` override contract at the top-level lib.rs
    /// boundary.
    #[test]
    fn license_override_replaces_default() {
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header =
            r#"{"ssl_encoder.embed.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("lic-in", "safetensors");
        let output_path = scratch_path("lic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        // Override the BSD-2-Clause default with apache-2.0 — both remain
        // Permissive, so the LicenseClass rederivation is a no-op; the
        // SPDX string is what changes.
        let report = convert_torchaudio_squim_file(&input_path, &output_path, Some("apache-2.0"))
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
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 stays Permissive (same class as bsd-2-clause default)",
        );
    }
}
