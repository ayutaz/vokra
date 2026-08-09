//! **ECAPA-TDNN** (SpeechBrain speaker encoder): safetensors checkpoint →
//! GGUF conversion (SoTA plan, 2026-07-25).
//!
//! Input: the upstream `speechbrain/spkrec-ecapa-voxceleb` release — an
//! ECAPA-TDNN 192-dim speaker embedding model trained on VoxCeleb 1+2.
//! Output: a GGUF carrying every float tensor plus `vokra.model.*` and
//! `vokra.provenance.*` metadata identifying the model as a `speaker`
//! category weight with an `apache-2.0` licence.
//!
//! # Provenance
//!
//! - **HF path**: `speechbrain/spkrec-ecapa-voxceleb`.
//! - **License (SPDX)**: `apache-2.0` — end-to-end (SpeechBrain code +
//!   trained weight; see `docs/license-audit.md §3.1` sign-off queue).
//! - **Category**: `speaker` (speaker encoder / embedding extractor —
//!   fbank-80 → 192-d embedding, alternate realisation of the same
//!   functional surface as `campplus.rs`). Category tag is written under
//!   the raw `vokra.model.category` key so the model-card tooling can
//!   classify without reaching into per-converter constants.
//!
//! # BF16 pass-through (mirror of qwen3_tts / vibevoice / voxcpm2)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm — no
//! convert-time widening. BF16 stays GGUF type 30 (`GgmlType::BF16`); the
//! runtime widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is
//! the top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`EcapaTdnnReport::bf16_passthrough`] records how many BF16
//! tensors landed on this arm so a silent widen / downcast cannot slip
//! in undetected.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VibeVoice /
//! VoxCPM contract). Real-weight parity is deferred to owner sign-off in
//! `docs/license-audit.md §3.1`.
//!
//! # No ONNX (permanent)
//!
//! SpeechBrain ships PyTorch checkpoints (safetensors); this converter
//! **never** touches ONNX (FR-LD-05).
//!
//! # Wiring status
//!
//! The module is a **landing pad**: [`convert_ecapa_tdnn_file`] is `pub`
//! but not yet re-exported at the crate root or dispatched from
//! [`crate::convert_file_licensed`] (a follow-up commit wires the
//! `ModelKind::EcapaTdnn` arm + `vokra-cli` selector). Until that
//! commit lands the constants below are only reached through this
//! module's own tests, so a module-level `#![allow(dead_code)]` prevents
//! the "never used" warnings from tripping the workspace's
//! `-D warnings` clippy gate. Removing the attribute is the tell-tale
//! for the wiring commit landing (`ModelKind::EcapaTdnn` will reach
//! [`convert_ecapa_tdnn_file`] from `convert_file_licensed`).

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for ECAPA-TDNN GGUFs — intentionally **distinct**
/// from `campplus` because ECAPA-TDNN and CAM++ share a functional
/// surface (fbank-80 → 192-d embedding) but NOT their tensor topology
/// (ECAPA-TDNN uses SE-Res2Blocks + attentive stat pooling; CAM++ uses
/// D-TDNN with context-aware masking). Silently sharing an arch tag
/// would mis-route runtime dispatch.
pub const ARCH: &str = "ecapa_tdnn";

/// `vokra.model.name` value written for the canonical
/// `speechbrain/spkrec-ecapa-voxceleb` GGUF.
pub const NAME: &str = "spkrec-ecapa-voxceleb";

/// `vokra.model.category` value written for every ECAPA-TDNN GGUF.
pub const CATEGORY: &str = "speaker";

/// `vokra.provenance.upstream_hf` value — the primary redistribution
/// source used by the model-card generator.
pub const UPSTREAM_HF: &str = "speechbrain/spkrec-ecapa-voxceleb";

/// Default upstream weight licence (SPDX).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication rule
// the sibling converters use applies).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of an ECAPA-TDNN conversion.
///
/// Mirrors the sibling converters' counter shape
/// (`super::qwen3_tts::Qwen3TtsReport`, `super::vibevoice::VibeVoiceReport`,
/// `super::voxcpm2::VoxCpm2Report`) adapted to the file-oriented
/// `convert_ecapa_tdnn_file` surface (adds `read` tracking every tensor
/// the safetensors reader surfaced so the invariant
/// `read == written + skipped_non_float` is auditable).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EcapaTdnnReport {
    /// Total tensors surfaced by the safetensors reader (before any
    /// dispatch to the pass-through / skipped arm).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so a non-zero
    /// here would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — a latent
    /// silent widen / downcast cannot slip in undetected without this
    /// counter also drifting.
    pub bf16_passthrough: usize,
}

/// Converts a `speechbrain/spkrec-ecapa-voxceleb` safetensors checkpoint
/// at `input` into a Vokra-native GGUF at `output`, returning an
/// [`EcapaTdnnReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream name;
/// the `vokra.model.*` (arch / name / category) and `vokra.provenance.*`
/// (weight_license / license / model_id / source / upstream_hf) chunks
/// are stamped for the runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"apache-2.0"`, `Permissive`) — the upstream
/// HF release ships apache-2.0 end-to-end.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input.
pub fn convert_ecapa_tdnn_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<EcapaTdnnReport, ConvertError> {
    // Load the whole checkpoint into memory: the ECAPA-TDNN release is
    // ~15 MiB (192-d embedding backbone) — 1-2 orders of magnitude
    // smaller than the streaming-mandated Moshi 14 GiB tier, so the
    // simple `std::fs::read` posture the sibling non-streaming
    // converters (qwen3_tts / vibevoice / voxcpm2) use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Default provenance stamp — Permissive apache-2.0 end-to-end
    // (upstream `speechbrain/spkrec-ecapa-voxceleb` model card + repo
    // LICENSE). The optional `license` argument overrides below.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        DEFAULT_LICENSE_SPDX,
        Some(NAME),
        Some("speechbrain/spkrec-ecapa-voxceleb (apache-2.0 end-to-end)"),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = EcapaTdnnReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (mirror of qwen3_tts / vibevoice / voxcpm2 / moshi); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
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

    // Optional weight-license override — mirrors the outer
    // `convert_file_licensed` (lib.rs) branch so both a Vokra-CLI caller
    // and a direct `convert_ecapa_tdnn_file` caller land the same
    // provenance surface for the same SPDX string. Restates the source
    // neutrally so it does not contradict the stamped default's
    // parenthetical.
    if let Some(lic) = license {
        let class = LicenseClass::from_license_str(lic);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, class.as_str());
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, lic);
        b.add_string(
            chunks::KEY_PROVENANCE_SOURCE,
            &format!("{UPSTREAM_HF} (licence {lic} per source)"),
        );
    }

    // Serialize and land the emitted GGUF at `output`. `to_bytes()`
    // stamps `vokra.schema.version` + `vokra.schema.producer` on its
    // own via the writer's built-in schema stamper — no per-converter
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

    /// Per-test unique scratch path (PID + a suffix derived from the
    /// caller — every test in this module uses a distinct `name` so
    /// concurrent runs do not collide).
    fn scratch_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-ecapa-tdnn-{name}-{}.tmp",
            std::process::id()
        ));
        p
    }

    /// Builds a minimal single-BF16-tensor safetensors buffer and returns
    /// `(safetensors_bytes, raw_bf16_payload)` so a downstream test can
    /// assert byte-identity on the payload after the GGUF round-trip.
    fn safetensors_one_bf16() -> (Vec<u8>, Vec<u8>) {
        // Non-zero bit patterns so a silent widen / downcast could not
        // round-trip trivially.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16");
        let header = r#"{"embedding_model.blocks.0.tdnn.conv.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        input.extend_from_slice(&bf16);
        (input, bf16)
    }

    /// Builds a mixed F32 + F16 safetensors buffer. Header layout:
    ///   `embedding_model.a.weight` — F32, `[2,3]` → 24 bytes @ [0..24)
    ///   `embedding_model.b.weight` — F16, `[2,3]` → 12 bytes @ [24..36)
    fn safetensors_f32_and_f16() -> Vec<u8> {
        let f32_vals: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Non-zero F16 half-precision bit patterns (1.0 = 0x3C00, 2.0 = 0x4000, …).
        let f16_patterns: [u16; 6] = [0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 24);
        assert_eq!(f16_bytes.len(), 12);
        let header = r#"{"embedding_model.a.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"embedding_model.b.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[24,36]}}"#;
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        input.extend_from_slice(&f32_bytes);
        input.extend_from_slice(&f16_bytes);
        input
    }

    /// Pins the BF16 pass-through end-to-end: the tensor survives the
    /// converter's `convert_ecapa_tdnn_file` file → file round-trip with
    /// its dtype preserved (`GgmlType::BF16`, GGUF type 30) and its
    /// payload byte-identical. Mirrors
    /// `qwen3_tts::tests::bf16_tensor_passes_through_verbatim` at the
    /// file-oriented surface. A silent widen at convert time would
    /// still round-trip _values_ (the BF16 → f32 widen is exact), so
    /// this test asserts on the dtype AND the raw bytes — two concentric
    /// fences.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (bytes, bf16_payload) = safetensors_one_bf16();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        // Cleanest posture even on early panic: overwrite / remove
        // regardless of prior test state.
        std::fs::write(&input, &bytes).expect("write input");

        let report = convert_ecapa_tdnn_file(&input, &output, None).expect("convert");

        let out_bytes = std::fs::read(&output).expect("read output");
        // Best-effort cleanup — a failed test still surfaces the assert.
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();

        assert_eq!(report.read, 1, "one input tensor surfaced");
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );

        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("embedding_model.blocks.0.tdnn.conv.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16_payload.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );
    }

    /// Pins that F32 and F16 tensors both ride the pass-through arm in
    /// the same conversion (mixed-dtype loops don't collapse to one
    /// arm), and that the BF16 counter stays at its `Default 0` when no
    /// BF16 tensor is present (additive-field regression guard).
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let bytes = safetensors_f32_and_f16();
        let input = scratch_path("mixed-in");
        let output = scratch_path("mixed-out");
        std::fs::write(&input, &bytes).expect("write input");

        let report = convert_ecapa_tdnn_file(&input, &output, None).expect("convert");

        let out_bytes = std::fs::read(&output).expect("read output");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();

        assert_eq!(report.read, 2, "two input tensors surfaced");
        assert_eq!(report.written, 2, "F32 + F16 both pass through");
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32-only + F16-only input must leave the BF16 counter at Default 0"
        );
        assert_eq!(report.skipped_non_float, 0);

        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let a = file
            .tensor_info("embedding_model.a.weight")
            .expect("F32 tensor present");
        assert_eq!(a.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(a.dimensions, vec![2, 3]);
        let b = file
            .tensor_info("embedding_model.b.weight")
            .expect("F16 tensor present");
        assert_eq!(b.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(b.dimensions, vec![2, 3]);
    }
}
