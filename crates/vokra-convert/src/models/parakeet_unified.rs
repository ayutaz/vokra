//! NVIDIA **Parakeet-Unified-EN-0.6B**: safetensors checkpoint → GGUF
//! conversion (coverage-audit 2026-08-03 Wave B ticket).
//!
//! Input: an upstream `nvidia/parakeet-unified-en-0.6b` safetensors
//! checkpoint. The reference release may ship as a NeMo `.nemo` tarball
//! (NGC delivery format for the whole Parakeet family) — callers
//! pre-flatten that to safetensors offline via the shared
//! `tools/parity/nemo_pt_to_safetensors.py` bridge (the sibling
//! `parakeet` / `parakeet_ctc` / `canary` pattern; no per-model prep
//! script is needed because the flattener is arch-agnostic). Output: a
//! GGUF carrying every F32 / F16 / BF16 tensor verbatim plus the
//! `vokra.model.*` / `vokra.provenance.*` metadata chunks the future
//! native `vokra-models::parakeet_unified` loader will read.
//!
//! # What the "unified" tag means
//!
//! Parakeet-Unified-EN-0.6B **unifies** offline and streaming inference
//! into a single ~0.6B-parameter model with a FastConformer encoder
//! shared across both modes plus a built-in **punctuation +
//! capitalization** post-processor. Distinct from:
//!
//! - [`super::parakeet`] (Parakeet-TDT-0.6B-v3, offline TDT decoder
//!   only, arch tag `parakeet-tdt`).
//! - [`super::parakeet_ctc`] (Parakeet-CTC-1.1B, streaming CTC head
//!   only, arch tag `parakeet-ctc`).
//!
//! Silently sharing an arch tag with either sibling would mis-route the
//! runtime dispatch (FR-EX-08), since the unified model's tensor
//! topology carries the shape signatures of BOTH decoder families plus
//! the punc/cap head, and a `parakeet-tdt` loader would look for
//! `joint.*` tensors the CTC-only path never emits (and vice-versa).
//!
//! # License
//!
//! - SPDX default: **`apache-2.0`** (`LicenseClass::Permissive`) per
//!   the coverage-audit ticket header (workflow orchestrator's stated
//!   License). Owner override at `docs/license-audit.md` §3.1 sign-off
//!   is required before publish; the sibling `parakeet` /
//!   `parakeet_ctc` / `canary` NVIDIA family precedent is **CC-BY-4.0**
//!   (Attribution Required), so a downstream that obtains the weight
//!   from an NGC / HF card explicitly stamped CC-BY-4.0 SHOULD override
//!   at conversion time (`vokra-cli convert --license cc-by-4.0`) or at
//!   the publish gate. The `license: Option<&str>` parameter of
//!   [`convert_parakeet_unified_file`] wires the same override boundary
//!   `convert_file_licensed` exposes in `lib.rs`, so the GGUF stays the
//!   single source of truth the model card is generated from (no
//!   card/artifact drift).
//! - Category: **`asr`** (asr-offline+streaming — a single tag that
//!   groups both modes; the runtime picks a decode path from the tensor
//!   shapes, not the category string).
//!
//! # BF16 pass-through (mirror of `qwen3_tts` / `vibevoice` /
//! `voxcpm2` / `neucodec` / `emotion2vec` / `parakeet_ctc`)
//!
//! Every F32 / F16 / BF16 tensor passes through under its upstream
//! safetensors name; BF16 emits as GGUF type 30 (`GgmlType::BF16`)
//! verbatim, with no convert-time widening. The runtime widens BF16 →
//! f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 = top
//! 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`ParakeetUnifiedReport::bf16_passthrough`] guards against
//! a silent widen / downcast regression.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / speaker_3d /
//! ecapa_tdnn / parakeet_ctc contract). Real-weight parity binding to
//! a future `vokra-models::parakeet_unified` module (FastConformer
//! encoder + shared attention / TDT / CTC decode paths + punc/cap head)
//! is deferred to owner sign-off in `docs/license-audit.md` §3.1; this
//! converter passes every float tensor through unchanged so a future
//! `ParakeetUnifiedWeights::from_gguf` can walk the same names.
//!
//! # Prep step
//!
//! The upstream Parakeet-Unified release may ship as a `.nemo` tarball;
//! use the shared `tools/parity/nemo_pt_to_safetensors.py` bridge to
//! flatten to safetensors before invoking this converter. No
//! per-model prep script is needed — the flattener is arch-agnostic
//! and already handles the tar/tar.gz/zip NeMo container plus the
//! sibling Parakeet TDT / CTC / Canary safetensors direct paths.
//!
//! # No ONNX (permanent)
//!
//! Parakeet-Unified is distributed as `.nemo` (NGC) or safetensors
//! (HF); this converter **never** touches ONNX (FR-LD-05). A future
//! native loader lives under `crates/vokra-models/src/parakeet_unified/`
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Parakeet-Unified GGUFs. Intentionally distinct
/// from `parakeet-tdt` and `parakeet-ctc` — the unified model's tensor
/// topology carries BOTH decoder families plus the punc/cap head, so
/// silently sharing an arch tag would mis-route the runtime dispatch
/// (FR-EX-08). Kept in sync with the future runtime constant
/// `vokra-models::parakeet_unified::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "parakeet-unified";

/// `vokra.model.name` for Parakeet-Unified GGUFs (canonical model id).
/// Matches the `huggingface.co/vokra/parakeet-unified-en-0.6b` publish
/// slug and the `as_arg` return value in `lib.rs` so the CLI /
/// model-card / publish pipe all agree on a single identifier.
pub(crate) const NAME: &str = "parakeet-unified-en-0.6b";

/// `vokra.model.category` value — asr-offline+streaming (a single tag
/// that groups both modes; the runtime picks a decode path from the
/// tensor shapes, not the category string). Consumed by the model-card
/// generator + zoo manifest tier gate.
pub(crate) const CATEGORY: &str = "asr";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the artifact
/// back to its serving location without parsing the free-text
/// `vokra.provenance.source`. The ticket notes that the primary
/// distribution today is NGC (`catalog.ngc.nvidia.com/`); an HF mirror
/// at this canonical slug is the expected sibling location per the
/// NVIDIA Parakeet family precedent.
pub(crate) const UPSTREAM_HF: &str = "nvidia/parakeet-unified-en-0.6b";

/// Canonical weight license SPDX (`apache-2.0`). Overrides via the
/// [`convert_parakeet_unified_file`] `license` parameter — the standing
/// mechanism for "implementation is clean-room Apache-2.0 but the
/// upstream redistributed checkpoint is another SPDX" scenarios (mirror
/// of `convert_file_licensed` in `lib.rs`, and the same knob used
/// when the NVIDIA card is CC-BY-4.0 instead of Apache-2.0). Lowercase
/// per SPDX convention and matching the `PERMISSIVE_TOKENS` lookup path
/// in `LicenseClass::from_license_str` (which lower-cases before
/// matching).
pub(crate) const DEFAULT_LICENSE: &str = "apache-2.0";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) mirroring the neucodec /
/// emotion2vec / speaker_3d convention until a sibling `category`
/// consumer lands in `vokra-core`.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Ad-hoc metadata key for the upstream HF slug. Kept converter-side
/// mirroring the neucodec / emotion2vec convention (namespaced under
/// `vokra.provenance.*` so a future promotion to a `chunks::KEY_*`
/// alias is byte-compatible with existing GGUFs).
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a Parakeet-Unified conversion.
///
/// Mirrors the sibling BF16-passthrough converters' counter shape
/// ([`super::neucodec::NeucodecReport`],
/// [`super::parakeet_ctc::ParakeetCtcReport`]) — adds `read` tracking
/// every tensor the safetensors reader surfaced so the invariant
/// `read == written + skipped_non_float` is auditable at the report
/// level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ParakeetUnifiedReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all go through
    /// the same byte-copy path — the BF16 pass-through the sibling
    /// converters share).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for parity with the sibling converters).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; runtime widens BF16 → f32
    /// losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. A silent
    /// widen / downcast regression would surface as this counter
    /// drifting away from the input BF16 count.
    pub bf16_passthrough: usize,
}

/// Reads a Parakeet-Unified safetensors checkpoint at `input` (either
/// the upstream HF safetensors directly or the flattened output of
/// `tools/parity/nemo_pt_to_safetensors.py` for a `.nemo` tarball) and
/// writes a Vokra GGUF to `output`, returning a [`ParakeetUnifiedReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// safetensors name; the `vokra.model.*` (arch / name / category) +
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_hf) chunks are stamped for the runtime compliance gate
/// (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// [`DEFAULT_LICENSE`] (`"apache-2.0"`, `Permissive`) per the
/// coverage-audit ticket; a caller who obtained the weight under the
/// NVIDIA Parakeet family's CC-BY-4.0 precedent passes
/// `Some("cc-by-4.0")` to stamp Attribution Required correctly.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_parakeet_unified_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ParakeetUnifiedReport, ConvertError> {
    // Whole-file read: Parakeet-Unified-EN-0.6B is ~1.2 GB (fits in
    // 16 GB M1 iMac RAM comfortably per the ticket's "local safe"
    // note), so the simple `std::fs::read` posture the sibling
    // non-streaming converters (parakeet / parakeet_ctc / canary /
    // neucodec) use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Category / upstream-HF stamps — not covered by `stamp_provenance`
    // (which handles the SPDX + class + model_id + source group only),
    // so written directly. Consumers pick a decode path by category and
    // trace the artifact back to its serving location by upstream_hf.
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = apache-2.0 per the coverage-audit ticket; the
    // `license` override lets a downstream repackager (or a caller
    // whose card is CC-BY-4.0 per the NVIDIA precedent) stamp a
    // different SPDX with `LicenseClass` re-derived from it — the same
    // knob `convert_file_licensed` exposes in `lib.rs`.
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (
            DEFAULT_LICENSE.to_owned(),
            LicenseClass::from_license_str(DEFAULT_LICENSE),
        ),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "nvidia/parakeet-unified-en-0.6b (FastConformer encoder + unified \
             offline/streaming decoders + punc/cap head, ~0.6B params, 16 kHz \
             English ASR)",
        ),
    );

    let mut report = ParakeetUnifiedReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30), same posture as parakeet_ctc /
    // neucodec / qwen3_tts / vibevoice / voxcpm2; runtime widens
    // BF16 → f32 exactly at load via
    // `vokra-core::gguf::quant::decode_bf16` (`bits << 16` is exact).
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
    // stamps `vokra.schema.version` + `vokra.schema.producer` on its
    // own via the writer's built-in schema stamper — no per-converter
    // duplication needed.
    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch path (PID + nanos so concurrent runs
    /// never collide on the same file).
    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-parakeet-unified-{tag}-{}-{}.{ext}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        p
    }

    /// RAII cleanup so failing tests do not leak temp files.
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

    /// Arch string must NOT collide with the sibling parakeet arch
    /// tags — a silently-shared arch would mis-route the runtime
    /// dispatch (FR-EX-08). Pins the fact that `parakeet-unified` is
    /// distinct from both `parakeet-tdt` and `parakeet-ctc`.
    #[test]
    fn arch_does_not_collide_with_sibling_parakeet_variants() {
        assert_eq!(ARCH, "parakeet-unified");
        assert_ne!(ARCH, super::super::parakeet::ARCH);
        assert_ne!(ARCH, super::super::parakeet_ctc::ARCH);
    }

    /// Pins the BF16 pass-through end-to-end: the tensor survives the
    /// converter's `convert_parakeet_unified_file` file → file
    /// round-trip with its dtype preserved (`GgmlType::BF16`, GGUF
    /// type 30) and its payload byte-identical. Mirror of
    /// `neucodec::tests::bf16_tensor_passes_through_verbatim` /
    /// `parakeet_ctc::tests::bf16_tensor_passes_through_verbatim`.
    /// A silent widen at convert time would still round-trip _values_
    /// (BF16 → f32 widen is exact), so this test asserts on the dtype
    /// AND the raw bytes — two concentric fences.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero bit patterns so a silent widen / downcast cannot
        // round-trip trivially through a zero-fill regression.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload = bf16_bytes(&values);
        assert_eq!(payload.len(), 12, "6 elements × 2 bytes BF16");
        let header = r#"{"encoder.blocks.0.attn.qkv_proj.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
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
            convert_parakeet_unified_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of neucodec / parakeet_ctc)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter after the BF16 pass-through land"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through (additive observability)"
        );

        // Round-trip through the emitted GGUF: dtype preserved, payload
        // byte-identical (no convert-time widening).
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("encoder.blocks.0.attn.qkv_proj.weight")
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

    /// F32 and F16 tensors both ride the pass-through arm in the same
    /// conversion (mixed-dtype loops don't collapse to one arm), and
    /// `bf16_passthrough` stays at its Default 0 when no BF16 tensor
    /// is present (additive-field regression guard).
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000]; // 1.0, 2.0 in IEEE half.
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);

        let header = format!(
            r#"{{"encoder.blocks.0.attn.qkv_proj.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"encoder.blocks.0.attn.qkv_proj.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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

        let report = convert_parakeet_unified_file(&input_path, &output_path, None)
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
            .tensor_info("encoder.blocks.0.attn.qkv_proj.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(f32_info.dimensions, vec![1, 2]);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file
            .tensor_info("encoder.blocks.0.attn.qkv_proj.bias")
            .expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(f16_info.dimensions, vec![2]);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Provenance stamped through the default (Apache-2.0 /
        // Permissive), plus arch + name + category + upstream_hf.
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
            Some(DEFAULT_LICENSE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
    }

    /// License override boundary: passing `Some(spdx)` replaces both
    /// the raw SPDX string and the re-derived `LicenseClass`, keeping
    /// the GGUF the single source of truth the model card is generated
    /// from (no card / artifact drift). Mirror of the outer
    /// `convert_file_licensed` override contract at the top-level
    /// `lib.rs` boundary. Uses `cc-by-4.0` as the override to exercise
    /// the NVIDIA Parakeet family precedent path (the ticket flags
    /// this as a plausible per-card override).
    #[test]
    fn license_override_switches_class_when_spdx_differs() {
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = r#"{"encoder.blocks.0.attn.qkv_proj.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("lic-in", "safetensors");
        let output_path = scratch_path("lic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        // Override apache-2.0 with cc-by-4.0 — this crosses a
        // LicenseClass boundary (Permissive → AttributionRequired),
        // making the rederivation observable end-to-end.
        let report = convert_parakeet_unified_file(&input_path, &output_path, Some("cc-by-4.0"))
            .expect("convert with override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("cc-by-4.0"),
            "override replaces the raw SPDX string"
        );
        // Class rederivation must land AttributionRequired, not the
        // default Permissive — a regression that dropped the license →
        // class step would leave Permissive stamped here.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::AttributionRequired.as_str()),
            "class must re-derive from the overridden SPDX (Permissive → AttributionRequired \
             for CC-BY-4.0)"
        );
    }
}
