//! **FunAudioLLM SenseVoiceSmall** (FunASR MODEL_LICENSE):
//! safetensors → GGUF conversion (coverage-audit-2026-08-03 Wave B).
//!
//! Input: the upstream `FunAudioLLM/SenseVoiceSmall` release on HF — a
//! multi-task speech understanding model that jointly performs
//! multilingual ASR (zh / en / ja / yue / ko), LID (language ID), SER
//! (speech emotion recognition), and AED (audio event detection) from a
//! shared SAN-M (Modified Multi-head Attention) enhanced Conformer
//! encoder plus per-task heads. The upstream release ships as
//! `model.pt` (torch pickle) alongside auxiliary text assets
//! (`tokens.json`, `am.mvn`); callers pre-flatten the pickle to
//! safetensors offline via the existing
//! `tools/parity/nemo_pt_to_safetensors.py --allow-strip-any` bridge
//! (the same generic pickle-to-safetensors bridge Canary / Parakeet /
//! ReazonSpeech-NeMo-v2 reuse — pickles never enter the runtime,
//! FR-LD-05).
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`
//! and `vokra.provenance.*` metadata chunks the runtime ASR-multi-task
//! path binds against.
//!
//! # License — fail-closed default (owner §3.1 sign-off required)
//!
//! - SPDX: **`FunASR_MODEL_LICENSE`** (the upstream custom licence,
//!   `github.com/modelscope/FunASR/blob/main/MODEL_LICENSE`). This
//!   string is NOT one of the SPDX ids
//!   [`vokra_core::LicenseClass::from_license_str`] recognises today
//!   (no `mit` / `apache` / `bsd` / `cc-by*` / `gpl` token), so the
//!   classifier returns [`LicenseClass::Unknown`] (fail-closed) — the
//!   correct default per `[[feedback-license-signoff-primary-source]]`
//!   until the owner reads the primary-source licence text and either
//!   (a) adds the id to the [`LicenseClass`] matcher or
//!   (b) overrides at the CLI boundary
//!   (`vokra-cli convert --license <spdx>`) with a canonical SPDX id
//!   after review.
//! - Category: **asr** (multi-task speech understanding; the audit
//!   ticket's `asr-multitask-zh` label collapses to the shorter
//!   `asr` variant so runtime dispatch and model-card grouping stay
//!   uniform with the sibling `asr` family — the multi-task / language
//!   axis rides in the model name and per-task heads rather than
//!   multiplying category labels).
//! - Notes: the audit ticket
//!   (`docs/tickets/coverage-audit-2026-08-03/wave-b/sensevoicesmall.md`,
//!   gitignored internal PM artefact) records the ~470 MB size and
//!   the FunASR MODEL_LICENSE upstream disposition; publish must
//!   defer to owner sign-off (`docs/license-audit.md §3.1`).
//!
//! # BF16 pass-through (mirror of reazonspeech_nemo_v2 / neucodec /
//! # ecapa_tdnn / speaker_3d / qwen3_tts / vibevoice / voxcpm2 / moshi)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm on the same
//! match arm — no convert-time widening. BF16 is emitted as GGUF type
//! 30 ([`GgmlType::BF16`]); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`SenseVoiceSmallReport::bf16_passthrough`] guards against
//! a silent widen / downcast regression.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream FunASR state-dict keys
//! verbatim** (the CSM / Kokoro / CosyVoice2 / Chatterbox / speaker_3d /
//! ecapa_tdnn / canary / parakeet / reazonspeech_nemo_v2 contract; the
//! `nemo_pt_to_safetensors.py` sidecar preserves the dotted state-dict
//! keys). Real-weight parity binding to a future
//! `vokra-models::sensevoicesmall` module (SAN-M enhanced Conformer
//! encoder + per-task CTC / classifier heads native forward) is
//! deferred to owner sign-off per `docs/license-audit.md §3.1`.
//!
//! # Arch tag distinctness
//!
//! `vokra.model.arch = "sensevoicesmall"` is intentionally distinct
//! from every sibling ASR arch tag:
//!
//! - `kotoba-whisper` / `whisper` / `distil-whisper` — OpenAI Whisper
//!   family (Whisper encoder + Whisper decoder).
//! - `canary` / `parakeet-tdt` / `parakeet-ctc` — NVIDIA FastConformer
//!   family (multi-head attention, single-task ASR / AST).
//! - `omniasr-ctc` — Meta wav2vec 2.0 waveform-in encoder.
//! - `reazonspeech_nemo_v2` — Longformer local-attention encoder
//!   (Japanese long-form single-task ASR).
//!
//! Silently sharing an arch tag with any of these would mis-route the
//! runtime dispatch — SenseVoice's SAN-M enhanced Conformer plus its
//! four task heads (ASR + LID + SER + AED sharing the encoder trunk)
//! is a distinct topology whose `from_gguf` walk cannot be handled by
//! any of the above.
//!
//! # No ONNX (permanent)
//!
//! The upstream SenseVoiceSmall release ships PyTorch pickle files
//! (`model.pt`); this converter **never** touches ONNX (FR-LD-05). The
//! pickle-to-safetensors bridge is `tools/parity/nemo_pt_to_safetensors.py`,
//! run offline (the ReazonSpeech-NeMo-v2 / Canary / Parakeet posture).
//!
//! # Wiring status
//!
//! This is the TDD skeleton (BF16 / F16 / F32 pass-through plus
//! provenance / category stamps). The runtime native SAN-M enhanced
//! Conformer encoder with the four task heads is a follow-up wave,
//! deferred to owner sign-off (see `docs/license-audit.md` §3.1).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for SenseVoiceSmall GGUFs. Intentionally
/// distinct from every sibling ASR arch tag
/// (`kotoba-whisper` / `whisper` / `distil-whisper` / `canary` /
/// `parakeet-tdt` / `parakeet-ctc` / `omniasr-ctc` /
/// `reazonspeech_nemo_v2`) — SenseVoice's SAN-M enhanced Conformer +
/// four task heads (ASR + LID + SER + AED) is a distinct topology, so
/// silently sharing would mis-route the runtime dispatch.
pub const ARCH: &str = "sensevoicesmall";

/// `vokra.model.name` value written for the canonical
/// `FunAudioLLM/SenseVoiceSmall` release.
pub const NAME: &str = "sensevoicesmall";

/// `vokra.model.category` value written for every SenseVoiceSmall GGUF.
///
/// The audit's category label is "asr-multitask-zh"; the shorter `asr`
/// variant is used here so runtime dispatch and model-card grouping
/// stay uniform with the existing `asr` family
/// (`kotoba-whisper` / `canary` / `parakeet-tdt` / `parakeet-ctc` /
/// `omniasr-ctc` / `reazonspeech_nemo_v2`) — the multi-task /
/// subject-language axis rides in the model name and per-task heads
/// rather than multiplying category labels.
pub const CATEGORY: &str = "asr";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the artifact
/// back to its serving location without parsing the free-text
/// `vokra.provenance.source`. Verified against
/// `huggingface.co/FunAudioLLM/SenseVoiceSmall`.
pub const UPSTREAM_HF: &str = "FunAudioLLM/SenseVoiceSmall";

/// Default upstream weight licence (SPDX). Verbatim from the upstream
/// FunASR release (`github.com/modelscope/FunASR/blob/main/MODEL_LICENSE`).
/// This is a custom licence that is NOT in
/// [`vokra_core::LicenseClass::from_license_str`]'s SPDX matcher; the
/// classifier returns [`LicenseClass::Unknown`] on this string
/// (fail-closed default per `[[feedback-license-signoff-primary-source]]`).
/// Owner sign-off (`docs/license-audit.md §3.1`) OR an explicit CLI
/// override (`--license <canonical-spdx>`) is required for
/// commercial redistribution.
pub const DEFAULT_LICENSE_SPDX: &str = "FunASR_MODEL_LICENSE";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// convention the sibling BF16-passthrough converters use applies).

/// `vokra.model.category` metadata key. Local per the established
/// funcodec / wespeaker / speaker_3d / ecapa_tdnn / neucodec /
/// reazonspeech_nemo_v2 precedent (not yet centralized in
/// `vokra-core::gguf::chunks`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` metadata key — the primary
/// redistribution source HF slug for models mirrored on the Hugging
/// Face hub. Parallel to `vokra.provenance.upstream_url` (the raw-URL
/// sibling key for GitHub-only releases such as `nkf-aec` or
/// `sber-gigaam-*`). Local per the same convention as
/// [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a SenseVoiceSmall conversion.
///
/// Mirrors the sibling BF16-passthrough converters' counter shape
/// ([`super::reazonspeech_nemo_v2::ReazonspeechNemoV2Report`],
/// [`super::neucodec::NeucodecReport`],
/// [`super::ecapa_tdnn::EcapaTdnnReport`],
/// [`super::speaker_3d::Speaker3dReport`]) — the invariant
/// `read == written + skipped_non_float` is auditable at the report
/// level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SenseVoiceSmallReport {
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

/// Converts a SenseVoiceSmall safetensors checkpoint at `input` (as
/// emitted by `tools/parity/nemo_pt_to_safetensors.py --allow-strip-any`
/// unwrapping the upstream `model.pt` torch pickle) into a Vokra-native
/// GGUF at `output`, returning a [`SenseVoiceSmallReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// FunASR state-dict key; the `vokra.model.*` (arch / name / category)
/// and `vokra.provenance.*` (weight_license / license / model_id /
/// source / upstream_hf) chunks are stamped for the runtime compliance
/// gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// [`DEFAULT_LICENSE_SPDX`] (`"FunASR_MODEL_LICENSE"`) which resolves
/// to [`LicenseClass::Unknown`] (fail-closed) — a caller who has read
/// the primary-source license and made a compliance judgement may
/// supply a canonical SPDX id (e.g. `"apache-2.0"`) to reclassify.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_sensevoicesmall_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SenseVoiceSmallReport, ConvertError> {
    // Load the whole checkpoint into memory — the SenseVoiceSmall
    // release is ~470 MB (well below the streaming-mandated Moshi
    // 14 GiB tier), so the simple `std::fs::read` posture the sibling
    // non-streaming converters (reazonspeech_nemo_v2 / canary /
    // parakeet / omniasr-ctc / neucodec / ecapa_tdnn) use applies.
    // Callers on memory-constrained hosts can use
    // `restamp_provenance` post-conversion to update metadata without
    // re-reading the whole file.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Default provenance stamp — the upstream FunASR MODEL_LICENSE
    // string. `LicenseClass::from_license_str` returns `Unknown` on
    // this custom licence (fail-closed default per
    // `[[feedback-license-signoff-primary-source]]`) — the runtime
    // research-flag gate refuses to load with a strict policy until
    // the owner performs §3.1 sign-off or the caller overrides via
    // `--license <canonical-spdx>` after reading the primary source.
    let effective_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some(
            "FunAudioLLM/SenseVoiceSmall (SAN-M enhanced Conformer encoder + \
             multi-task heads: ASR (zh/en/ja/yue/ko) + LID + SER + AED, \
             FunASR MODEL_LICENSE — owner §3.1 sign-off required for \
             commercial redistribution)",
        ),
    );

    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted BF16-passthrough
    // ADR the sibling non-streaming converters (reazonspeech_nemo_v2 /
    // speaker_3d / ecapa_tdnn / qwen3_tts / vibevoice / voxcpm2 / moshi /
    // neucodec) share; the runtime widens BF16 → f32 exactly at load via
    // the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    // decode_bf16`.
    let mut report = SenseVoiceSmallReport::default();
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

    /// Per-test unique scratch path (PID + nanos + a caller-supplied
    /// tag so parallel `cargo test` runs do not collide on the same
    /// file name).
    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-sensevoicesmall-{tag}-{}-{}.{ext}",
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
    /// converter's `convert_sensevoicesmall_file` file → file round-trip
    /// with its dtype preserved ([`GgmlType::BF16`], GGUF type 30) and
    /// its payload byte-identical. Mirrors
    /// `reazonspeech_nemo_v2::tests::bf16_tensor_passes_through_verbatim`.
    /// A silent widen at convert time would still round-trip _values_
    /// (BF16 → f32 widen is exact), so this test asserts on the dtype
    /// AND the raw bytes — two concentric fences.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero bit patterns so a silent widen / downcast cannot
        // round-trip trivially.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload = bf16_bytes(&values);
        assert_eq!(payload.len(), 12, "6 elements × 2 bytes BF16");
        // SAN-M enhanced Conformer tensor name — the FunASR state-dict
        // key convention preserved verbatim through the
        // `nemo_pt_to_safetensors.py --allow-strip-any` bridge.
        let header = r#"{"encoder.encoders.0.self_attn.linear_q.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
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
            convert_sensevoicesmall_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of reazonspeech_nemo_v2 / \
             neucodec / ecapa_tdnn / speaker_3d)"
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
            .tensor_info("encoder.encoders.0.self_attn.linear_q.weight")
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
    /// through the default (fail-closed FunASR MODEL_LICENSE →
    /// [`LicenseClass::Unknown`]) code path.
    #[test]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_fail_closed() {
        // Two tensors in one safetensors file:
        //   encoder.encoders.0.norm.weight      — F32, [1, 2] →  8 bytes @ [0..8)
        //   ctc_lo.lm_head.bias                 — F16, [2]    →  4 bytes @ [8..12)
        // Both dtypes must reach the pass-through arm and neither must
        // increment `bf16_passthrough`.
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000]; // 1.0, 2.0 in IEEE half.
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);

        let header = format!(
            r#"{{"encoder.encoders.0.norm.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"ctc_lo.lm_head.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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

        let report = convert_sensevoicesmall_file(&input_path, &output_path, None)
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
            .tensor_info("encoder.encoders.0.norm.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(f32_info.dimensions, vec![1, 2]);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file.tensor_info("ctc_lo.lm_head.bias").expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(f16_info.dimensions, vec![2]);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Provenance stamped through the default — the SPDX string is
        // the verbatim upstream custom licence, and the classifier
        // returned `Unknown` (fail-closed) because that string is not
        // in the [`LicenseClass::from_license_str`] matcher.
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
            Some(LicenseClass::Unknown.as_str()),
            "FunASR MODEL_LICENSE must resolve to Unknown (fail-closed) — \
             owner sign-off required for commercial redistribution"
        );
    }

    /// Pins the license override boundary: passing `Some(spdx)`
    /// replaces both the raw SPDX string and the re-derived
    /// [`LicenseClass`], keeping the GGUF the single source of truth
    /// the model card is generated from (no card / artifact drift).
    /// Mirrors the outer `convert_file_licensed` override contract at
    /// the top-level lib.rs boundary. Also demonstrates that a caller
    /// who has read the primary-source licence and made a compliance
    /// judgement can reclassify the artifact away from the fail-closed
    /// `Unknown` default.
    #[test]
    fn license_override_replaces_default() {
        // Minimal single-F32-tensor safetensors buffer — the license
        // override contract is independent of tensor shape / count.
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = r#"{"encoder.embed.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("lic-in", "safetensors");
        let output_path = scratch_path("lic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        // Override the fail-closed default with apache-2.0 — a caller
        // who has read the FunASR MODEL_LICENSE text and made a
        // compliance judgement OR is re-distributing under stricter
        // terms may reclassify at this boundary.
        let report = convert_sensevoicesmall_file(&input_path, &output_path, Some("apache-2.0"))
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
            "apache-2.0 reclassifies away from the fail-closed Unknown default"
        );
    }
}
