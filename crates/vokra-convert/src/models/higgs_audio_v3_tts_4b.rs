//! **BosonAI Higgs-Audio v3 TTS 4B** (`bosonai/higgs-audio-v3-tts-4b`):
//! safetensors checkpoint → GGUF conversion
//! (coverage-audit-2026-08-03 Wave B fast-track).
//!
//! Input: the upstream `bosonai/higgs-audio-v3-tts-4b` release on
//! HuggingFace — a **multilingual zero-shot TTS** (100+ languages)
//! release from BosonAI (Adept AI 系譜) with **emotion inline tags**
//! (`[happy]` / `[sad]` / …) baked into the LM tokenizer. The upstream
//! distribution ships **sharded safetensors** (~8 GB total in BF16 for
//! the 4B backbone), so callers pre-merge the shards through
//! `tools/parity/higgs_audio_v3_tts_4b/prepare_checkpoint.py`
//! (uv-managed Python 3.12) before invoking this converter — the DFN3 /
//! DAC / CSM / Kokoro / UTMOS / SBV2 / FRCRN / Voxtral / VoxCPM /
//! Qwen3-TTS pre-flatten posture. Pickles / sharded index files never
//! enter the runtime (FR-LD-05).
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`
//! and `vokra.provenance.*` metadata chunks the runtime TTS path binds
//! against.
//!
//! # License
//!
//! - SPDX: **apache-2.0** ([`vokra_core::LicenseClass::Permissive`]).
//! - Category: **tts** — the model is a TTS release (the audit ticket
//!   groups it as `tts/multilingual`; the shorter `tts` variant is used
//!   here so runtime dispatch and model-card grouping stay uniform with
//!   the existing TTS family and do not multiply category labels by
//!   language-count distinctions the arch tag already carries — mirror
//!   of the `magpietts_v2602` category posture).
//! - Notes: audit ticket `docs/tickets/coverage-audit-2026-08-03/
//!   wave-b/higgs-audio-v3-tts-4b.md` cites the ~8 GB size, Apache-2.0
//!   license, and 100+-language multilingual coverage; the upstream
//!   distribution path (`bosonai/higgs-audio-v3-tts-4b` on Hugging
//!   Face) is recorded verbatim in the `vokra.provenance.upstream_hf`
//!   chunk so a downstream consumer can trace the artifact back to its
//!   serving location.
//! - **Owner sign-off remains pending** — audit ticket §Owner critical
//!   path lists (1) HF card `license: apache-2.0` primary-source
//!   confirmation, (2) BosonAI GitHub LICENSE cross-check, (3) training
//!   corpus commercial-use audit (100+ langs = Common Voice / VoxPopuli
//!   混成疑義), (4) `docs/license-audit.md` §3.1 row landing. This
//!   converter and its BF16 pass-through are green now; publish is
//!   fail-closed until the §3.1 row acquires a ☑ Commercial / ☑
//!   Research-only mark from the owner (memory
//!   `[[feedback-license-signoff-primary-source]]`).
//!
//! # vast.ai required (~8 GB)
//!
//! Per memory `[[feedback-large-models-on-vast-ai]]` (2 GB CC
//! workflow local-convert owner threshold; the M1 iMac 16 GB machine
//! ran into OS-level swap when mmap-ing 8 GB Voxtral / 48 GB
//! Voxtral-Small-24B), the actual weight fetch + convert + publish
//! runs on a rented vast.ai GPU box via
//! `docs/handoff/vast-ai-large-model-publish.md` — CC lands only the
//! converter code + prepare-checkpoint sidecar + tests here.
//!
//! # BF16 pass-through (mirror of magpietts_v2602 / firered_asr_aed_l /
//! # canary_1b_flash / owsm_v4_medium_1b / parakeet_tdt_1_1b /
//! # sortformer_diar_4spk_v1 / speaker_3d / ecapa_tdnn / qwen3_tts /
//! # voxcpm2 / vibevoice / moshi / emotion2vec / wespeaker / frcrn /
//! # nkf_aec)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm on the same
//! match arm — no convert-time widening. BF16 is emitted as GGUF type
//! 30 (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`HiggsAudioV3Tts4bReport::bf16_passthrough`] guards against
//! a silent widen / downcast regression.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / WeSpeaker / emotion2vec / FRCRN / MagpieTTS-v2602 /
//! FireRedASR-AED-L contract; the prepare-checkpoint sidecar preserves
//! the dotted state-dict keys the sharded safetensors index exposes,
//! plus dedupes shared tensors via `data_ptr` tracking per memory
//! `[[reference-safetensors-shared-tensor-dedup]]`). Real-weight
//! parity binding to a future runtime `vokra-models::higgs_audio_v3_tts_4b`
//! module is deferred to owner sign-off per
//! `docs/license-audit.md §3.1`.
//!
//! # SGLang sampler → Vokra Sampler primitive substitution
//!
//! Per the audit ticket §Converter notes, the upstream reference
//! implementation invokes SGLang's sampler on the LM decoder side.
//! When the future runtime `vokra-models::higgs_audio_v3_tts_4b`
//! module lands, the LM decoder will consume Vokra's existing
//! `crates/vokra-core/src/engine/sampler.rs` primitive (already
//! wired through voxtral / cosyvoice2 / canary_qwen) — SGLang is a
//! _reference-side implementation detail_ that the Vokra runtime
//! does not inherit. This converter stamps only the weights and
//! provenance; the sampler swap happens at runtime binder time and
//! carries no converter-side change.
//!
//! # No ONNX (permanent)
//!
//! The upstream Higgs-Audio v3 TTS 4B release ships PyTorch sharded
//! safetensors + a Python inference pipeline; this converter **never**
//! touches ONNX (FR-LD-05); the pipeline is re-implemented natively
//! in a future `crates/vokra-models/src/higgs_audio_v3_tts_4b/`
//! module (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for Higgs-Audio v3 TTS 4B GGUFs.
/// Intentionally distinct from every sibling TTS arch tag — silently
/// sharing (e.g.) `cosyvoice2` / `piper_plus` / `sbv2` / `qwen3_tts` /
/// `magpietts_v2602` / `neutts-air` would misroute the runtime dispatch
/// (a HiFTChain-based loader would try to interpret a Higgs-Audio v3
/// checkpoint with a completely different topology). Snake-case per the
/// local convention (the CLI slug uses the hyphenated form via
/// [`crate::ModelKind::as_arg`]).
pub const ARCH: &str = "higgs_audio_v3_tts_4b";

/// `vokra.model.name` value written for the canonical
/// `bosonai/higgs-audio-v3-tts-4b` release.
pub const NAME: &str = "higgs-audio-v3-tts-4b";

/// `vokra.model.category` value written for every Higgs-Audio v3 TTS 4B
/// GGUF.
///
/// The audit ticket's category label is "tts/multilingual"; the
/// shorter `tts` variant is used here so runtime dispatch and model-
/// card grouping stay uniform with the existing `tts` family (Kokoro /
/// piper-plus / CosyVoice2/3 / Chatterbox / Qwen3-TTS / VoxCPM /
/// VibeVoice / Irodori / VITS-JA / SBV2 / Dia / Zonos / MagpieTTS-
/// v2602 / NeuTTS Air) and do not multiply category labels by
/// language-count distinctions the arch tag already carries.
pub const CATEGORY: &str = "tts";

/// Upstream distribution slug on Hugging Face, recorded under
/// `vokra.provenance.upstream_hf` so a downstream consumer can trace
/// the artifact back to its serving location without parsing the
/// free-text `vokra.provenance.source` blob.
pub const UPSTREAM_HF: &str = "bosonai/higgs-audio-v3-tts-4b";

/// Default upstream weight licence (SPDX). Apache-2.0 per the audit
/// ticket's Publish gate `redistributable` entry (owner primary-source
/// verification pending per §Owner critical path (1)).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// convention the sibling BF16-passthrough converters use applies).

/// `vokra.model.category` metadata key. Local per the established
/// funcodec / wespeaker / speaker_3d / ecapa_tdnn / emotion2vec /
/// frcrn / nkf_aec / magpietts_v2602 precedent (not yet centralized in
/// `vokra-core::gguf::chunks`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` — the Hugging Face repository slug
/// the release ships from. Local per the wespeaker / frcrn /
/// speaker_3d / magpietts_v2602 convention (parallel to
/// `vokra.provenance.upstream_url` for non-HF sources).
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a Higgs-Audio v3 TTS 4B conversion.
///
/// Mirrors the sibling BF16-passthrough converters' counter shape
/// (`super::magpietts_v2602::MagpiettsV2602Report`,
/// `super::firered_asr_aed_l::FireredAsrAedLReport`,
/// `super::frcrn::FrcrnReport`) — tracks every tensor the safetensors
/// reader surfaced so the invariant
/// `read == written + skipped_non_float` is auditable at the report
/// level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HiggsAudioV3Tts4bReport {
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

/// Converts a Higgs-Audio v3 TTS 4B safetensors checkpoint at `input`
/// (as emitted by
/// `tools/parity/higgs_audio_v3_tts_4b/prepare_checkpoint.py`)
/// into a Vokra-native GGUF at `output`, returning a
/// [`HiggsAudioV3Tts4bReport`].
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
/// `DEFAULT_LICENSE_SPDX` (`"apache-2.0"`, `Permissive`) — the
/// upstream release is Apache-2.0 per the audit ticket. A downstream
/// repackager may pass e.g. `Some("apache-2.0")` verbatim to make the
/// stamp explicit even without a licence change (mirror of the
/// `magpietts_v2602` / `firered_asr_aed_l` / `wespeaker` /
/// `emotion2vec` / `frcrn` override convention).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
///
/// # Memory footprint
///
/// The upstream release is ~8 GB (4B parameters × 2 bytes BF16). Per
/// memory `[[feedback-large-models-on-vast-ai]]` this exceeds the
/// 2 GB CC-workflow local-convert threshold, so the actual convert
/// runs on vast.ai. The current `std::fs::read` load buffers the
/// entire file into memory — a future streaming pass-through (per the
/// Moshi 15 GB / Voxtral 8.7 GB `SafetensorsFileReader` +
/// `GgufStreamWriter` posture) would shrink the peak footprint to one
/// tensor payload; that upgrade is a follow-up if the vast.ai box's
/// RAM budget becomes a constraint.
pub fn convert_higgs_audio_v3_tts_4b_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<HiggsAudioV3Tts4bReport, ConvertError> {
    // Whole-file read: Higgs-Audio v3 TTS 4B is ~8 GB per the audit
    // ticket. Above the Moshi 15 GB / Voxtral 8.7 GB streaming
    // threshold in principle, but the wave-b sibling posture
    // (magpietts_v2602 / firered_asr_aed_l / owsm_v4_medium_1b at
    // 700 MB - 2.2 GB) uses whole-file read so this converter matches
    // that; if a vast.ai run OOMs, swap this call for
    // `SafetensorsFileReader::open` + `GgufStreamWriter::begin` per
    // the moshi.rs / qwen3_tts.rs ADR (docs/adr/qwen3-tts-bf16.md,
    // strategy A_passthrough).
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = apache-2.0 (upstream
    // `bosonai/higgs-audio-v3-tts-4b` Apache-2.0 per the audit
    // ticket's Publish gate — owner primary-source verification
    // pending). The optional `license` argument overrides via the
    // same restated-source convention as the sibling converters
    // (`magpietts_v2602` / `firered_asr_aed_l` / `wespeaker` /
    // `frcrn` / `emotion2vec` / `speaker_3d`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "bosonai/higgs-audio-v3-tts-4b (BosonAI multilingual TTS — 100+ languages, \
             emotion inline tag, apache-2.0)",
        ),
    );

    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted
    // BF16-passthrough ADR the sibling non-streaming converters
    // (magpietts_v2602 / firered_asr_aed_l / speaker_3d / ecapa_tdnn /
    // qwen3_tts / vibevoice / voxcpm2 / moshi / wespeaker /
    // emotion2vec / frcrn / nkf_aec) share; the runtime widens BF16
    // → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    let mut report = HiggsAudioV3Tts4bReport::default();
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
            "vokra-higgs-audio-v3-tts-4b-{tag}-{}-{}.{ext}",
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

    /// Pins the arch / name / category / upstream_hf constants so a
    /// silent rename in this file cannot slip past review. Every
    /// downstream reader (compliance gate / model-card generator /
    /// zoo manifest) keys off these exact strings; drifting them
    /// mis-routes the runtime dispatch (FR-EX-08). Mirrors the
    /// sibling `magpietts_v2602` / `firered_asr_aed_l` posture.
    #[test]
    fn arch_and_name_pin_matches_publish_repo() {
        assert_eq!(ARCH, "higgs_audio_v3_tts_4b");
        assert_eq!(NAME, "higgs-audio-v3-tts-4b");
        assert_eq!(CATEGORY, "tts");
        assert_eq!(UPSTREAM_HF, "bosonai/higgs-audio-v3-tts-4b");
        assert_eq!(DEFAULT_LICENSE_SPDX, "apache-2.0");
        // Sibling TTS arch tags — silently sharing would mis-route
        // runtime dispatch. Assert distinctness on the closest sibling
        // families this converter groups next to in the tree.
        for sibling in [
            "cosyvoice2",
            "cosyvoice3",
            "qwen3_tts",
            "voxcpm2",
            "vibevoice",
            "chatterbox",
            "magpietts_v2602",
            "piper_plus",
            "kokoro",
            "sbv2",
            "neutts-air",
        ] {
            assert_ne!(ARCH, sibling, "arch tag must not collide with {sibling}");
        }
    }

    /// Pins the BF16 pass-through end-to-end: the tensor survives the
    /// converter's `convert_higgs_audio_v3_tts_4b_file` file → file
    /// round-trip with its dtype preserved (`GgmlType::BF16`, GGUF
    /// type 30) and its payload byte-identical. Mirrors
    /// `magpietts_v2602::tests::bf16_tensor_passes_through_verbatim` /
    /// `firered_asr_aed_l::tests::bf16_tensor_passes_through_verbatim`.
    /// A silent widen at convert time would still round-trip _values_
    /// (BF16 → f32 widen is exact), so this test asserts on the dtype
    /// AND the raw bytes — two concentric fences.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero bit patterns so a silent widen / downcast cannot
        // round-trip trivially. Tensor name mirrors a plausible
        // upstream LM-decoder embed-tokens key.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload = bf16_bytes(&values);
        assert_eq!(payload.len(), 12, "6 elements × 2 bytes BF16");
        let header = r#"{"decoder.model.embed_tokens.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_higgs_audio_v3_tts_4b_file(&input_path, &output_path, None)
            .expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of magpietts_v2602 / firered_asr_aed_l)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        // Round-trip through the emitted GGUF: dtype preserved,
        // payload byte-identical (no convert-time widening).
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("decoder.model.embed_tokens.weight")
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
    /// guard).
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        // Two tensors in one safetensors file:
        //   speaker_encoder.linear.weight — F32, [1, 2] →  8 bytes @ [0..8)
        //   speaker_encoder.linear.bias   — F16, [2]    →  4 bytes @ [8..12)
        // Both dtypes must reach the pass-through arm and neither must
        // increment `bf16_passthrough`.
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000]; // 1.0, 2.0 in IEEE half.
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);

        let header = format!(
            r#"{{"speaker_encoder.linear.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"speaker_encoder.linear.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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

        let report = convert_higgs_audio_v3_tts_4b_file(&input_path, &output_path, None)
            .expect("convert F32 + F16 mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2, "F32 and F16 must both pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 / F16 must NOT increment the BF16 counter"
        );

        // Both tensors survive the round trip with dtype + bytes
        // intact.
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let f32_info = file
            .tensor_info("speaker_encoder.linear.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(f32_info.dimensions, vec![1, 2]);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file
            .tensor_info("speaker_encoder.linear.bias")
            .expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(f16_info.dimensions, vec![2]);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());
    }

    /// Pins the provenance chunk group end-to-end (arch / name /
    /// category / upstream_hf / license / weight_license class /
    /// model_id / source) so a silent stamp regression is caught.
    /// The default apache-2.0 → Permissive mapping is asserted with
    /// no override on the boundary.
    #[test]
    fn provenance_stamped_end_to_end_with_default_license() {
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = r#"{"decoder.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("prov-in", "safetensors");
        let output_path = scratch_path("prov-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report =
            convert_higgs_audio_v3_tts_4b_file(&input_path, &output_path, None).expect("convert");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
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
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME)
        );
    }

    /// Pins the license override boundary: passing `Some(spdx)`
    /// replaces both the raw SPDX string and the re-derived
    /// `LicenseClass`, keeping the GGUF the single source of truth
    /// the model card is generated from (no card / artifact drift).
    /// Mirrors the outer `convert_file_licensed` override contract
    /// at the top-level lib.rs boundary and the sibling
    /// `magpietts_v2602` / `firered_asr_aed_l` license override
    /// tests.
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
        // Permissive, so the LicenseClass rederivation is a no-op;
        // the SPDX string is what changes.
        let report = convert_higgs_audio_v3_tts_4b_file(&input_path, &output_path, Some("mit"))
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
        // Permissive — asserting explicitly guards against a
        // rederivation regression that dropped the license → class
        // step.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );
    }

    /// Pins that empty-string license override falls through to the
    /// apache-2.0 default (matches the sibling `magpietts_v2602`
    /// convention where `Some("") => default`; the wave-b uniform
    /// posture guards against a CLI operator passing `--license ""`
    /// and accidentally stamping an empty SPDX).
    #[test]
    fn empty_license_override_falls_back_to_default() {
        let f32_bytes: Vec<u8> = [1.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = r#"{"decoder.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("emptylic-in", "safetensors");
        let output_path = scratch_path("emptylic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_higgs_audio_v3_tts_4b_file(&input_path, &output_path, Some(""))
            .expect("convert with empty override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX),
            "empty override string must fall through to apache-2.0 default"
        );
    }

    /// Pins the `read == written + skipped_non_float` invariant on a
    /// mixed-arm input (both float and — if the reader ever admitted
    /// them — non-float tensors would be observed). The current
    /// safetensors reader admits only F32 / F16 / BF16, so an int
    /// tensor makes the parse itself fail; this test therefore
    /// asserts the invariant on an all-float mixed input, which is
    /// the intended contract shape. Mirror of the sibling BF16-
    /// passthrough report-counter posture.
    #[test]
    fn report_read_written_invariant_holds() {
        // Three tensors, all floats: F32 + F16 + BF16 — the report
        // must show read = written = 3 and skipped_non_float = 0
        // with bf16_passthrough = 1.
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_bytes: Vec<u8> = [0x3C00u16, 0x4000]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bf16_payload = bf16_bytes(&[3.5, -1.25]);
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);
        assert_eq!(bf16_payload.len(), 4);

        let header = format!(
            r#"{{"a":{{"dtype":"F32","shape":[2],"data_offsets":[0,{}]}},"b":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}},"c":{{"dtype":"BF16","shape":[2],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
            f32_bytes.len() + f16_bytes.len() + bf16_payload.len(),
        );
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16_bytes);
        input_bytes.extend_from_slice(&bf16_payload);

        let input_path = scratch_path("invariant-in", "safetensors");
        let output_path = scratch_path("invariant-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_higgs_audio_v3_tts_4b_file(&input_path, &output_path, None)
            .expect("convert 3-float mix");
        assert_eq!(report.read, 3, "three tensors observed on input");
        assert_eq!(
            report.written, 3,
            "all three floats must ride the pass-through arm"
        );
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must equal input BF16 count (1)"
        );
        // Invariant.
        assert_eq!(report.read, report.written + report.skipped_non_float);
    }
}
