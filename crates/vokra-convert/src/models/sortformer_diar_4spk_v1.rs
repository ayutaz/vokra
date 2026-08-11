//! **NVIDIA Sortformer-Diar-4spk-v1** (coverage-audit 2026-08-03 Wave B
//! ticket): safetensors checkpoint → GGUF conversion.
//!
//! Input: the upstream `nvidia/diar_sortformer_4spk-v1` release — an
//! **end-to-end 4-speaker diarization** model consisting of an 18-layer
//! NeMo Encoder for Speech Tasks (NEST) encoder based on Fast-Conformer,
//! followed by an 18-layer Transformer encoder (hidden size 192) with a
//! diarization head that emits four per-frame sigmoid outputs (one per
//! target speaker). Output: a GGUF carrying every float tensor verbatim
//! under its upstream safetensors name plus the `vokra.model.*` /
//! `vokra.provenance.*` metadata chunks the future native Sortformer
//! loader will read.
//!
//! # HF / licence / category
//!
//! - Upstream HF: `nvidia/diar_sortformer_4spk-v1` (recorded under
//!   `vokra.provenance.upstream_hf`).
//! - SPDX default: **`cc-by-nc-4.0`** →
//!   [`LicenseClass::NonCommercial`]. The HF model-card front-matter
//!   carries `license: cc-by-nc-4.0` and the card body reaffirms
//!   "License to use this model is covered by the CC-BY-NC-4.0."
//!   (WebFetch-verified 2026-08-03.) The workflow ticket header
//!   nominally reads `CC-BY-4.0`; the primary source (the HF model-card
//!   front-matter itself) governs the actual license class, and the
//!   `NC` clause forces a **fail-closed** posture — an unmarked
//!   commercial-mode caller cannot silently bring the weights up. That
//!   mirrors the `xcodec2` posture that already lives in
//!   `license_class.rs` (T4 tier precedent, `docs/license-audit.md`
//!   §3.1 2026-07-23).
//! - Model category: **`diarize`** (end-to-end 4-speaker diarization).
//!   The `-e2e-4speaker` subcategory is a runtime axis the loader
//!   derives from the four sigmoid outputs / hparams — the category
//!   chunk stays the top-level taxonomy tag consumed by
//!   `docs/license-audit.md` and the model-zoo manifest.
//!
//! # Distinct arch tag (rationale)
//!
//! `ARCH = "sortformer"` — deliberately **not** sharing the
//! `parakeet-tdt` / `parakeet-ctc` / `parakeet-unified` / `canary`
//! FastConformer-encoder-based arch tags, even though the NEST
//! encoder is Fast-Conformer-derived. Silently aliasing any Parakeet
//! arch tag would mis-route the runtime dispatch (FR-EX-08): a
//! FastConformer-ASR loader would look for `joint.*` / `decoder.*`
//! tensors the diarization-head-only Sortformer never emits, and
//! Sortformer's per-frame 4-sigmoid diarization head has no ASR-side
//! analog. The `sortformer` tag is version-neutral (a future
//! Sortformer-8spk / Sortformer-16spk stays classifier-compatible),
//! while the `NAME` (`"sortformer-diar-4spk-v1"`) is versioned to
//! match the canonical publish slug.
//!
//! # BF16 pass-through (mirror of `xcodec2` / `neucodec` /
//! `parakeet_unified` / `hibiki`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`) — the same posture as the sibling BF16
//! pass-through converters. No convert-time widening; runtime widens
//! BF16 → f32 losslessly via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). Every F32 / F16
//! tensor passes through under its upstream name. The upstream HF
//! release advertises F32 tensors today, but keeping the same
//! pass-through skeleton as the BF16 fleet means a future
//! Sortformer BF16 re-release does not need a converter re-write —
//! the arm already handles all three float dtypes.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the Moshi / Kyutai STT / CSM / Kokoro / Parakeet / Hibiki
//! contract). Real-weight binding is a follow-up wave gated on the
//! upstream tensor-name manifest fetch; this converter passes every
//! F32 / F16 / BF16 tensor through unchanged so a future
//! `SortformerWeights::from_gguf` can walk the same names.
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream NeMo Python pipeline is
//! deferred to owner (`docs/license-audit.md` §3.1 sign-off) — this
//! converter provides the byte-parallel GGUF surface only.
//!
//! # Prep step
//!
//! The upstream Sortformer release ships both a `.nemo` tarball
//! (NGC / NeMo delivery format) and direct safetensors on the HF
//! model card. For the `.nemo` path callers pre-flatten to
//! safetensors offline via the shared
//! `tools/parity/nemo_pt_to_safetensors.py` bridge (the sibling
//! `parakeet` / `parakeet_ctc` / `canary` / `parakeet_unified` /
//! `reazonspeech_nemo_v2` pattern; no per-model prep script is
//! needed because the flattener is arch-agnostic). The safetensors
//! path is direct — no prep step needed.
//!
//! # No ONNX (permanent)
//!
//! Sortformer is distributed as `.nemo` (NGC) or safetensors (HF);
//! this converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in a future
//! `crates/vokra-models/src/sortformer/` module (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Sortformer GGUFs. Version-neutral short form
/// — deliberately distinct from every Parakeet / Canary FastConformer
/// arch tag so a diarization checkpoint never mis-routes onto an ASR
/// loader (FR-EX-08). Kept in sync with the future runtime constant
/// `vokra-models::sortformer::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "sortformer";

/// `vokra.model.name` value written for the canonical
/// `sortformer-diar-4spk-v1` GGUF. Matches the
/// `huggingface.co/vokra/sortformer-diar-4spk-v1` publish slug and the
/// `as_arg` return value in `lib.rs` so the CLI / model-card / publish
/// pipe all agree on a single identifier.
pub(crate) const NAME: &str = "sortformer-diar-4spk-v1";

/// `vokra.model.category` value — `diarize` (end-to-end speaker
/// diarization). The `-e2e-4speaker` subcategory is a runtime axis the
/// loader derives from the four sigmoid outputs; the category chunk
/// stays the top-level taxonomy tag consumed by
/// `docs/license-audit.md` and the model-zoo manifest.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const MODEL_CATEGORY: &str = "diarize";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the artifact
/// back to its serving location without parsing the free-text
/// `vokra.provenance.source`.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const UPSTREAM_HF: &str = "nvidia/diar_sortformer_4spk-v1";

/// The default upstream weight license — `cc-by-nc-4.0`, per the HF
/// model card `license: cc-by-nc-4.0` (WebFetch-verified 2026-08-03;
/// sign-off in `docs/license-audit.md` §3.1 is fail-closed until owner
/// confirmation). Callers can override at the outer
/// `convert_file --license <spdx>` boundary when they legitimately
/// hold the weight under a distinct SPDX id (same pattern as
/// xcodec2 / Whisper / kokoro).
const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-4.0";

/// Human-readable upstream source note stored in
/// `vokra.provenance.source` (`KEY_PROVENANCE_SOURCE`). Kept short — the
/// license machine class is carried separately in the
/// `vokra.provenance.weight_license` chunk.
const UPSTREAM_SOURCE: &str =
    "nvidia/diar_sortformer_4spk-v1 (end-to-end 4-speaker diarization, cc-by-nc-4.0)";

/// Outcome of a Sortformer conversion.
///
/// Mirrors the sibling BF16-passthrough converters' counter shape
/// (`super::xcodec2::XCodec2Report`, `super::neucodec::NeucodecReport`,
/// `super::parakeet_unified::ParakeetUnifiedReport`) — every counter
/// is additive so the invariant `read == written + skipped_non_float`
/// is auditable at the report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SortformerDiar4spkV1Report {
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

/// File-based Sortformer converter
/// (`vokra-cli convert --model sortformer-diar-4spk-v1`).
///
/// Reads `input` (upstream `nvidia/diar_sortformer_4spk-v1`
/// `model.safetensors`, or the flattened output of
/// `tools/parity/nemo_pt_to_safetensors.py` for the `.nemo` NGC
/// distribution path), writes a Vokra GGUF to `output`, returning a
/// [`SortformerDiar4spkV1Report`].
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
/// `DEFAULT_LICENSE_SPDX` (`"cc-by-nc-4.0"`, `NonCommercial`) per the
/// primary source (the HF model-card front-matter). A caller who
/// legitimately holds the weight under a distinct SPDX id passes
/// `Some(spdx)` to swap the class; the source parenthetical is
/// neutralised on the override path (same behaviour as the xcodec2
/// arm's `convert_xcodec2_file`) so the stamped `source` never
/// contradicts the license.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_sortformer_diar_4spk_v1_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SortformerDiar4spkV1Report, ConvertError> {
    // Whole-file read: Sortformer-Diar-4spk-v1 is ~0.1B params / ~1 GB
    // (fits in 16 GB M1 iMac RAM comfortably per the ticket's "local
    // safe" note), so the simple `std::fs::read` posture the sibling
    // non-streaming converters (parakeet / parakeet_ctc / canary /
    // parakeet_unified / neucodec / xcodec2) use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Category / upstream-HF stamps — not covered by `stamp_provenance`
    // (which handles the SPDX + class + model_id + source group only),
    // so written directly. Consumers pick a decode path by category and
    // trace the artifact back to its serving location by upstream_hf.
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = cc-by-nc-4.0 / NonCommercial per the primary
    // source (HF model-card front-matter). The `license` override lets
    // a downstream repackager (or a caller who legitimately holds the
    // weight under a distinct SPDX id) stamp a different SPDX with
    // `LicenseClass` re-derived from it — the same knob
    // `convert_file_licensed` exposes in `lib.rs`. The built-in gate
    // fails **closed** at load time in commercial mode
    // (`LicenseClass::NonCommercial::requires_research_flag = true`),
    // so an operator who never touched the license flag cannot
    // silently bring up an NC weight in production. Mirror of the
    // xcodec2 posture.
    let (spdx, class, source_note) = match license {
        Some(s) if !s.is_empty() => {
            let owned = s.to_owned();
            let class = LicenseClass::from_license_str(&owned);
            // Neutralise the source parenthetical on the override path
            // (mirror of `convert_xcodec2_file`) so the stamped
            // `source` never contradicts the license.
            let source = format!("upstream distribution source (licence {owned} per source)");
            (owned, class, source)
        }
        _ => (
            DEFAULT_LICENSE_SPDX.to_owned(),
            LicenseClass::NonCommercial,
            UPSTREAM_SOURCE.to_owned(),
        ),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(&source_note));

    let mut report = SortformerDiar4spkV1Report::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the sibling ADR shared with
    // neucodec / xcodec2 / parakeet_unified / hibiki / qwen3-tts /
    // vibevoice / voxcpm2; runtime widens BF16 → f32 exactly at load
    // via `vokra-core::gguf::quant::decode_bf16` (`bits << 16` is
    // exact).
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
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    /// A unique temp path — per-process id **plus** a monotonic counter
    /// so two tests in the same process never race on the same file.
    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-sortformer-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    /// Encodes an f32 array as little-endian BF16 bytes (top 16 bits of
    /// the f32 pattern — the exact inverse of the runtime's
    /// `decode_bf16 : bits << 16`).
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Builds a synthetic single-tensor safetensors buffer with a
    /// caller-declared dtype and raw payload.
    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// Arch string must NOT collide with the sibling Parakeet /
    /// Canary FastConformer-encoder-based arch tags — a silently-
    /// shared arch would mis-route the runtime dispatch (FR-EX-08).
    /// Pins the fact that `sortformer` is distinct from every
    /// FastConformer-ASR sibling that lives in this tree today
    /// (`parakeet_unified` is a sibling wave-B ticket on a separate
    /// branch — a future merge should add its ARCH here too).
    #[test]
    fn arch_does_not_collide_with_fastconformer_asr_variants() {
        assert_eq!(ARCH, "sortformer");
        assert_ne!(ARCH, super::super::parakeet::ARCH);
        assert_ne!(ARCH, super::super::parakeet_ctc::ARCH);
    }

    /// BF16 pass-through end-to-end: the tensor survives the
    /// converter's `convert_sortformer_diar_4spk_v1_file` file → file
    /// round-trip with its dtype preserved (`GgmlType::BF16`, GGUF
    /// type 30) and its payload byte-identical. Mirror of
    /// `xcodec2::tests::bf16_tensor_passes_through_verbatim` /
    /// `parakeet_unified::tests::bf16_tensor_passes_through_verbatim`.
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
        let input_bytes = safetensors_one(
            "encoder.layers.0.self_attn.qkv.weight",
            "BF16",
            &[2, 3],
            &payload,
        );

        let input_path = tmp_path("bf16-in");
        let output_path = tmp_path("bf16-out");
        std::fs::write(&input_path, &input_bytes).expect("write input");

        let report = convert_sortformer_diar_4spk_v1_file(&input_path, &output_path, None)
            .expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of xcodec2 / parakeet_unified)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
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
            .tensor_info("encoder.layers.0.self_attn.qkv.weight")
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

        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
    }

    /// F32 pass-through with all arch / provenance / category stamps —
    /// the upstream Sortformer HF release advertises F32 tensors today,
    /// so this is the "expected-shape" path. The default license path
    /// must stamp `cc-by-nc-4.0` / `NonCommercial` (the whole point of
    /// keeping this converter separate from the Permissive fleet arm
    /// — silently defaulting to Apache-2.0 / Permissive would
    /// mis-classify NC weights on load).
    #[test]
    fn f32_tensor_passes_through_and_default_license_is_noncommercial() {
        let f32_vals: [f32; 4] = [7.0, -8.25, 0.5, -0.5];
        let f32_bytes_raw: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes_raw.len(), 16);

        let input_bytes = safetensors_one(
            "encoder.layers.0.self_attn.qkv.weight",
            "F32",
            &[2, 2],
            &f32_bytes_raw,
        );
        let input_path = tmp_path("f32-in");
        let output_path = tmp_path("f32-out");
        std::fs::write(&input_path, &input_bytes).expect("write input");

        let report = convert_sortformer_diar_4spk_v1_file(&input_path, &output_path, None)
            .expect("convert F32");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1, "F32 must pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 must NOT increment the BF16 counter (additive-default invariant)"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("encoder.layers.0.self_attn.qkv.weight")
            .expect("F32 tensor present");
        assert_eq!(info.dtype, GgmlType::F32);
        assert_eq!(info.dimensions, vec![2, 2]);
        assert_eq!(file.tensor_bytes(info), f32_bytes_raw.as_slice());

        // Arch / name / category / provenance chunks land with the
        // built-in cc-by-nc-4.0 NonCommercial stamp.
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
            Some(MODEL_CATEGORY)
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
            Some(LicenseClass::NonCommercial.as_str())
        );

        // The M2-13 gate refuses to load this artifact in commercial
        // mode (`LicenseClass::NonCommercial::requires_research_flag =
        // true`) — this fail-closed default is the whole point of the
        // NC-vs-Permissive flip.
        let res = vokra_core::resolve_license_class(&file);
        assert_eq!(res.class, LicenseClass::NonCommercial);
        assert!(res.is_research_only());
        let err = vokra_core::check_weight_license(&file, &vokra_core::CompliancePolicy::strict())
            .expect_err("strict policy MUST refuse cc-by-nc-4.0 without a research flag");
        // The error surfaces the license class in its message so a
        // downstream operator can act on it without re-inspecting the
        // GGUF.
        let msg = format!("{err}");
        assert!(
            msg.contains("NonCommercial") || msg.contains("non-commercial") || msg.contains("nc"),
            "strict-policy refusal must mention the NC class in its error, got {msg:?}"
        );

        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
    }

    /// A caller-supplied `license` (e.g. re-published under a different
    /// SPDX at the source) overrides the built-in cc-by-nc-4.0
    /// NonCommercial stamp. Same override pattern as
    /// `convert_file_licensed` — the model_id / arch / category /
    /// upstream_hf strings survive but the license / weight_license /
    /// source change.
    #[test]
    fn caller_license_override_swaps_the_stamp() {
        // Non-zero payloads that are NOT approximations of π/e — avoids
        // clippy::approx_constant.
        let f32_vals: [f32; 2] = [11.5, -6.25];
        let f32_bytes_raw: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one(
            "encoder.layers.0.linear.weight",
            "F32",
            &[1, 2],
            &f32_bytes_raw,
        );
        let input_path = tmp_path("override-in");
        let output_path = tmp_path("override-out");
        std::fs::write(&input_path, &input_bytes).expect("write input");

        // Override to Apache-2.0 (Permissive) — e.g. the caller
        // retrained on a permissive corpus. The class must re-derive.
        let report =
            convert_sortformer_diar_4spk_v1_file(&input_path, &output_path, Some("apache-2.0"))
                .expect("convert with override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "override SPDX must land in vokra.provenance.license"
        );
        // Class rederivation must land Permissive, not the default
        // NonCommercial — a regression that dropped the license →
        // class step would leave NonCommercial stamped here.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "class must re-derive from the overridden SPDX (NonCommercial → \
             Permissive for apache-2.0)"
        );
        // Model id / arch / category / upstream_hf remain the built-in
        // values — the override changes only the license triple + the
        // source parenthetical.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(MODEL_CATEGORY)
        );

        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
    }

    /// An empty `Some("")` license override must NOT wipe the built-in
    /// stamp — that would be a silent research-flag downgrade. The
    /// `filter(|s| !s.is_empty())` guard in
    /// `convert_sortformer_diar_4spk_v1_file` keeps the default
    /// cc-by-nc-4.0 NonCommercial stamp.
    #[test]
    fn empty_string_license_override_keeps_the_default_stamp() {
        let f32_vals: [f32; 2] = [0.5, -0.5];
        let f32_bytes_raw: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one(
            "encoder.layers.0.norm.weight",
            "F32",
            &[1, 2],
            &f32_bytes_raw,
        );
        let input_path = tmp_path("empty-in");
        let output_path = tmp_path("empty-out");
        std::fs::write(&input_path, &input_bytes).expect("write input");

        let _ = convert_sortformer_diar_4spk_v1_file(&input_path, &output_path, Some(""))
            .expect("convert");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX),
            "empty string must NOT downgrade the license stamp"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercial.as_str()),
            "empty string must NOT downgrade the class"
        );

        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
    }
}
