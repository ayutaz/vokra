//! **ESPnet OWSM v4 (Open Whisper-style Model) medium 1B** (coverage-audit
//! 2026-08-03 Wave B ticket): safetensors checkpoint → GGUF conversion.
//!
//! Input: the upstream `espnet/owsm_v4_medium_1B` release — a **fully-open**
//! Whisper alternative developed by the ESPnet community and trained on
//! 320k h of multilingual speech (CC-BY 4.0). Output: a GGUF carrying every
//! float tensor verbatim under its upstream safetensors name plus the
//! `vokra.model.*` / `vokra.provenance.*` metadata chunks the future native
//! OWSM loader will read.
//!
//! # HF / licence / category
//!
//! - Upstream HF: `espnet/owsm_v4_medium_1B` (recorded under
//!   `vokra.provenance.upstream_hf`).
//! - SPDX: `cc-by-4.0` → `LicenseClass::AttributionRequired` (ESPnet's
//!   consistent posture across the OWSM family); the converter stamps the
//!   FR-MD-09 attribution text alongside.
//! - Model category: `asr` (open-whisper-alt is a subcategory that the
//!   runtime dispatch resolves by inspecting the arch tag — recorded under
//!   `vokra.model.category`).
//!
//! # Distinct arch tag (rationale)
//!
//! `ARCH = "owsm-v4-medium-1b"` — deliberately **not** sharing the
//! `whisper` / `distil-whisper` / `kotoba-whisper` arch tags even though
//! OWSM was designed as a Whisper alternative. ESPnet trained OWSM with its
//! own recipe (independent tokenizer / hyperparameters / architectural
//! choices) so the tensor names and topology diverge from OpenAI's Whisper
//! release. Silently aliasing `whisper` would mis-route a Whisper loader
//! (openai-flavour tensor names, GPT-2 tokenizer) onto an OWSM checkpoint
//! (ESPnet-flavour tensor names, its own SentencePiece tokenizer) — the
//! kind of drift `FR-EX-08` explicitly forbids. A future `owsm-v4-large`
//! (or `owsm-v5-*`) would be a distinct `ModelKind` variant.
//!
//! # BF16 pass-through (mirror of `neucodec` / `emotion2vec` / `hibiki`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30 (`GgmlType::BF16`) —
//! the same posture as the sibling BF16 skeletons and the CC-BY 4.0
//! Kyutai (`moshi` / `mimi` / `kyutai_stt` / `hibiki`) family. No
//! convert-time widening; runtime widens BF16 → f32 losslessly via the
//! single choke point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`
//! (BF16 is the top 16 bits of an f32 — `bits << 16` is exact). Every
//! F32 / F16 tensor passes through under its upstream name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim** (the
//! Moshi / Kyutai STT / CSM / Kokoro / Hibiki contract). Real-weight
//! binding — including the OWSM-specific SentencePiece tokenizer embed —
//! is a follow-up wave gated on the upstream tensor-name manifest fetch;
//! this converter passes every F32 / F16 / BF16 tensor through unchanged
//! so a future `OwsmV4Weights::from_gguf` can walk the same names.
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream ESPnet Python pipeline is
//! deferred to owner (`docs/license-audit.md` §3.1 sign-off queue) — this
//! converter provides the byte-parallel GGUF surface only.
//!
//! # Prep script needed
//!
//! ESPnet historically ships raw `.pth` (torch state-dict) checkpoints;
//! the HF release also carries safetensors alongside. When the caller
//! downloads the safetensors variant they can feed the file directly to
//! this converter; when they hold only the `.pth` they run
//! `tools/parity/owsm_v4_medium_1b_prepare_checkpoint.py` first to
//! bridge to safetensors (mirror of `dfn3_prepare_checkpoint.py` /
//! `kokoro_prepare_checkpoint.py`; the shared `nemo_pt_to_safetensors.py`
//! is the underlying `.pth` → safetensors converter).
//!
//! # No ONNX (permanent)
//!
//! OWSM is distributed as ESPnet `.pth` + optional safetensors; this
//! converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in a future `crates/vokra-models/src/owsm_v4/`
//! module (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for OWSM v4 medium 1B GGUFs.
///
/// Intentionally distinct from `whisper` / `distil-whisper` /
/// `kotoba-whisper` — OWSM has its own ESPnet-derived tensor topology
/// and tokenizer, and silently aliasing any Whisper arch tag would
/// mis-route the runtime dispatch (FR-EX-08). A future
/// `owsm-v4-large` variant would be a distinct `ModelKind` with its
/// own arch tag.
pub(crate) const ARCH: &str = "owsm-v4-medium-1b";

/// `vokra.model.name` value written for the canonical
/// `espnet/owsm_v4_medium_1B` release.
pub(crate) const NAME: &str = "owsm-v4-medium-1b";

/// Model-category tag written under `vokra.model.category`. `"asr"`
/// keeps OWSM grouped with Whisper / Parakeet / Canary; the
/// open-whisper-alt sub-category is derivable from the arch tag but
/// not carried explicitly (runtime dispatch consumers should read the
/// arch, not the category, for that granularity).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const MODEL_CATEGORY: &str = "asr";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the
/// artifact back to its serving location without parsing the free-text
/// `vokra.provenance.source`.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const UPSTREAM_HF: &str = "espnet/owsm_v4_medium_1B";

/// The FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with the sibling
/// CC-BY 4.0 converters (`moshi::MOSHI_ATTRIBUTION_TEXT` /
/// `mimi::MIMI_ATTRIBUTION_TEXT` /
/// `kyutai_stt::KYUTAI_STT_ATTRIBUTION_TEXT` /
/// `hibiki::HIBIKI_ATTRIBUTION_TEXT`) and the `docs/license-audit.md`
/// ESPnet row (final legal sufficiency = owner sign-off).
pub(crate) const OWSM_V4_ATTRIBUTION_TEXT: &str = "This application uses the ESPnet OWSM v4 medium 1B model (Open Whisper-style Model, \
     fully-open Whisper alternative trained on 320k h of multilingual speech). Model weights \
     are licensed under CC-BY 4.0 (attribution required; commercial use permitted). \
     Copyright (c) ESPnet Community. Source: \
     https://huggingface.co/espnet/owsm_v4_medium_1B";

/// Outcome of an OWSM v4 medium 1B conversion.
#[derive(Debug, Default)]
pub struct OwsmV4Medium1bReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling `hibiki` /
    /// `neucodec` / `emotion2vec` reports).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). The upstream ESPnet release ships F32 /
    /// F16 today; a future BF16-released variant would populate this
    /// counter (the runtime widens BF16 → f32 losslessly at load via
    /// `vokra-core::gguf::quant::decode_bf16`).
    pub bf16_passthrough: usize,
}

/// File-based OWSM v4 medium 1B converter
/// (`vokra-cli convert --model owsm-v4-medium-1b`).
///
/// Reads `input` (upstream `espnet/owsm_v4_medium_1B`
/// `model.safetensors` — or the safetensors emitted by
/// `tools/parity/owsm_v4_medium_1b_prepare_checkpoint.py` when the
/// caller starts from the `.pth` release), writes a Vokra GGUF to
/// `output`. `license` overrides the default `cc-by-4.0` provenance
/// stamp (Whisper / kokoro-family override pattern — see
/// `convert_file_licensed` in `lib.rs`); pass `None` to keep the
/// built-in `cc-by-4.0` stamp with its `AttributionRequired` class.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_owsm_v4_medium_1b_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<OwsmV4Medium1bReport, ConvertError> {
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
    // licence. Default = cc-by-4.0 (upstream `espnet/owsm_v4_medium_1B`
    // ESPnet's consistent OWSM-family posture); the corresponding
    // class is `AttributionRequired` and the runtime surfaces the
    // attribution text via `Session::attribution`. `license` overrides
    // for callers who obtained the weight under a different SPDX (see
    // `convert_file_licensed`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => ("cc-by-4.0".to_owned(), LicenseClass::AttributionRequired),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "espnet/owsm_v4_medium_1B (Open Whisper-style Model, ESPnet fully-open Whisper \
             alternative, cc-by-4.0)",
        ),
    );
    // CC-BY 4.0 obliges whoever redistributes these weights to carry
    // the credit. Burning it into the artifact is what lets a
    // downstream consumer — and `scripts/publish/make_model_card.py` —
    // discharge that without having to know ESPnet's terms
    // independently. Mirror of `hibiki::convert_hibiki_file` /
    // `moshi::convert` / `mimi::convert`.
    vokra_core::stamp_attribution(&mut b, OWSM_V4_ATTRIBUTION_TEXT);

    let mut report = OwsmV4Medium1bReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30), same posture as neucodec /
    // emotion2vec / hibiki / kyutai_stt / moshi; runtime widens BF16 →
    // f32 exactly at load via `vokra-core::gguf::quant::decode_bf16`
    // (`bits << 16` is exact).
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

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Builds a single-BF16-tensor safetensors buffer with a
    /// caller-supplied raw payload.
    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(
            bf16_bytes.len(),
            expected,
            "test fixture: payload len must match shape × 2 BF16"
        );
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
        out
    }

    /// Builds a single-F32-tensor safetensors buffer with a
    /// caller-supplied raw payload. Used to exercise the F32 leg of the
    /// union match arm (ESPnet's `.pth` → safetensors bridge typically
    /// lands F32 tensors).
    fn safetensors_one_f32(name: &str, shape: &[u64], f32_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 4;
        assert_eq!(
            f32_bytes.len(),
            expected,
            "test fixture: payload len must match shape × 4 F32"
        );
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"F32","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            f32_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out
    }

    /// Writes `bytes` to a fresh temp file and returns its path.
    /// Nanosecond suffix keeps parallel `cargo test` runs from
    /// colliding on the same PID.
    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-owsm-v4-medium-1b-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    #[test]
    fn arch_string_is_distinct_from_whisper_family() {
        // Silently aliasing the whisper / distil-whisper / kotoba-whisper
        // arch would mis-route the runtime dispatch (FR-EX-08); pin the
        // distinctness explicitly so a future refactor cannot collapse
        // these under a shared "whisper-like" tag.
        assert_eq!(ARCH, "owsm-v4-medium-1b");
        assert_ne!(ARCH, "whisper");
        assert_ne!(ARCH, "distil-whisper");
        assert_ne!(ARCH, "kotoba-whisper");
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim_with_attribution() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity assert
        // catches any silent widen / downcast attempt (zeroed payloads
        // would round-trip trivially through F32 / F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        let input_bytes = safetensors_one_bf16("encoder.embed.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_owsm_v4_medium_1b_file(&input_path, &output_path, None)
            .expect("convert_owsm_v4_medium_1b_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror hibiki / neucodec / emotion2vec)"
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
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("encoder.embed.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info).len(),
            12,
            "2 rows × 3 cols × 2 B BF16 verbatim"
        );
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        // Provenance / attribution stamps (the ESPnet / CC-BY 4.0
        // posture is what distinguishes this converter from the
        // neucodec permissive skeleton).
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
            Some("cc-by-4.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::AttributionRequired.as_str())
        );
        let attribution = file
            .get(chunks::KEY_PROVENANCE_ATTRIBUTION)
            .and_then(|v| v.as_str())
            .expect("attribution text must be stamped for CC-BY 4.0");
        assert!(
            attribution.contains("ESPnet"),
            "attribution must name the author, got: {attribution}"
        );
        assert!(
            attribution.contains("CC-BY 4.0"),
            "attribution must state the licence, got: {attribution}"
        );

        // The M2-13 gate resolves AttributionRequired and passes the
        // strict (commercial) policy WITHOUT a research flag — CC-BY
        // 4.0 is commercial-OK (never confuse with the CC-BY-NC gate).
        let res = vokra_core::resolve_license_class(&file);
        assert_eq!(res.class, LicenseClass::AttributionRequired);
        assert!(!res.is_research_only());
        vokra_core::check_weight_license(&file, &vokra_core::CompliancePolicy::strict())
            .expect("CC-BY 4.0 passes the strict gate");

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn f32_tensor_passes_through_verbatim() {
        // Non-zero F32 payload so a silent-widen / downcast regression
        // cannot hide behind a zero round-trip. Six values for a [2,3]
        // tensor = 24 bytes.
        let vals: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let f32_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 24, "6 × 4 F32 payload");

        let input_bytes = safetensors_one_f32("decoder.output.weight", &[2, 3], &f32_bytes);
        let input_path = write_temp("f32-in", &input_bytes);
        let output_path = write_temp("f32-out", &[]);

        let report = convert_owsm_v4_medium_1b_file(&input_path, &output_path, None)
            .expect("convert_owsm_v4_medium_1b_file must accept a well-formed F32 checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 must not increment the BF16 counter"
        );
        assert_eq!(report.skipped_non_float, 0);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("decoder.output.weight")
            .expect("F32 tensor present");
        assert_eq!(info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), f32_bytes.as_slice());

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn license_override_reclassifies_the_stamped_licence() {
        // Non-zero BF16 for the same anti-silent-widen reason as above.
        let bf16: Vec<u8> = [1.0f32, -2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("encoder.embed.weight", &[1, 2], &bf16);
        let input_path = write_temp("override-in", &input_bytes);
        let output_path = write_temp("override-out", &[]);

        // Override with a plain MIT SPDX id — the default path would
        // stamp cc-by-4.0. A caller who obtained the weight under a
        // permissive licence (say, a derivative retrained on
        // permissive corpora) can bypass the built-in
        // AttributionRequired default this way; the runtime gate then
        // sees Permissive with no attribution surface.
        let report = convert_owsm_v4_medium_1b_file(&input_path, &output_path, Some("MIT"))
            .expect("convert_owsm_v4_medium_1b_file must accept a licence override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("MIT"),
            "override SPDX must land in `vokra.provenance.license`"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "override must re-classify the weight-class alongside the SPDX"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
