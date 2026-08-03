//! **Kyutai Hibiki-2B** (coverage-audit 2026-08-03 Wave B ticket):
//! safetensors checkpoint → GGUF conversion.
//!
//! Input: the upstream `kyutai/hibiki-2b-pytorch-bf16` release — a
//! **simultaneous speech-to-speech translation** model (French ↔
//! English, released 2025-02) in the same Kyutai Helium-temporal +
//! depformer + Mimi codec lineage as `Moshi` and `Kyutai STT`, but
//! trained for streaming translation rather than full-duplex chat.
//! Output: a GGUF carrying every float tensor verbatim under its
//! upstream safetensors name plus the `vokra.model.*` /
//! `vokra.provenance.*` metadata chunks the future native Hibiki
//! loader will read.
//!
//! # HF / licence / category
//!
//! - Upstream HF: `kyutai/hibiki-2b-pytorch-bf16` (recorded under
//!   `vokra.provenance.upstream_hf`).
//! - SPDX: `cc-by-4.0` → `LicenseClass::AttributionRequired` (same
//!   posture as sibling Moshi / Mimi / Kyutai STT); the converter
//!   stamps the FR-MD-09 attribution text alongside.
//! - Model category: `s2s` (simultaneous translation is a subcategory —
//!   recorded under `vokra.model.category`).
//!
//! # Distinct arch tag (rationale)
//!
//! `ARCH = "hibiki"` — deliberately **not** sharing the `moshi` arch
//! tag, mirror of the `Kyutai STT` decision (also Kyutai-family but
//! keeps its own converter and arch). Silently aliasing `moshi` would
//! mis-route the runtime dispatch: a Moshi loader would read
//! `vokra.moshi.*` hparams and apply the 7 B chat topology to a 2 B
//! translation checkpoint. Hibiki's tensor topology (2 B,
//! translation-tuned) differs from Moshi's 7 B (chat-tuned) — hparams
//! (n_layer / d_model / delay structure / n_q_in / dep_q) would need
//! re-derivation, not verbatim reuse.
//!
//! # BF16 pass-through (mirror of `neucodec` / `emotion2vec`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`) — the same posture as the sibling Kyutai
//! converters. No convert-time widening; runtime widens BF16 → f32
//! losslessly via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). Every F32 / F16
//! tensor passes through under its upstream name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the Moshi / Kyutai STT / CSM / Kokoro contract). Real-weight
//! binding is a follow-up wave gated on the upstream tensor-name
//! manifest fetch; this converter passes every F32 / F16 / BF16
//! tensor through unchanged so a future `HibikiWeights::from_gguf`
//! can walk the same names.
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream Python pipeline is deferred
//! to owner (`docs/license-audit.md` §3.1 sign-off) — this converter
//! provides the byte-parallel GGUF surface only.
//!
//! # No prep script needed
//!
//! Upstream ships safetensors directly (mirror of Kyutai STT), no
//! `.pth` bridge required.
//!
//! # No ONNX (permanent)
//!
//! Hibiki is distributed as safetensors + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in a future `crates/vokra-models/src/hibiki/`
//! module (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断
//! 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Hibiki GGUFs.
pub(crate) const ARCH: &str = "hibiki";

/// `vokra.model.name` value written for the canonical Hibiki-2B GGUF.
pub(crate) const NAME: &str = "hibiki-2b";

/// Model-category tag written under `vokra.model.category`. `"s2s"`
/// keeps Hibiki grouped with Moshi / CSM / Voxtral (S2S / dialog);
/// simultaneous translation is a subcategory that the runtime dispatch
/// resolves by inspecting the arch tag rather than the category
/// string.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const MODEL_CATEGORY: &str = "s2s";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the
/// artifact back to its serving location without parsing the free-text
/// `vokra.provenance.source`.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const UPSTREAM_HF: &str = "kyutai/hibiki-2b-pytorch-bf16";

/// The FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with the sibling
/// Kyutai converters (`moshi::MOSHI_ATTRIBUTION_TEXT` /
/// `kyutai_stt::KYUTAI_STT_ATTRIBUTION_TEXT` /
/// `mimi::MIMI_ATTRIBUTION_TEXT`) and the `docs/license-audit.md`
/// Kyutai row (final legal sufficiency = T29-equivalent owner
/// sign-off).
pub(crate) const HIBIKI_ATTRIBUTION_TEXT: &str = "This application uses the Kyutai Hibiki-2B model \
     (simultaneous speech-to-speech translation, Helium + Mimi codec). Model \
     weights are licensed under CC-BY 4.0 (attribution required; commercial \
     use permitted). Copyright (c) Kyutai. Source: \
     https://huggingface.co/kyutai/hibiki-2b-pytorch-bf16";

/// Outcome of a Hibiki conversion.
#[derive(Debug, Default)]
pub struct HibikiReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling `neucodec` /
    /// `emotion2vec` reports).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). The upstream release is `-pytorch-bf16` so
    /// this counter is expected to equal `written` on a real
    /// conversion.
    pub bf16_passthrough: usize,
}

/// File-based Hibiki converter (`vokra-cli convert --model hibiki-2b`).
///
/// Reads `input` (upstream `kyutai/hibiki-2b-pytorch-bf16`
/// `model.safetensors`), writes a Vokra GGUF to `output`. `license`
/// overrides the default `cc-by-4.0` provenance stamp (Whisper /
/// kokoro-family override pattern — see `convert_file_licensed` in
/// `lib.rs`); pass `None` to keep the built-in `cc-by-4.0` stamp with
/// its `AttributionRequired` class.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_hibiki_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<HibikiReport, ConvertError> {
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
    // licence. Default = cc-by-4.0 (upstream `kyutai/hibiki-2b-
    // pytorch-bf16` model card); the corresponding class is
    // `AttributionRequired` and the runtime surfaces the attribution
    // text via `Session::attribution`. `license` overrides for callers
    // who obtained the weight under a different SPDX (see
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
        Some("kyutai/hibiki-2b-pytorch-bf16 (simultaneous FR↔EN S2S translation, cc-by-4.0)"),
    );
    // CC-BY 4.0 obliges whoever redistributes these weights to carry
    // the credit. Burning it into the artifact is what lets a
    // downstream consumer — and `scripts/publish/make_model_card.py` —
    // discharge that without having to know Kyutai's terms
    // independently. Mirror of `moshi::convert` / `mimi::convert`.
    vokra_core::stamp_attribution(&mut b, HIBIKI_ATTRIBUTION_TEXT);

    let mut report = HibikiReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30), same posture as neucodec /
    // emotion2vec / kyutai_stt / moshi; runtime widens BF16 → f32
    // exactly at load via `vokra-core::gguf::quant::decode_bf16`
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

    /// Writes `bytes` to a fresh temp file and returns its path.
    /// Nanosecond suffix keeps parallel `cargo test` runs from
    /// colliding on the same PID.
    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-hibiki-{kind}-{}-{}.bin",
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

        let input_bytes = safetensors_one_bf16("depformer.linear.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_hibiki_file(&input_path, &output_path, None)
            .expect("convert_hibiki_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror neucodec / emotion2vec / moshi)"
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
            .tensor_info("depformer.linear.weight")
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

        // Provenance / attribution stamps (the Kyutai / CC-BY 4.0
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
            attribution.contains("Kyutai"),
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
    fn license_override_is_honored() {
        // Non-zero BF16 for the same anti-silent-widen reason as above.
        let bf16: Vec<u8> = [1.0f32, -2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("depformer.emb.weight", &[1, 2], &bf16);
        let input_path = write_temp("override-in", &input_bytes);
        let output_path = write_temp("override-out", &[]);

        // Override with a plain MIT SPDX id — the default path would
        // stamp cc-by-4.0. A caller who obtained the weight under a
        // permissive licence (say, a derivative retrained on
        // permissive corpora) can bypass the built-in
        // AttributionRequired default this way; the runtime gate then
        // sees Permissive with no attribution surface.
        let report = convert_hibiki_file(&input_path, &output_path, Some("MIT"))
            .expect("convert_hibiki_file must accept a licence override");
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
