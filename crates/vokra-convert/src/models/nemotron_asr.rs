//! **NVIDIA Nemotron-3.5-ASR-Streaming-0.6B**
//! (`nvidia/nemotron-3.5-asr-streaming-0.6b`, **OpenMDW-1.1** permissive):
//! safetensors → GGUF conversion (2026-07-30 CC owner ADR unblock).
//!
//! # Owner ADR (2026-07-30 完了)
//!
//! From `HF api/models/nvidia/nemotron-3.5-asr-streaming-0.6b`:
//!
//! - `cardData.license: "other"`
//! - `cardData.license_name: "openmdw-1.1"`
//! - `cardData.license_link: "https://openmdw.ai/license/1-1/"`
//! - `gated: False`
//!
//! **OpenMDW-1.1** (Open Model Derivatives Work 1.1, openmdw.ai/license/1-1/,
//! CC 直接照合 2026-07-30) = **Permissive** MIT-analog for ML weights:
//!
//! - commercial 可
//! - redistribution 可 (要 existing notice 保持)
//! - **no** share-alike / copyleft
//! - **no** non-commercial / field-of-use restriction
//! - attribution = notice 保持のみ (Apache-2.0 と同 tier)
//!
//! `LicenseClass::from_license_str("openmdw")` → `Permissive`
//! (`crates/vokra-core/src/compliance/license_class.rs` の `PERMISSIVE_TOKENS`
//! に `openmdw` token を 2026-07-30 追加)。owner ADR 完了で defer marker
//! から本 converter へ昇格。
//!
//! # HF / license / category
//!
//! - Upstream HF: `nvidia/nemotron-3.5-asr-streaming-0.6b` (recorded under
//!   `vokra.provenance.upstream_hf`).
//! - SPDX: **`openmdw-1.1`** (mapped to `LicenseClass::Permissive`).
//! - Category: `asr` (streaming ASR, 36 langs per model card).
//!
//! # BF16 pass-through (mirror of wespeaker / omniasr_ctc)
//!
//! Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
//! safetensors name. No convert-time widening; runtime widens BF16 → f32
//! losslessly via `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`
//! (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the upstream safetensors names verbatim. Real-
//! weight parity + runtime forward binding is a follow-up wave gated on
//! Nemotron-3.5 tensor-name manifest fetch + native ASR streaming
//! architecture implementation (Nemotron is a custom NVIDIA ASR arch,
//! not a Conformer / FastConformer sibling of the existing Parakeet /
//! Canary family).
//!
//! # No ONNX (permanent)
//!
//! Nemotron ships safetensors; this converter **never** touches ONNX
//! (FR-LD-05); the pipeline is re-implemented natively in a future
//! `crates/vokra-models/src/nemotron_asr/` module when the runtime lands
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Nemotron-ASR GGUFs.
pub(crate) const ARCH: &str = "nemotron_asr_streaming";

/// `vokra.model.name` value written for the canonical checkpoint.
pub(crate) const NAME: &str = "nemotron-3.5-asr-streaming-0.6b";

/// Model-category tag written under `vokra.model.category`. `"asr"` groups
/// this with the Whisper / Voxtral / Parakeet / Canary / Cohere-Transcribe
/// family so downstream consumers can pick a load path without inspecting
/// the arch.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub(crate) const MODEL_CATEGORY: &str = "asr";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf`.
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub(crate) const UPSTREAM_HF: &str = "nvidia/nemotron-3.5-asr-streaming-0.6b";

/// Default weight-license SPDX. `openmdw-1.1` is not on the SPDX list yet
/// (2026-07-30), so we use the lower-case identifier NVIDIA advertises on
/// the model card (`cardData.license_name: "openmdw-1.1"`).
pub(crate) const DEFAULT_LICENSE: &str = "openmdw-1.1";

/// Outcome of a Nemotron-ASR conversion.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NemotronAsrReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader rejects unknown dtypes at parse time).
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    pub bf16_passthrough: usize,
}

/// File-based Nemotron-ASR converter (`vokra-cli convert --model
/// nemotron-asr-streaming`).
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input.
pub fn convert_nemotron_asr_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<NemotronAsrReport, ConvertError> {
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = openmdw-1.1 (upstream `nvidia/nemotron-3.5-asr-
    // streaming-0.6b` `cardData.license_name`, 2026-07-30 CC 照合).
    // `license` overrides for callers who obtained the weight under a
    // different SPDX (see `convert_file_licensed` in `lib.rs`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "nvidia/nemotron-3.5-asr-streaming-0.6b \
             (NVIDIA Nemotron-3.5 streaming ASR 0.6B, OpenMDW-1.1 permissive)",
        ),
    );

    let mut report = NemotronAsrReport::default();
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
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    //! Sibling-mirror unit tests (ast / clap / funcodec / speechtokenizer
    //! pattern, FQ-01 coverage close 2026-07-31).
    //!
    //! The module IS live at `huggingface.co/vokra/nemotron-3.5-asr-
    //! streaming-0.6b` — any silent regression in the pass-through loop or
    //! the provenance / licence-override stamp would ship to production,
    //! so the tests here pin (a) F32 / F16 / BF16 pass-through round-trip
    //! via synthetic 2-tensor safetensors bytes, (b) provenance metadata
    //! (arch / name / category / upstream_hf / weight_license) with the
    //! default OpenMDW-1.1 → `LicenseClass::Permissive` resolution, (c)
    //! empty / truncated input → [`ConvertError::Parse`], (d) the
    //! defensive `skipped_non_float` counter — the underlying safetensors
    //! reader rejects non-F32/F16/BF16 dtypes at parse time
    //! (`SafetensorsError::UnsupportedDtype` → [`ConvertError::Parse`]),
    //! so the reader-side rejection is what pins the counter as
    //! defensively unreachable through the public entry point today.
    //!
    //! No external fixtures: every safetensors buffer is hand-woven so the
    //! tests run in the standard `cargo test -p vokra-convert --lib`
    //! matrix without a `VOKRA_*` env gate.
    use super::*;
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    /// Returns a unique per-test tempfile path. PID + monotonic-nanos
    /// suffix keeps parallel `cargo test` invocations from clashing.
    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-nemotron-asr-{tag}-{}-{}.{ext}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        p
    }

    /// Builds a single-tensor safetensors byte buffer with a caller-
    /// supplied dtype tag, shape and raw payload. Generalizes the private
    /// `safetensors_one_bf16` helper in the sibling `ast` / `funcodec` /
    /// `speechtokenizer` tests across dtypes so this module can pin the
    /// reader-side non-float rejection too.
    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(u64::to_string)
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

    /// Two-tensor safetensors buffer: F32 then F16, contiguous data
    /// region. Pins the union `F32 | F16 | BF16` match arm on the F32
    /// and F16 legs simultaneously so a regression that drops F16 (or
    /// mis-counts either as BF16) trips loudly.
    fn safetensors_f32_then_f16() -> Vec<u8> {
        let f32_bytes: Vec<u8> = [1.0_f32, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(f32_bytes.len(), 8);
        // 1.0, 2.0, 3.0 as IEEE-754 F16 bit patterns.
        let f16_bytes: Vec<u8> = [0x3C00_u16, 0x4000, 0x4200]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(f16_bytes.len(), 6);
        let header = format!(
            r#"{{"encoder.layers.0.self_attn.q_proj.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"encoder.layers.0.self_attn.k_proj.weight":{{"dtype":"F16","shape":[3],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&f32_bytes);
        out.extend_from_slice(&f16_bytes);
        out
    }

    /// Distinctive BF16 bit patterns (top-16 bits of the IEEE-754 f32
    /// encodings of 1.0, -2.5, 0.15625, 3.5, -0.5, 42.0). A silent
    /// widen-to-f32 or byte-swap would flip the payload-bytes assertion.
    fn distinctive_bf16_payload() -> Vec<u8> {
        [1.0_f32, -2.5, 0.15625, 3.5, -0.5, 42.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// (a) + (b): BF16 pass-through round-trip pins the tensor bytes
    /// byte-identical (no silent widen), the counter subset firing, the
    /// arch / name / category / upstream_hf provenance stamps, and the
    /// default OpenMDW-1.1 → `LicenseClass::Permissive` resolution
    /// (`crates/vokra-core/src/compliance/license_class.rs`
    /// `PERMISSIVE_TOKENS` — `openmdw` token added 2026-07-30).
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let payload = distinctive_bf16_payload();
        assert_eq!(payload.len(), 12, "6 elements × 2 B BF16 payload");
        // Realistic Nemotron-ASR-Streaming tensor name.
        let input_bytes = safetensors_one(
            "encoder.layers.0.self_attn.out_proj.weight",
            "BF16",
            &[2, 3],
            &payload,
        );

        let input = scratch_path("bf16-in", "safetensors");
        let output = scratch_path("bf16-out", "gguf");
        std::fs::write(&input, &input_bytes).expect("write input safetensors");

        let report =
            convert_nemotron_asr_file(&input, &output, None).expect("BF16 convert must succeed");
        assert_eq!(report.read, 1, "one tensor observed on input");
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(report.skipped_non_float, 0, "BF16 is float — no skip");
        assert_eq!(report.bf16_passthrough, 1, "BF16 subset counter must fire");

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        let info = file
            .tensor_info("encoder.layers.0.self_attn.out_proj.weight")
            .expect("emitted GGUF must carry the tensor");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16 (type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            payload.as_slice(),
            "BF16 payload must round-trip byte-for-byte (no silent widen)",
        );

        // Provenance stamps — the arch / name / category / upstream_hf
        // group is what the M2-13 compliance gate + model-zoo indexer
        // read.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH),
            "vokra.model.arch = `nemotron_asr_streaming`",
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME),
            "vokra.model.name = canonical release string",
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(MODEL_CATEGORY),
            "vokra.model.category = `asr`",
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF),
            "vokra.provenance.upstream_hf = upstream HF path",
        );

        // License resolution: OpenMDW-1.1 default → Permissive.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE),
            "vokra.provenance.license = `openmdw-1.1`",
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "openmdw resolves to Permissive (LicenseClass::from_license_str)",
        );
    }

    /// (a): the F32 + F16 legs of the union match arm surface too, and
    /// the BF16 subset counter stays at Default 0 when no BF16 tensor is
    /// present. Also proves the `read == written + skipped_non_float`
    /// invariant on a well-formed all-float input.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let blob = safetensors_f32_then_f16();
        let input = scratch_path("f32-f16-in", "safetensors");
        let output = scratch_path("f32-f16-out", "gguf");
        std::fs::write(&input, &blob).expect("write input safetensors");

        let report = convert_nemotron_asr_file(&input, &output, None)
            .expect("F32 + F16 convert must succeed");
        assert_eq!(report.read, 2, "two tensors observed on input");
        assert_eq!(report.written, 2, "both F32 and F16 must pass through");
        assert_eq!(report.skipped_non_float, 0, "no non-float tensors here");
        assert_eq!(
            report.bf16_passthrough, 0,
            "no BF16 tensor — subset counter stays at Default 0",
        );
        assert_eq!(
            report.read,
            report.written + report.skipped_non_float,
            "read = written + skipped_non_float invariant",
        );

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        let a = file
            .tensor_info("encoder.layers.0.self_attn.q_proj.weight")
            .expect("F32 tensor present");
        assert_eq!(a.dtype, GgmlType::F32);
        assert_eq!(a.dimensions, vec![1, 2]);
        let b = file
            .tensor_info("encoder.layers.0.self_attn.k_proj.weight")
            .expect("F16 tensor present");
        assert_eq!(b.dtype, GgmlType::F16);
        assert_eq!(b.dimensions, vec![3]);
    }

    /// (c): empty / truncated input surfaces as [`ConvertError::Parse`]
    /// (via `SafetensorsError::Truncated` at the reader boundary, mapped
    /// through the crate-root `From<SafetensorsError>` impl at
    /// `crates/vokra-convert/src/lib.rs:2213`). Guards the boundary where
    /// a silent success on empty input would ship a zero-tensor GGUF to
    /// the publisher.
    #[test]
    fn empty_input_returns_parse_error() {
        let input = scratch_path("empty-in", "safetensors");
        let output = scratch_path("empty-out", "gguf");
        // Zero-byte safetensors: reader must reject with Truncated
        // (< 8 B header prefix).
        std::fs::write(&input, b"").expect("write empty input");

        let err = convert_nemotron_asr_file(&input, &output, None)
            .expect_err("empty safetensors must fail — do not ship a zero-tensor GGUF");
        std::fs::remove_file(&input).ok();
        // Output must not exist — the converter aborts before writing.
        assert!(
            !output.exists(),
            "empty input must not produce an output GGUF",
        );

        match err {
            ConvertError::Parse(_) => {}
            other => panic!("expected ConvertError::Parse, got {other:?}"),
        }
    }

    /// (d): non-F32/F16/BF16 dtypes are rejected at the safetensors
    /// reader boundary (`SafetensorsError::UnsupportedDtype` →
    /// [`ConvertError::Parse`]). This is what pins the
    /// `skipped_non_float` arm as defensive — a caller cannot reach it
    /// through the public `convert_nemotron_asr_file` entry point today
    /// because `SafetensorsFile::parse` refuses I8 / I32 / I64 / F64
    /// before the walk ever begins
    /// (`crates/vokra-core/src/safetensors.rs:411-418`). A regression
    /// that widened the reader's dtype whitelist without adding the
    /// corresponding non-float branch to `convert_nemotron_asr_file`
    /// would land here.
    #[test]
    fn non_float_dtype_rejected_at_parse() {
        // Two 32-bit little-endian sentinel words. Content is irrelevant
        // — the reader rejects the dtype token before decoding payload.
        let payload: Vec<u8> = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let blob = safetensors_one("dummy.i32", "I32", &[2], &payload);
        let input = scratch_path("i32-in", "safetensors");
        let output = scratch_path("i32-out", "gguf");
        std::fs::write(&input, &blob).expect("write I32 input safetensors");

        let err = convert_nemotron_asr_file(&input, &output, None)
            .expect_err("I32 tensors must be rejected at the reader boundary");
        std::fs::remove_file(&input).ok();
        assert!(
            !output.exists(),
            "I32 rejection must abort before any output write",
        );
        match err {
            ConvertError::Parse(_) => {}
            other => panic!("expected ConvertError::Parse, got {other:?}"),
        }
    }

    /// The `license: Option<&str>` boundary rewrites the SPDX + class
    /// pair in place (mirror of `convert_file_licensed` posture in
    /// `crates/vokra-convert/src/lib.rs`). Regression guard: a caller
    /// obtaining the weight under a different distribution licence must
    /// see their override reflected in both
    /// `vokra.provenance.license` and `vokra.provenance.weight_license`.
    #[test]
    fn license_override_replaces_openmdw_default() {
        let payload = distinctive_bf16_payload();
        let blob = safetensors_one("encoder.embed.weight", "BF16", &[2, 3], &payload);
        let input = scratch_path("license-override-in", "safetensors");
        let output = scratch_path("license-override-out", "gguf");
        std::fs::write(&input, &blob).expect("write input safetensors");

        // Override with apache-2.0 (also Permissive but a distinct SPDX
        // string — asserts the raw string is what's written, not a
        // silent normalization back to openmdw-1.1).
        let report = convert_nemotron_asr_file(&input, &output, Some("apache-2.0"))
            .expect("licensed convert must succeed");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "raw SPDX override must be written verbatim",
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 also resolves to Permissive (distinct SPDX, same class)",
        );
    }
}
