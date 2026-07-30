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
