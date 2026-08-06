//! **Audiobox Aesthetics** (`facebook/audiobox-aesthetics`, cc-by-4.0):
//! safetensors checkpoint → GGUF conversion (TIER 2 land, 2026-07-30).
//!
//! Input: the upstream `facebook/audiobox-aesthetics` release —
//! `model.safetensors` (~104M F32 params, arXiv:2502.05139 — "Meta
//! Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech,
//! Music, and Sound"). Output: a GGUF carrying every float tensor
//! verbatim under its upstream safetensors name, plus the
//! `vokra.audiobox_aesthetics.*` / `vokra.provenance.*` /
//! `vokra.model.*` metadata chunks a future native
//! `vokra-models::audiobox_aesthetics::*` implementation will read.
//!
//! # What the model produces (5-dim audio-quality rating)
//!
//! Audiobox Aesthetics is an **audio-classification** head: given an
//! audio clip, it emits a five-dimensional real-valued rating over
//! BALANCED / CONTENT_ENJOYMENT (CE) / CONTENT_USEFULNESS (CU) /
//! PRODUCTION_COMPLEXITY (PC) / PRODUCTION_QUALITY (PQ). Per the
//! upstream `config.json` (fetched 2026-07-30 —
//! `https://huggingface.co/facebook/audiobox-aesthetics/raw/main/config.json`,
//! CLAUDE.md「ハルシネーション厳禁」):
//!
//! - Backbone: wav2vec2-style SSL encoder (weighted-layer-sum over the
//!   `nth_layer` = 13 encoder outputs, `use_weighted_layer_sum: true`).
//! - Head: 5-layer projection MLP (`proj_num_layer: 5`, `proj_act_fn:
//!   gelu`, `proj_dropout: 0.0`, `proj_ln: true`) producing an
//!   `output_dim: 1` scalar per axis (per-axis heads share the backbone).
//! - Precision: `"32"` (F32 tensors upstream — the safetensors payload
//!   is F32 verbatim, ~104M × 4 B ≈ 415 MB on disk).
//! - Target normalisation: per-axis `{mean, std}` transform recorded in
//!   `config.json.target_transform` (CE: μ 5.06865 σ 1.93029, CU:
//!   μ 5.73633 σ 1.75669, PC: μ 3.18591 σ 1.86637, PQ: μ 6.57505
//!   σ 1.51466). BALANCED is `undefined` upstream and derived per-clip
//!   from the four axes.
//! - Embed normalisation: `normalize_embed: true`.
//!
//! # HF / licence / category (primary-source verified 2026-07-30)
//!
//! - Upstream HF: `facebook/audiobox-aesthetics` (recorded under
//!   `vokra.provenance.upstream_hf`).
//! - HF cardData `license: cc-by-4.0` — `LicenseClass::AttributionRequired`
//!   (`docs/license-audit.md` §3.1 Facebook / Audiobox row).
//!   The M2-13 gate passes commercially *and* the FR-MD-09 attribution
//!   surface activates (Meta / Facebook AI Research attribution, mirror
//!   of the Kyutai `AttributionRequired` templates).
//! - Model category: `classification` (**first Vokra converter with this
//!   category** — audiobox-aesthetics is an audio-quality regression
//!   head, distinct from ASR / TTS / codec / speaker / emotion / s2s /
//!   tts / bert; silently sharing an existing category would misroute a
//!   downstream catalog consumer that ranks-by-category).
//!
//! # BF16 pass-through (mirror of `wespeaker` / `emotion2vec` / …)
//!
//! Even though upstream ships F32 today, the pass-through arm accepts
//! F32 / F16 / BF16 verbatim so a future distilled / BF16 fine-tune of
//! the same architecture converts through the same entry without a
//! silent widen. BF16 lands as GGUF type 30 (`GgmlType::BF16`) — the
//! runtime widens BF16 → f32 losslessly via
//! `vokra-core::gguf::quant::decode_bf16` (`bits << 16` is exact).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / WeSpeaker contract). Real-weight
//! binding is a follow-up wave gated on the upstream tensor-name
//! manifest fetch + license §3.1 sign-off; this converter passes every
//! float tensor through unchanged so a future
//! `AudioboxAestheticsWeights::from_gguf` can walk the same names.
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream `audiobox-aesthetics` Python
//! pipeline is deferred to owner (`docs/license-audit.md` §3.1
//! sign-off) — this converter provides the byte-parallel GGUF surface
//! only.
//!
//! # No ONNX (permanent)
//!
//! Audiobox-Aesthetics is distributed as safetensors + a Python
//! pipeline; this converter **never** touches ONNX (FR-LD-05).
//! The pipeline is re-implemented natively in a future
//! `crates/vokra-models/src/audiobox_aesthetics/` module (whisper.cpp 型
//! self re-implementation, CLAUDE.md 設計判断 4).

// Skeleton-only allowance: the public API surface is exercised by the
// in-module tests and will be wired to the CLI + `pub use` re-export in
// the same land — this attribute is removed as soon as that wiring lands.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Audiobox-Aesthetics GGUFs. Distinct from
/// every sibling arch tag — audiobox-aesthetics is a wav2vec2-backboned
/// 5-dim quality regression head, and silently sharing an arch would
/// mis-route the runtime dispatch (an ASR / speaker / emotion path
/// would try to interpret the projection head as its own).
pub(crate) const ARCH: &str = "audiobox-aesthetics";

/// `vokra.model.name` for the canonical Audiobox-Aesthetics GGUF.
pub(crate) const NAME: &str = "audiobox-aesthetics";

/// `vokra.model.category` value — `"classification"`. This is the
/// **first Vokra converter with this category tag**; sibling categories
/// today are `asr` / `tts` / `codec` / `speaker` / `emotion` / `s2s` /
/// `bert` / `vad`. Category is a taxonomy tag orthogonal to `arch`
/// (the runtime dispatches on arch, not category); zoo / catalog
/// surfaces group by category so a per-axis quality regressor is
/// visibly distinct from a per-label emotion classifier.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub(crate) const MODEL_CATEGORY: &str = "classification";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the
/// artifact back to its serving location without parsing the free-text
/// `vokra.provenance.source`.
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub(crate) const UPSTREAM_HF: &str = "facebook/audiobox-aesthetics";

/// The FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with `NOTICE` and
/// the `docs/license-audit.md` Meta / Audiobox row. CC-BY 4.0 requires
/// attribution on display / distribution; this text is what the runtime
/// + catalog generator surface verbatim.
pub(crate) const AUDIOBOX_AESTHETICS_ATTRIBUTION_TEXT: &str = "This application uses the Audiobox \
     Aesthetics model (wav2vec2 SSL backbone + 5-layer projection MLP head predicting \
     BALANCED / CONTENT_ENJOYMENT / CONTENT_USEFULNESS / PRODUCTION_COMPLEXITY / \
     PRODUCTION_QUALITY audio-quality axes; arXiv:2502.05139). Model weights are licensed \
     under CC-BY 4.0 (attribution required; commercial use permitted). Copyright (c) Meta / \
     Facebook AI Research. Source: \
     https://github.com/facebookresearch/audiobox-aesthetics / \
     https://huggingface.co/facebook/audiobox-aesthetics";

/// Outcome of an Audiobox-Aesthetics conversion.
///
/// Mirrors [`crate::models::wespeaker::WespeakerReport`]'s counter
/// contract (leading `read` count + `written`/`skipped_non_float` split
/// plus a BF16 subset counter). `read == written + skipped_non_float`
/// is an invariant preserved by [`convert_audiobox_aesthetics_file`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AudioboxAestheticsReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all pass
    /// through byte-for-byte under their upstream safetensors name;
    /// upstream today is F32 per `config.json.precision: "32"`, but the
    /// BF16 arm accepts a future distilled fine-tune of the same
    /// architecture without a silent widen).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time; kept
    /// for symmetry with the sibling reports).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Zero on the upstream F32 release; non-zero
    /// only for a future BF16 fine-tune.
    pub bf16_passthrough: usize,
}

/// File-based Audiobox-Aesthetics converter
/// (`vokra-cli convert --model audiobox-aesthetics`).
///
/// Reads `input` (upstream `facebook/audiobox-aesthetics`
/// `model.safetensors`), writes a Vokra GGUF to `output`. `license`
/// overrides the default `cc-by-4.0` provenance stamp (the same
/// `convert_file_licensed` override mechanism the Whisper / kokoro
/// family paths use); pass `None` to keep the built-in `cc-by-4.0`
/// stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_audiobox_aesthetics_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<AudioboxAestheticsReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Category / upstream-HF stamps — not covered by `stamp_provenance`
    // (which handles the SPDX + class + model_id + source group only),
    // so written directly.
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = cc-by-4.0 (upstream
    // `facebook/audiobox-aesthetics` cardData `license: cc-by-4.0`,
    // primary-source verified 2026-07-30 via
    // `https://huggingface.co/api/models/facebook/audiobox-aesthetics`).
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
            "facebook/audiobox-aesthetics (wav2vec2 backbone + 5-dim quality regression, cc-by-4.0)",
        ),
    );
    // FR-MD-09 attribution surface — CC-BY 4.0 requires attribution on
    // *display / distribution*; we stamp the text so the runtime + the
    // catalog generator surface it verbatim.
    vokra_core::stamp_attribution(&mut b, AUDIOBOX_AESTHETICS_ATTRIBUTION_TEXT);

    let mut report = AudioboxAestheticsReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // Upstream today is all F32 (per `config.json.precision: "32"`), but
    // the BF16 arm accepts a future distilled fine-tune of the same
    // architecture; runtime widens BF16 → f32 losslessly at load via
    // `vokra-core::gguf::quant::decode_bf16`.
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
    use vokra_core::gguf::{GgmlType, GgufFile};

    /// Builds a single-F32-tensor safetensors buffer with a caller-
    /// supplied raw payload — upstream is F32 today, so this pins the
    /// primary conversion path.
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

    /// Builds a single-BF16-tensor safetensors buffer — used by the
    /// future-fine-tune arm to prove the pass-through never silently
    /// widens.
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

    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-audiobox-aesthetics-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    /// Arch string is the shared handshake with a future
    /// `vokra-models::audiobox_aesthetics::EXPECTED_ARCH`; if it drifts,
    /// the runtime binder cannot recognise this GGUF.
    #[test]
    fn arch_constant_is_stable() {
        assert_eq!(ARCH, "audiobox-aesthetics");
    }

    /// The category tag must be `"classification"` — the first Vokra
    /// converter with this taxonomy. Silently changing it to
    /// `"emotion"` / `"speaker"` / etc. would misroute a downstream
    /// catalog consumer that ranks-by-category.
    #[test]
    fn category_is_classification() {
        assert_eq!(MODEL_CATEGORY, "classification");
    }

    #[test]
    fn f32_tensor_passes_through_verbatim_with_cc_by_4_0_stamp() {
        // Non-zero F32 payload so a silent widen / downcast would flip a
        // fence rather than trivially round-trip a zero buffer.
        let f32_vals: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Mirror a plausible upstream wav2vec2 feature-extractor tensor
        // name (`feature_extractor.conv_layers.0.conv.weight`).
        let input_bytes = safetensors_one_f32(
            "feature_extractor.conv_layers.0.conv.weight",
            &[2, 3],
            &f32_bytes,
        );
        let input_path = write_temp("f32-in", &input_bytes);
        let output_path = write_temp("f32-out", &[]);

        let report = convert_audiobox_aesthetics_file(&input_path, &output_path, None)
            .expect("convert_audiobox_aesthetics_file must accept a well-formed F32 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 must NOT increment the BF16 counter"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("feature_extractor.conv_layers.0.conv.weight")
            .expect("F32 tensor present in output");
        assert_eq!(info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), f32_bytes.as_slice());

        // Provenance chunks land: arch / name / category / upstream_hf /
        // license / class / attribution.
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

        // FR-MD-09 attribution: text is non-empty, Meta / Facebook AI
        // Research named, cc-by-4.0-labelled.
        let attr = file
            .get(chunks::KEY_PROVENANCE_ATTRIBUTION)
            .and_then(|v| v.as_str())
            .expect("attribution present");
        assert!(
            (attr.contains("Meta") || attr.contains("Facebook")) && attr.contains("CC-BY 4.0"),
            "attribution names Meta/Facebook + CC-BY 4.0: {attr}"
        );

        // M2-13 gate: AttributionRequired passes the commercial policy
        // WITHOUT a research flag.
        let res = vokra_core::resolve_license_class(&file);
        assert_eq!(res.class, LicenseClass::AttributionRequired);
        assert!(!res.is_research_only());

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// Even though upstream is F32 today, the pass-through arm must
    /// keep BF16 verbatim for a future distilled fine-tune — silent
    /// widen would let a regression slip in undetected.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12);

        let input_bytes = safetensors_one_bf16("projector.layers.0.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_audiobox_aesthetics_file(&input_path, &output_path, None)
            .expect("BF16 pass-through must succeed");
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1, "BF16 subset counter fires");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("projector.layers.0.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16"
        );
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload byte-identical to input"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
