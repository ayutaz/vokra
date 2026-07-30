//! **Kyutai TTS 1.6B EN/FR** (`kyutai/tts-1.6b-en_fr`, cc-by-4.0):
//! safetensors checkpoint → GGUF conversion (TIER 2 land, 2026-07-30).
//!
//! Input: the upstream `kyutai/tts-1.6b-en_fr` release —
//! `dsm_tts_1e68beda@240.safetensors` (the TTS moshi variant) plus, on
//! disk, the sibling `tokenizer-e351c8d8-checkpoint125.safetensors`
//! (Mimi codec, converted separately by `--model mimi`) and
//! `tokenizer_spm_8k_en_fr_audio.model` (SentencePiece text tokenizer).
//! Output: a GGUF carrying every float tensor verbatim under its upstream
//! safetensors name, plus the `vokra.kyutai_tts.*` /
//! `vokra.provenance.*` / `vokra.model.*` metadata chunks that a future
//! native `vokra-models::kyutai_tts::*` implementation will read.
//!
//! # Distinct arch from `ModelKind::KyutaiStt`
//!
//! Kyutai ships two separate model families under the same "delayed
//! streams modeling" umbrella (arXiv:2410.00037) — **STT** (speech →
//! text) already lives at `ModelKind::KyutaiStt`, and **TTS** (text →
//! speech) lands here as `ModelKind::KyutaiTts`. They share the Moshi /
//! Helium transformer topology (RMSNorm `rms_norm_f32`, RoPE, SiLU
//! gating, depformer) but the two directions are wired differently:
//!
//! - **STT**: `dep_q=0` (text-only prediction), no cross-attention, no
//!   conditioners — the Mimi audio codebooks are inputs, text is output.
//! - **TTS**: `dep_q=32` (audio-token prediction, one per Mimi codebook),
//!   cross-attention on `conditioners.speaker_wavs` (512-d reference
//!   speaker embedding), LUT-typed `conditioners.cfg` (7-bin CFG scale
//!   selector), LUT-typed `conditioners.control` (2048-d unified control
//!   token), `depformer_multi_linear=true`, `demux_second_stream=true`,
//!   `depformer_weights_per_step_schedule` [0,1,2,3,4,5,6,7,
//!   8×8,9×8,10×8] (multi-step weight sharing).
//!
//! Silently sharing an arch tag would mis-route the runtime dispatch (an
//! STT decoder would try to predict text where audio codebooks live, and
//! vice versa); the two are landed as distinct
//! `ModelKind` variants for that reason.
//!
//! # HF / licence / category (all primary-source verified 2026-07-30)
//!
//! - Upstream HF: `kyutai/tts-1.6b-en_fr` (recorded under
//!   `vokra.provenance.upstream_hf`).
//! - HF cardData `license: cc-by-4.0` — `LicenseClass::AttributionRequired`
//!   (`docs/license-audit.md` §3.1 Kyutai row).
//!   The M2-13 gate passes commercially *and* the FR-MD-09 attribution
//!   surface activates (attribution text below is Kyutai-named and
//!   cc-by-4.0-labelled, mirror of `moshi::MOSHI_ATTRIBUTION_TEXT` and
//!   `kyutai_stt::KYUTAI_STT_ATTRIBUTION_TEXT`).
//! - Model category: `tts` (recorded under `vokra.model.category`).
//!
//! # BF16 pass-through (mirror of `kyutai_stt` / `moshi` / `voxtral` /
//! `qwen3_tts` / `vibevoice` / `voxcpm2`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30 (`GgmlType::BF16`)
//! — the same posture as the sibling converters. No convert-time
//! widening; runtime widens BF16 → f32 losslessly via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is
//! the top 16 bits of an f32 — `bits << 16` is exact). Every F32 / F16
//! tensor passes through under its upstream name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Moshi / Kyutai STT
//! contract). Real-weight binding is a follow-up wave gated on the
//! upstream tensor-name manifest fetch + license §3.1 sign-off; this
//! converter passes every F32 / F16 / BF16 tensor through unchanged so
//! a future `KyutaiTtsWeights::from_gguf` can walk the same names.
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream `kyutai/tts-1.6b-en_fr`
//! Python pipeline is deferred to owner
//! (`docs/license-audit.md` §3.1 sign-off) — this converter provides the
//! byte-parallel GGUF surface only.
//!
//! # No ONNX (permanent)
//!
//! Kyutai TTS ships as safetensors + a Python pipeline; this converter
//! **never** touches ONNX (FR-LD-05). The pipeline is re-implemented
//! natively in a future `crates/vokra-models/src/kyutai_tts/` module
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

// Skeleton-only allowance: the public API surface is exercised by the
// in-module tests and will be wired to the CLI + `pub use` re-export in
// the same land — this attribute is removed as soon as that wiring lands.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Kyutai TTS GGUFs — kept as `kyutai-tts`
/// (distinct from `kyutai-stt`, the STT sibling at `ModelKind::KyutaiStt`).
pub(crate) const ARCH: &str = "kyutai-tts";

/// `vokra.model.name` for the canonical Kyutai TTS 1.6B EN/FR GGUF.
pub(crate) const NAME: &str = "kyutai-tts-1.6b-en-fr";

/// `vokra.model.category` value — `"tts"` (distinct from `"asr"` /
/// `"codec"` / `"speaker"` / `"emotion"` / `"classification"` / `"s2s"`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub(crate) const MODEL_CATEGORY: &str = "tts";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the
/// artifact back to its serving location without parsing the free-text
/// `vokra.provenance.source`.
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub(crate) const UPSTREAM_HF: &str = "kyutai/tts-1.6b-en_fr";

/// The FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with `NOTICE` §5
/// and the `docs/license-audit.md` Kyutai row (mirror of the moshi /
/// kyutai_stt attribution templates; Kyutai's whole audio/text release
/// family ships CC-BY-4.0).
pub(crate) const KYUTAI_TTS_ATTRIBUTION_TEXT: &str = "This application uses the Kyutai TTS-1.6B \
     EN/FR model (Helium temporal transformer + depformer TTS, English + French, over Mimi audio \
     codebooks). Model weights are licensed under CC-BY 4.0 (attribution required; commercial use \
     permitted). Copyright (c) Kyutai. Source: \
     https://github.com/kyutai-labs/delayed-streams-modeling / \
     https://huggingface.co/kyutai/tts-1.6b-en_fr";

/// Outcome of a Kyutai TTS conversion.
///
/// Mirrors [`crate::models::wespeaker::WespeakerReport`]'s counter
/// contract (leading `read` count + `written`/`skipped_non_float` split
/// plus a BF16 subset counter). `read == written + skipped_non_float`
/// is an invariant preserved by [`convert_kyutai_tts_file`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct KyutaiTtsReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all pass
    /// through byte-for-byte under their upstream safetensors name).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any tensor
    /// reaching this counter would signal a reader change upstream;
    /// kept for symmetry with the sibling `qwen3_tts` / `wespeaker` /
    /// `moshi` reports).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Mirrors
    /// `moshi::MoshiReport::bf16_passthrough` /
    /// `kyutai_stt::KyutaiSttReport::bf16_passthrough`.
    pub bf16_passthrough: usize,
}

/// File-based Kyutai TTS converter
/// (`vokra-cli convert --model kyutai-tts`).
///
/// Reads `input` (upstream `kyutai/tts-1.6b-en_fr`
/// `dsm_tts_1e68beda@240.safetensors`), writes a Vokra GGUF to `output`.
/// `license` overrides the default `cc-by-4.0` provenance stamp (the
/// same `convert_file_licensed` override mechanism the Whisper / kokoro
/// family paths use); pass `None` to keep the built-in `cc-by-4.0`
/// stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_kyutai_tts_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<KyutaiTtsReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Category / upstream-HF stamps — not covered by `stamp_provenance`
    // (which handles the SPDX + class + model_id + source group only),
    // so written directly. Consumers pick a load path by category and
    // trace the artifact back to its serving location by upstream_hf.
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = cc-by-4.0 (upstream
    // `kyutai/tts-1.6b-en_fr` model card cardData `license: cc-by-4.0`,
    // primary-source verified 2026-07-30 via
    // `https://huggingface.co/api/models/kyutai/tts-1.6b-en_fr`).
    // `license` overrides for callers who obtained the weight under a
    // different SPDX (see `convert_file_licensed` in `lib.rs`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => ("cc-by-4.0".to_owned(), LicenseClass::AttributionRequired),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some("kyutai/tts-1.6b-en_fr (Helium TTS over Mimi audio codebooks, EN + FR, cc-by-4.0)"),
    );
    // FR-MD-09 attribution surface — CC-BY 4.0 requires attribution on
    // *display / distribution*; we stamp the text so the runtime + the
    // catalog generator surface it verbatim (the same conduit
    // `moshi::MOSHI_ATTRIBUTION_TEXT` /
    // `kyutai_stt::KYUTAI_STT_ATTRIBUTION_TEXT` use).
    vokra_core::stamp_attribution(&mut b, KYUTAI_TTS_ATTRIBUTION_TEXT);

    let mut report = KyutaiTtsReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30), same posture as
    // qwen3_tts / vibevoice / voxcpm2 / kyutai_stt / moshi; runtime
    // widens BF16 → f32 exactly at load via
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

    /// Builds a single-BF16-tensor safetensors buffer with a
    /// caller-supplied raw payload (mirror of the wespeaker test fixture).
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
            "vokra-kyutai-tts-{kind}-{}-{}.bin",
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
    /// `vokra-models::kyutai_tts::EXPECTED_ARCH`; if it drifts, the
    /// runtime binder cannot recognise this GGUF. Pinned here rather
    /// than the runtime side because the runtime module is a future wave
    /// and the two crates only share `vokra-core`.
    #[test]
    fn arch_constant_is_stable() {
        assert_eq!(ARCH, "kyutai-tts");
    }

    /// The arch string must differ from the STT sibling — silently
    /// sharing would misroute the runtime dispatch between text-out
    /// (STT) and audio-out (TTS) paths.
    #[test]
    fn arch_differs_from_kyutai_stt() {
        assert_ne!(ARCH, crate::models::kyutai_stt::ARCH);
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim_with_cc_by_4_0_stamp() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity assert
        // catches any silent widen / downcast attempt (zeroed payloads
        // would round-trip trivially through F32 / F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror a plausible upstream Moshi/Helium temporal transformer
        // tensor name (e.g. `transformer.layers.0.self_attn.in_proj_weight`)
        // so the round-trip exercises a realistic string.
        let input_bytes = safetensors_one_bf16(
            "transformer.layers.0.self_attn.in_proj_weight",
            &[2, 3],
            &bf16,
        );
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_kyutai_tts_file(&input_path, &output_path, None)
            .expect("convert_kyutai_tts_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror qwen3_tts / vibevoice / voxcpm2 / moshi / kyutai_stt)"
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
            .tensor_info("transformer.layers.0.self_attn.in_proj_weight")
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

        // Provenance chunks land: arch / name / category / upstream_hf /
        // license = cc-by-4.0 / class = AttributionRequired /
        // attribution text carries the Kyutai + cc-by-4.0 handshake.
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

        // FR-MD-09 attribution surface: text is non-empty, Kyutai-named,
        // and cc-by-4.0-labelled (mirror of the moshi round-trip test).
        let attr = file
            .get(chunks::KEY_PROVENANCE_ATTRIBUTION)
            .and_then(|v| v.as_str())
            .expect("attribution present");
        assert!(
            attr.contains("Kyutai") && attr.contains("CC-BY 4.0"),
            "attribution names Kyutai + CC-BY 4.0: {attr}"
        );

        // The M2-13 gate resolves AttributionRequired and passes the
        // strict (commercial) policy WITHOUT a research flag — CC-BY 4.0
        // is commercial-OK.
        let res = vokra_core::resolve_license_class(&file);
        assert_eq!(res.class, LicenseClass::AttributionRequired);
        assert!(!res.is_research_only());

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// Caller-supplied `--license` overrides the default cc-by-4.0
    /// stamp (mirror of the wespeaker test), for callers who
    /// legitimately hold the weight under a distinct SPDX id.
    #[test]
    fn license_override_stamps_caller_spdx() {
        let bf16 = vec![0u8; 12]; // 6 BF16 elements
        let input_bytes = safetensors_one_bf16("transformer.layers.0.norm1.alpha", &[2, 3], &bf16);
        let input_path = write_temp("override-in", &input_bytes);
        let output_path = write_temp("override-out", &[]);

        let _ = convert_kyutai_tts_file(&input_path, &output_path, Some("apache-2.0"))
            .expect("license override should succeed");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "license override took effect"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "class re-derived from the SPDX override"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
