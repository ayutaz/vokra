//! **NeuTTS Air** (`neuphonic/neutts-air`, apache-2.0):
//! safetensors → GGUF conversion (SoTA plan candidate wave,
//! 2026-08-04).
//!
//! Input: the upstream `neuphonic/neutts-air` release
//! (`huggingface.co/neuphonic/neutts-air`) — a **single-file BF16
//! safetensors** (~1.40 GB / 747,930,496 BF16 params, HF API
//! primary-source verified 2026-08-04) Qwen2-family 0.5B LLM
//! backbone (Qwen2 hidden_size=896 / intermediate_size=4864 /
//! 24 layers / 14 attention heads / 2 KV heads / RoPE θ=1e6 /
//! vocab_size=217652) fine-tuned to emit NeuCodec audio tokens
//! after text tokens — the LM half of an on-device instant-voice-
//! clone TTS pipeline whose audio-token side is decoded by the
//! sibling `super::neucodec::NeucodecVariant::Base`
//! (`neuphonic/neucodec`) codec that already ships in the Vokra
//! catalog. Output: a Vokra GGUF carrying every float tensor
//! plus the `vokra.model.*` / `vokra.provenance.*` metadata
//! chunks the future native NeuTTS Air runtime side will read.
//!
//! # Model card
//!
//! - **HF path**: `neuphonic/neutts-air`
//! - **License SPDX**: `apache-2.0` (weight + code, end-to-end;
//!   Neuphonic ships both under the same permissive tag per HF
//!   cardData primary source 2026-08-04)
//! - **Not gated**: `gated: False` / `private: False` per HF API
//!   `https://huggingface.co/api/models/neuphonic/neutts-air`
//!   (2026-08-04) — no acknowledgement flow required.
//! - **Category**: `tts` — NeuTTS Air is a text-to-speech LM
//!   backbone (Qwen2 causal-LM emitting NeuCodec audio tokens
//!   conditioned on text + a short reference audio prompt for
//!   instant voice cloning). Complements the existing Kokoro /
//!   piper-plus / CosyVoice2 / CSM TTS stack; sibling audio
//!   codec is `super::neucodec` (already published as
//!   `vokra/neucodec`).
//! - **Base model**: Qwen2 0.5B (with vocab extended from 151,936
//!   to 217,652 to carry the NeuCodec audio-token space) —
//!   `config.architectures = ["Qwen2ForCausalLM"]`
//!   / `model_type = "qwen2"` per upstream `config.json`
//!   primary source 2026-08-04.
//! - **Distinct arch tag**: `neutts-air` — silently sharing an
//!   arch tag with a sibling TTS model (`kokoro` / `piper-plus` /
//!   `cosyvoice2` / `cosyvoice3` / `csm` / `moshi` / `voxcpm2` /
//!   `dia` / `zonos` / `chatterbox` / `bark` / `styletts2` /
//!   `vibevoice` / `qwen3-tts` / `sbv2` / `chattts` / `irodori` /
//!   `melotts` / `vits-ja` / `vieneu` / `parler`) would mis-route
//!   the runtime dispatch — NeuTTS Air's Qwen2-backbone-emits-
//!   NeuCodec-tokens topology is unique enough to warrant its
//!   own routing arm (FR-EX-08). Distinct also from the sibling
//!   Qwen2-derived TTS `qwen3-tts` (different backbone size,
//!   different vocab extension strategy, different codec
//!   companion).
//!
//! # BF16 posture
//!
//! Follows the bicodec / neucodec / focalcodec / xcodec2 / miocodec
//! landed pattern: F32 / F16 / BF16 all pass through **verbatim**
//! under their upstream safetensors names. BF16 is emitted as GGUF
//! type 30 (`GgmlType::BF16`); the runtime widens BF16 → f32
//! losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is
//! the top 16 bits of an f32 — `bits << 16` is exact). No convert-
//! time widening, no silent F16 downcast (FR-EX-08). Upstream
//! ships **BF16 end-to-end** (`torch_dtype: bfloat16` per
//! `config.json` + HF API-verified 2026-08-04: `"safetensors":
//! {"parameters": {"BF16": 747930496}}`), so every real tensor
//! rides the BF16 pass-through arm; the F32 / F16 arms are
//! defensive today for future re-quantized derivatives.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the neucodec / bicodec / focalcodec / xcodec2 / miocodec /
//! kokoro / cosyvoice2 / chatterbox / qwen3-tts / voxcpm2 /
//! vibevoice contract). Real-weight binding is a follow-up wave
//! gated on the upstream tensor-name manifest fetch; this
//! converter passes every F32 / F16 / BF16 tensor through unchanged
//! so a future `NeuTtsAirWeights::from_gguf` can walk the same
//! names.
//!
//! # Real-weight parity
//!
//! Deferred to the owner sign-off queue (`docs/license-audit.md`
//! §3.1). This converter is a native-side skeleton that pins the
//! metadata contract (arch / name / category / upstream-HF /
//! license) plus the BF16 pass-through invariant so a future
//! `NeuTtsAir::from_gguf` can bind against the same upstream
//! tensor names once real weights are audited. Loud-partial
//! landing pattern per RMVPE / Charsiu / MOSS-Audio-Tokenizer /
//! MioCodec precedent.
//!
//! # Upstream-GGUF sibling exclusion (FR-LD-05)
//!
//! The upstream `neuphonic/neutts-air` repo also ships a
//! foreign GGUF (`neutss-air-BF16.gguf`, ~1.40 GB — note the
//! upstream filename typo `neutss` vs `neutts`) alongside the
//! `model.safetensors` this converter consumes. **Vokra runtime
//! never loads a foreign GGUF** (FR-LD-05, permanent); the
//! ONNX / foreign-GGUF boundary lives strictly in the
//! offline-conversion tools, and only the safetensors path is
//! walked here. The upstream Q8 / Q4 GGUF quantizations are
//! separate HF repos (`neuphonic/neutts-air-q8-gguf` /
//! `-q4-gguf`) whose bytes are irrelevant to this converter —
//! Vokra's own K-quants land through the sibling
//! `vokra-convert` quantize pipeline against the safetensors
//! after re-conversion.
//!
//! # No ONNX (permanent)
//!
//! NeuTTS Air ships `model.safetensors` + `config.json` +
//! `tokenizer.*` + a foreign `neutss-air-BF16.gguf` sibling
//! directly (no ONNX mirror on the upstream repo). This
//! converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in a future
//! `crates/vokra-models/src/neutts_air/` module (whisper.cpp 型
//! self re-implementation, CLAUDE.md 設計判断 4). Tokenizer
//! embedding, config-side hparams (RoPE θ / KV head split /
//! sliding-window flag), and NeuCodec-token vocab-slot mapping
//! land in that follow-up wave alongside the runtime binder.

// Skeleton-only allowance: the public API (`convert_neutts_air_file`,
// `NeuTtsAirReport`, `KEY_*` / `MODEL_CATEGORY` / `UPSTREAM_HF` /
// `DEFAULT_LICENSE_SPDX`) is exercised by the in-module tests +
// lib.rs `convert_file` dispatch; this attribute is removed once
// the runtime `NeuTtsAirWeights::from_gguf` binding lands and
// starts consuming the constants directly (miocodec / neucodec /
// bicodec / focalcodec skeleton precedent).
#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for NeuTTS Air GGUFs — kept intentionally distinct
/// from every sibling TTS model so the runtime dispatch cannot silently
/// route a NeuTTS Air artifact through a Kokoro / piper-plus / CosyVoice2
/// / CSM / Moshi / VoxCPM2 / Qwen3-TTS / VibeVoice / Chatterbox / Bark /
/// StyleTTS2 / Dia / Zonos / SBV2 / ChatTTS / Irodori / MeloTTS /
/// VITS-JA / VieNeu / Parler decoder (FR-EX-08).
pub const ARCH: &str = "neutts-air";

/// `vokra.model.name` value for the canonical `neuphonic/neutts-air`
/// release. Matches the publish repo slug spelling
/// (`vokra/neutts-air` — HF repo naming = dashes only, lowercase).
pub const NAME: &str = "neutts-air";

/// `vokra.model.category` key — TTS bucket for the artifact.
///
/// Kept as a local constant rather than a `vokra_core::gguf::chunks::*`
/// re-export because it is not yet part of the shared
/// `vokra_core::gguf::chunks` surface (mirrors the bicodec / neucodec /
/// focalcodec / miocodec convention).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Category string written into [`KEY_MODEL_CATEGORY`].
pub const MODEL_CATEGORY: &str = "tts";

/// `vokra.provenance.upstream_hf` key — HF repo path of the upstream
/// weight.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Value written under [`KEY_PROVENANCE_UPSTREAM_HF`]. Preserves the
/// upstream lower-case spelling — the HF repo slug is case-sensitive
/// and the primary source for the model-card generator.
pub const UPSTREAM_HF: &str = "neuphonic/neutts-air";

/// Default weight-license SPDX. Verified 2026-08-04 via HF cardData
/// API primary source (`api/models/neuphonic/neutts-air`
/// → `license: apache-2.0`). May be overridden via the `license`
/// argument to [`convert_neutts_air_file`] (the whisper / kokoro /
/// neucodec / miocodec override pattern).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

/// Advisory source note stamped alongside the license into the
/// `vokra.provenance.source` chunk.
const PROVENANCE_SOURCE_NOTE: &str = "neuphonic/neutts-air (Qwen2 0.5B LLM backbone emitting NeuCodec audio tokens, \
     on-device instant voice cloning, apache-2.0 end-to-end; sibling codec = neuphonic/neucodec)";

/// Outcome of a NeuTTS Air conversion.
///
/// Mirrors the field set on the sibling BF16-pass-through converters
/// (`super::bicodec::BicodecReport`,
/// `super::neucodec::NeucodecReport`,
/// `super::focalcodec::FocalcodecReport`,
/// `super::miocodec::MioCodecReport`) — the `read` counter lets the
/// caller distinguish a zero-tensor safetensors file from a zero-write
/// outcome caused by every tensor being quantized (defensive — the
/// safetensors reader rejects unknown dtypes at parse time today, so
/// anything reaching the non-float arm is a real regression on that
/// reader).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NeuTtsAirReport {
    /// Total upstream tensors observed in the safetensors input.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three
    /// go through the same byte-copy arm per the accepted BF16 pass-
    /// through posture).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time;
    /// anything that reaches this arm is a quantized dtype the runtime
    /// is not expected to consume).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens
    /// BF16 → f32 losslessly at load via `decode_bf16` (`bits << 16`
    /// is exact — BF16 is the top 16 bits of an f32). Upstream is
    /// BF16 end-to-end (HF-verified 2026-08-04: 747,930,496 BF16
    /// params) so this counter should match `written` on the real
    /// checkpoint.
    pub bf16_passthrough: usize,
}

/// Convert a `neuphonic/neutts-air` safetensors checkpoint into a
/// Vokra GGUF.
///
/// `input` is the upstream `model.safetensors` path; the emitted GGUF
/// is written to `output`. `license` overrides the raw SPDX string
/// stamped into `vokra.provenance.license` — the default is
/// `DEFAULT_LICENSE_SPDX` (`"apache-2.0"`), matching the Neuphonic
/// weight card at `huggingface.co/neuphonic/neutts-air`. Pass
/// `Some(other_spdx)` when the immediate redistribution source has
/// re-tagged the artifact (mirror of the neucodec / bicodec /
/// focalcodec / miocodec override pattern).
///
/// The upstream sibling `neutss-air-BF16.gguf` (~1.40 GB, foreign
/// GGUF) is NOT processed — FR-LD-05 forbids Vokra runtime from
/// loading foreign GGUFs, and this converter's contract is
/// safetensors-in / Vokra-GGUF-out only. Quantization to Vokra's own
/// K-quants runs through the separate `vokra-convert` quantize
/// pipeline against the Vokra GGUF produced here.
///
/// # Errors
///
/// - I/O reading `input` or writing `output` propagates as
///   [`ConvertError::Io`].
/// - Safetensors parse failure propagates as [`ConvertError::Parse`].
/// - GGUF serialization failure propagates as the `From<GgufError>`
///   impl on `ConvertError`.
pub fn convert_neutts_air_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<NeuTtsAirReport, ConvertError> {
    // NeuTTS Air is ~1.40 GB single-file BF16 safetensors (747.9M
    // BF16 params, HF-verified 2026-08-04). Well within the sibling
    // non-streaming BF16 pass-through posture (~1 order of magnitude
    // smaller than the streaming-mandated Moshi 14 GiB tier that
    // requires the `MappedTextBlocks` / `restamp_provenance` mmap
    // path), so a plain `std::fs::read` is safe on M1 iMac 16 GB.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. The `license` param overrides the raw SPDX string
    // (`vokra.provenance.license`) and — when overridden — re-derives
    // the class through `LicenseClass::from_license_str` so the
    // compliance gate stays honest (a caller who overrides to a
    // non-permissive SPDX would otherwise get a silent Permissive
    // verdict). `None` keeps the Neuphonic default (apache-2.0 →
    // Permissive) that matches the upstream weight card.
    let license_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let class = match license {
        Some(_) => LicenseClass::from_license_str(license_spdx),
        None => LicenseClass::Permissive,
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        license_spdx,
        Some(NAME),
        Some(PROVENANCE_SOURCE_NOTE),
    );

    let mut report = NeuTtsAirReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the bicodec /
    // neucodec / focalcodec / miocodec ADR-A_passthrough posture; the
    // runtime widens BF16 → f32 exactly at load via the single choke
    // point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`
    // (`bits << 16` is exact — BF16 is the top 16 bits of an f32).
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

    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Builds a single-BF16-tensor safetensors buffer with a
    /// caller-supplied raw payload. Panics if
    /// `bf16_bytes.len() != shape × 2` — that would declare an invalid
    /// safetensors header the reader would reject.
    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(
            bf16_bytes.len(),
            elems as usize * 2,
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

    /// Builds a single-F32-tensor safetensors buffer — defensive
    /// coverage for a hypothetical future F32 re-release.
    fn safetensors_one_f32(name: &str, shape: &[u64], f32_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(
            f32_bytes.len(),
            elems as usize * 4,
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
            "vokra-neutts-air-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    /// Upstream NeuTTS Air is BF16 (HF API verified 2026-08-04
    /// `"safetensors": {"parameters": {"BF16": 747930496}}`) —
    /// this test pins the primary code path.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity
        // assert catches any silent widen / downcast attempt (zeroed
        // payloads would round-trip trivially through F32 / F16 widen
        // too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror a realistic upstream tensor name from a Qwen2-family
        // LLM backbone's attention stack (upstream config primary
        // source: `model_type: qwen2`, `num_hidden_layers: 24`).
        let input_bytes =
            safetensors_one_bf16("model.layers.0.self_attn.q_proj.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_neutts_air_file(&input_path, &output_path, None)
            .expect("convert_neutts_air_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror bicodec / neucodec / focalcodec / miocodec)"
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
            .tensor_info("model.layers.0.self_attn.q_proj.weight")
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

        // Provenance / category chunks landed (task-spec pins).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
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
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(MODEL_CATEGORY),
            "vokra.model.category must be `tts`",
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// Defensive test — a hypothetical future F32 re-release should
    /// ride the same pass-through arm as the sibling BF16-pass-through
    /// converters (bicodec / neucodec / focalcodec / miocodec).
    #[test]
    fn f32_tensor_passes_through() {
        let f32_vals: [f32; 6] = [0.5, -0.25, 1.5, -3.0, 42.0, 0.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        let input_bytes = safetensors_one_f32("model.embed_tokens.weight", &[2, 3], &f32_bytes);
        let input_path = write_temp("f32-in", &input_bytes);
        let output_path = write_temp("f32-out", &[]);

        let report = convert_neutts_air_file(&input_path, &output_path, None)
            .expect("convert_neutts_air_file must accept an F32 checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 does not increment BF16 counter"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("model.embed_tokens.weight")
            .expect("F32 tensor present in output");
        assert_eq!(info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), f32_bytes.as_slice());

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// The `license` override must flow through to the provenance stamp
    /// and re-derive the class (guards against a silent Permissive
    /// verdict when a caller ships under a non-permissive SPDX).
    #[test]
    fn license_override_flows_through() {
        let f32_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("model.norm.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("license-in", &input_bytes);
        let output_path = write_temp("license-out", &[]);

        convert_neutts_air_file(&input_path, &output_path, Some("mit"))
            .expect("license override must succeed");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "license override must be honored"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "mit is Permissive class (same bucket as apache-2.0 default)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
