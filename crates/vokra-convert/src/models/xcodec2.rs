//! **X-Codec 2 (Llasa)**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 5 codec, 2026-07-28).
//!
//! Upstream: `HKUSTAudio/xcodec2` (HKUST-Audio). Neural audio codec paired
//! with the Llasa TTS family — the M4-16 (`xcodec2_fsq`) op-only landing
//! implemented the FSQ decode path (`crates/vokra-ops/src/fsq_codec.rs`,
//! parity fixture is synthetic vector-quantize-pytorch 1.17.8 projection);
//! this converter completes the missing "convert an upstream checkpoint into
//! a Vokra GGUF" side.
//!
//! # License posture — CC-BY-NC 4.0 default (**NonCommercial**)
//!
//! Weight redistribution default is [`LicenseClass::NonCommercial`]. The HF
//! model card at `huggingface.co/HKUSTAudio/xcodec2` carries
//! `license: cc-by-nc-4.0` on its YAML front-matter (CC-verified 2026-07-15;
//! `docs/license-audit.md` §3.1 sign-off 2026-07-23 yousan =
//! ☑ Research-only). The code layer at `github.com/zhenye234/X-Codec-2.0`
//! remains MIT — but this converter is a weight-redistribution surface, and
//! the weight-distribution repo governs the license of the redistributed
//! artifact. Callers may override at the outer
//! `convert_file --license <spdx>` boundary when they legitimately hold the
//! weight under a distinct SPDX id (the same pattern Whisper /
//! kokoro / vits-ja use).
//!
//! The stamped `LicenseClass::NonCommercial` activates
//! [`LicenseClass::requires_research_flag`] at load time — this is
//! **fail-closed**: an unmarked commercial-mode caller cannot silently
//! bring the weights up. That mirrors the F5-TTS / EnCodec posture already
//! landed in `license_class.rs`.
//!
//! # BF16 pass-through (mirror of `neucodec` / `step_audio2_mini` / `voxcpm2`
//! / `qwen3_tts` / `vibevoice`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30 (`GgmlType::BF16`) —
//! the same posture as the sibling codec / TTS converters. No convert-time
//! widening; runtime widens BF16 → f32 losslessly via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 = top
//! 16 bits of an f32 — `bits << 16` is exact). Every F32 / F16 tensor
//! passes through under its upstream safetensors name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim** (the
//! CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM / VibeVoice
//! / neucodec / step_audio2_mini contract). Real-weight binding is a
//! follow-up wave gated on the upstream tensor-name manifest fetch; this
//! converter passes every F32 / F16 / BF16 tensor through unchanged so a
//! future `XCodec2Weights::from_gguf` can walk the same names.
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream Python pipeline is deferred to
//! owner (`docs/license-audit.md` §3.1 sign-off). The M4-16 op-side
//! (`xcodec2_fsq`, `crates/vokra-ops/tests/parity_fsq_codec.rs`) already
//! runs the FSQ decode against the vector-quantize-pytorch 1.17.8
//! reference on a synthetic projection; a full end-to-end parity gate
//! (encode → codes → decode against `HKUSTAudio/xcodec2` published
//! reference audio) is a follow-up.
//!
//! # No ONNX (permanent)
//!
//! X-Codec 2 is distributed as safetensors + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05). The pipeline is
//! re-implemented natively in a future `crates/vokra-models/src/xcodec2/`
//! module (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for X-Codec 2 GGUFs. Distinct from every sibling arch
/// tag — X-Codec 2 is an **FSQ** codec (finite scalar quantization, single
/// FSQ level bank per codebook) vs the sibling RVQ codecs (Mimi / DAC /
/// WavTokenizer), so silently sharing an arch would mis-route the runtime
/// dispatch (RVQ vs FSQ decode paths differ; FSQ has no residual chain).
pub(crate) const ARCH: &str = "xcodec2";

/// `vokra.model.name` value written for the canonical X-Codec 2 GGUF.
pub(crate) const NAME: &str = "xcodec2";

/// `vokra.model.category` value — X-Codec 2 is a **codec** (neural audio
/// codec, encode/decode with FSQ discrete latents at 50 Hz frame rate).
/// The category chunk is a taxonomy tag orthogonal to `arch`; the runtime
/// does not dispatch on it (arch does), but it is machine-readable for
/// model-zoo / catalog surfaces (see `docs/license-audit.md`).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const MODEL_CATEGORY: &str = "codec";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the artifact
/// back to its serving location without parsing the free-text
/// `vokra.provenance.source`.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const UPSTREAM_HF: &str = "HKUSTAudio/xcodec2";

/// The default upstream weight license — `cc-by-nc-4.0`, per the HF model
/// card `license: cc-by-nc-4.0` (CC-verified 2026-07-15; sign-off
/// 2026-07-23 yousan = ☑ Research-only, `docs/license-audit.md` §3.1).
/// Callers can override at the
/// `convert_xcodec2_file(_, _, license=Some(_))` boundary when the source
/// distribution declares a different SPDX id.
const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-4.0";

/// Human-readable upstream source note stored in
/// `vokra.provenance.source` (`KEY_PROVENANCE_SOURCE`). Kept short — the
/// license machine class is carried separately in the
/// `vokra.provenance.weight_license` chunk.
const UPSTREAM_SOURCE: &str = "HKUSTAudio/xcodec2 (50 Hz FSQ codec, cc-by-nc-4.0)";

/// Outcome of an X-Codec 2 conversion. Additive counters — a non-zero
/// value on any field is a positive report; a zero `written` value means
/// the input safetensors carried no float tensors and the runtime will
/// refuse to bind any weights (FR-EX-08).
#[derive(Debug, Default)]
pub struct XCodec2Report {
    /// Total tensors observed in the input safetensors.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time, so any
    /// tensor reaching this counter would signal a reader change
    /// upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16
    /// (observability counter — the ADR pattern shared with neucodec /
    /// step_audio2_mini / qwen3_tts / vibevoice / voxcpm2 / moshi /
    /// voxtral so a latent silent-widen cannot slip in undetected).
    pub bf16_passthrough: usize,
}

/// Internal shared conversion: reads a parsed safetensors buffer, writes
/// every F32 / F16 / BF16 tensor verbatim under its upstream name, and
/// stamps the `vokra.model.*` + `vokra.provenance.*` metadata chunks.
///
/// Used by (a) [`convert_xcodec2_file`] (the standalone file-based entry)
/// and (b) the [`crate::ModelKind::XCodec2`] arm of
/// [`crate::convert_file_licensed`] (the CLI dispatch path). The two
/// entries share this function so there is exactly one place that walks
/// the tensors and stamps the license (single source of truth).
///
/// The caller is responsible for handling the `license` override at the
/// outer boundary — this function always stamps the built-in default
/// (`cc-by-nc-4.0`, [`LicenseClass::NonCommercial`]). The
/// [`crate::convert_file_licensed`] outer wrapper re-stamps the
/// `vokra.provenance.{license,weight_license,source}` chunks when the
/// caller supplied a non-default SPDX id.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, XCodec2Report), ConvertError> {
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

    // Built-in stamp = cc-by-nc-4.0 NonCommercial. The outer
    // `convert_file_licensed` layer overrides these three chunks if the
    // caller passed a distinct `--license <spdx>` — but with the built-in
    // gate the artifact fails **closed** at load time in commercial mode
    // (`LicenseClass::NonCommercial::requires_research_flag = true`), so
    // an operator who never touched the license flag cannot silently
    // bring up an NC weight in production.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::NonCommercial,
        DEFAULT_LICENSE_SPDX,
        Some(NAME),
        Some(UPSTREAM_SOURCE),
    );

    let mut report = XCodec2Report::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the sibling ADR shared with
    // neucodec / step_audio2_mini / qwen3-tts / vibevoice / voxcpm2 /
    // moshi / voxtral; the runtime widens BF16 → f32 exactly at load via
    // the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
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
    Ok((b, report))
}

/// File-based X-Codec 2 converter (standalone entry — mirror of
/// `convert_neucodec_file` / `convert_step_audio2_mini_file`).
///
/// Reads `input` (upstream `HKUSTAudio/xcodec2` `model.safetensors`),
/// writes a Vokra GGUF to `output`. `license` overrides the default
/// `cc-by-nc-4.0` provenance stamp (Whisper / kokoro-family override
/// pattern — see `convert_file_licensed` in `lib.rs`); pass `None` to
/// keep the built-in `cc-by-nc-4.0` stamp.
///
/// # Provenance defaults
///
/// - `vokra.provenance.license` = `"cc-by-nc-4.0"` (from
///   [`DEFAULT_LICENSE_SPDX`]).
/// - `vokra.provenance.weight_license` =
///   [`LicenseClass::NonCommercial`]`.as_str()`. The M2-13 runtime gate
///   refuses to load this artifact in commercial mode
///   (`LicenseClass::NonCommercial::requires_research_flag = true`) — a
///   research-only session must be opened explicitly.
/// - `vokra.provenance.model_id` = `"xcodec2"` (from [`NAME`]).
/// - `vokra.provenance.source` = the human-readable upstream note.
/// - `vokra.provenance.upstream_hf` = `"HKUSTAudio/xcodec2"`.
/// - `vokra.model.category` = `"codec"`.
///
/// # Errors
///
/// - [`ConvertError::Io`] if the input cannot be read or the output
///   cannot be written.
/// - [`ConvertError::Parse`] if the safetensors header is malformed.
/// - [`ConvertError::Gguf`] if the GGUF cannot be assembled.
pub fn convert_xcodec2_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<XCodec2Report, ConvertError> {
    let bytes = std::fs::read(input)?;
    let (mut b, report) = convert(bytes)?;

    // Standalone-entry license override: mirror the outer
    // `convert_file_licensed` logic so a caller invoking this function
    // directly (bypassing `ModelKind` dispatch) still gets the same
    // license-override semantics.
    if let Some(spdx) = license.filter(|s| !s.is_empty()) {
        let class = LicenseClass::from_license_str(spdx);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, class.as_str());
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, spdx);
        // The built-in `source` string names the converter's default
        // license (cc-by-nc-4.0); once the license is overridden that
        // parenthetical would contradict it, so restate the source
        // neutrally (same pattern as `convert_file_licensed`).
        b.add_string(
            chunks::KEY_PROVENANCE_SOURCE,
            &format!("upstream distribution source (licence {spdx} per source)"),
        );
    }

    let out_bytes = b.to_bytes()?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    /// A unique temp path — per-process id **plus** a monotonic counter so
    /// two tests in the same process never race on the same file.
    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-xcodec2-{tag}-{}-{n}",
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

    /// Builds a two-tensor safetensors buffer (F32 first, then F16) with
    /// caller-supplied payloads.
    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_bytes: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_bytes: &[u8],
    ) -> Vec<u8> {
        let f32_shape_str = f32_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let f16_shape_str = f16_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
        let header = format!(
            r#"{{"{f32_name}":{{"dtype":"F32","shape":[{f32_shape_str}],"data_offsets":[0,{f32_len}]}},"{f16_name}":{{"dtype":"F16","shape":[{f16_shape_str}],"data_offsets":[{f32_len},{total}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out.extend_from_slice(f16_bytes);
        out
    }

    /// The BF16 pass-through arm must emit GGUF type 30
    /// (`GgmlType::BF16`) with byte-identical payload — mirror of the
    /// neucodec / step_audio2_mini pin.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity assert
        // catches any silent widen / downcast (zeroed payloads would
        // round-trip trivially through F32/F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16 = bf16_bytes(&values);
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        let input_bytes = safetensors_one("codec.embed.weight", "BF16", &[2, 3], &bf16);
        let input = tmp_path("bf16-in");
        let output = tmp_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_xcodec2_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        let file = GgufFile::open(&output).expect("load output gguf");
        let info = file
            .tensor_info("codec.embed.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// F32 + F16 mixed-dtype pass-through with the additive-default
    /// invariant on `bf16_passthrough` and all arch / provenance /
    /// category stamps.
    #[test]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_noncommercial() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 exact-representable half-values: 1.0=0x3C00, -2.0=0xC000,
        // -0.5=0xB800, 3.0=0x4200, 0.15625=0x3100, 42.0=0x5140.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12);

        let input_bytes = safetensors_f32_then_f16(
            "codec.dense.weight",
            &[1, 2],
            &f32_bytes,
            "codec.embed.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input = tmp_path("mixed-in");
        let output = tmp_path("mixed-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_xcodec2_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2, "two tensors observed");
        assert_eq!(
            report.written, 2,
            "both F32 and F16 tensors must pass through"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32/F16-only input must leave the BF16 counter at the Default 0 (additive-default invariant)"
        );

        let file = GgufFile::open(&output).expect("load output gguf");

        let f32_info = file
            .tensor_info("codec.dense.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("codec.embed.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

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
        // The default license path must stamp cc-by-nc-4.0 / NonCommercial
        // (the whole point of the flip vs. neucodec/step_audio2_mini which
        // default to apache-2.0 / Permissive).
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

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// A caller-supplied `license` (e.g. re-published under a different
    /// SPDX at the source) overrides the built-in cc-by-nc-4.0
    /// NonCommercial stamp. Same override pattern as
    /// `convert_file_licensed` — the model_id / source strings survive
    /// but the license / weight_license / neutral source parenthetical
    /// change.
    #[test]
    fn caller_license_override_swaps_the_stamp() {
        // Non-zero payloads that are NOT approximations of π/e —
        // clippy::approx_constant would flag 3.14/2.71 as a naked
        // approximation of the standard constants.
        let f32_vals: [f32; 2] = [11.5, -6.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("codec.a.weight", "F32", &[1, 2], &f32_bytes);
        let input = tmp_path("override-in");
        let output = tmp_path("override-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        // Override to Apache-2.0 (Permissive) — e.g. the caller retrained
        // on a permissive corpus.
        let report = convert_xcodec2_file(&input, &output, Some("apache-2.0")).expect("convert");
        assert_eq!(report.written, 1);

        let file = GgufFile::open(&output).expect("load output gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "override SPDX must land in vokra.provenance.license"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "override class must be re-derived from the SPDX id"
        );
        // Model id / arch / category / upstream_hf remain the built-in
        // values — the override changes only the license triple.
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

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// An empty `Some("")` license override must NOT wipe the built-in
    /// stamp — that would be a silent research-flag downgrade. The
    /// `filter(|s| !s.is_empty())` guard in `convert_xcodec2_file` keeps
    /// the default cc-by-nc-4.0 NonCommercial stamp.
    #[test]
    fn empty_string_license_override_keeps_the_default_stamp() {
        let f32_vals: [f32; 2] = [0.5, -0.5];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("codec.b.weight", "F32", &[1, 2], &f32_bytes);
        let input = tmp_path("empty-in");
        let output = tmp_path("empty-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let _ = convert_xcodec2_file(&input, &output, Some("")).expect("convert");

        let file = GgufFile::open(&output).expect("load output gguf");
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

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }
}
