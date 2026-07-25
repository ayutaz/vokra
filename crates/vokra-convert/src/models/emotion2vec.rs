//! **emotion2vec+ Large**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 5 emotion tier, 2026-07-25).
//!
//! Input: the upstream `emotion2vec/emotion2vec_plus_large` release —
//! `model.safetensors`. Output: a GGUF carrying every float tensor plus
//! the `vokra.provenance.*` / `vokra.model.*` / `vokra.schema.*` metadata
//! chunks a future native `vokra-models::emotion2vec::*` implementation
//! will read.
//!
//! # Model class
//!
//! emotion2vec+ is a 9-class emotion **SSL pretrain** (ACL 2024,
//! `arXiv:2312.15185`): a self-supervised speech representation model
//! whose downstream head is a 9-way emotion classifier (Angry /
//! Disgusted / Fearful / Happy / Neutral / Other / Sad / Surprised /
//! `<unk>`). This is Vokra's first `category = "emotion"` model — an
//! audio-input SSL checkpoint the runtime will consume through the
//! shared feature-extraction ops (STFT / mel filterbank) into a
//! wav2vec 2.0-style encoder + emotion classification head.
//!
//! # License
//!
//! Both code and weights ship **MIT** end-to-end
//! (huggingface.co/emotion2vec/emotion2vec_plus_large model card
//! `license: mit`, fetched 2026-07-25 — CLAUDE.md「ハルシネーション厳禁」).
//! MIT is a `Permissive` license class — same commercial verdict as
//! apache-2.0 (no runtime-side attribution obligation).
//!
//! # BF16 posture
//!
//! Every F32 / F16 / BF16 tensor passes through **verbatim** as the
//! matching GGUF type (BF16 emits type 30 = `GgmlType::BF16`, no
//! convert-time widening — the runtime widens BF16 → f32 losslessly at
//! load via the single choke point `crates/vokra-core/src/gguf/quant/
//! mod.rs decode_bf16`). Mirror of `qwen3_tts` / `vibevoice` /
//! `voxcpm2` / `moshi` / `voxtral` — the landed sibling posture that
//! keeps the CI cache footprint at the smallest tensor payload while
//! preserving the exact upstream bit pattern.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice contract). Real-weight parity binding is a follow-up wave
//! gated on the upstream tensor-name manifest fetch + license §3.1
//! sign-off (`docs/license-audit.md`); this converter passes every
//! float tensor through unchanged so a future
//! `Emotion2vecWeights::from_gguf` can walk the same names.
//!
//! # No ONNX (permanent)
//!
//! emotion2vec+ is distributed as safetensors + a Python / FunASR
//! pipeline; this converter **never** touches ONNX (FR-LD-05); the
//! pipeline will be re-implemented natively when a
//! `crates/vokra-models/src/emotion2vec/` lands (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for emotion2vec+ GGUFs. Distinct from every
/// sibling arch tag because emotion2vec is the first
/// `category = "emotion"` SSL pretrain — silently sharing an arch tag
/// would misroute the runtime dispatch (an ASR / TTS backbone would try
/// to interpret the 9-class classifier head).
pub const ARCH: &str = "emotion2vec";

/// `vokra.model.name` value written for the canonical emotion2vec+ Large
/// GGUF.
pub const NAME: &str = "emotion2vec-plus-large";

/// `vokra.model.category` value — the first `"emotion"` in the
/// converter tree. Consumed by the model-card generator + zoo manifest
/// tier gate so an emotion classifier is not accidentally advertised as
/// an ASR / TTS release.
pub const CATEGORY: &str = "emotion";

/// `vokra.provenance.upstream_hf` value — the HuggingFace path the
/// release ships from. Recorded so a downstream consumer can re-fetch /
/// re-verify without a separate manifest lookup.
pub const UPSTREAM_HF: &str = "emotion2vec/emotion2vec_plus_large";

/// Canonical weight license SPDX (`mit`). Overrides via the
/// [`convert_emotion2vec_file`] `license` parameter — the standing
/// mechanism for "implementation is clean-room MIT but the upstream
/// distributed checkpoint is another license" scenarios (mirror of
/// `convert_file_licensed` in `lib.rs`).
pub const DEFAULT_LICENSE: &str = "mit";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) until a sibling `category`
/// consumer lands in `vokra-core`.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Outcome of an emotion2vec+ conversion.
///
/// All counters are additive and default to zero — a zero-tensor
/// checkpoint returns `Emotion2vecReport::default()` and the caller
/// remains responsible for surfacing the "no float tensors" loud note
/// (mirror of the qwen3_tts / vibevoice / voxcpm2 `Report` pattern with
/// an added `read` counter that pins the total tensor budget the
/// safetensors reader surfaced).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Emotion2vecReport {
    /// Total tensors seen in the upstream safetensors header (the sum
    /// of `written + skipped_non_float`). Additive over
    /// `qwen3_tts::Qwen3TtsReport` — pins the budget so a truncated
    /// header cannot silently drop tensors without the caller noticing.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all go through
    /// the same byte-copy path since the BF16 pass-through landed
    /// 2026-07-25).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm is a quantized dtype the runtime is not
    /// expected to consume).
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes an emotion2vec+
/// GGUF to `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.provenance.*` + `vokra.model.*` chunk groups pin the
/// upstream HF path, weight license, and model category so the zoo
/// manifest + model-card generator can gate on the artifact alone (no
/// side-car lookup). `vokra.schema.*` is written unconditionally by the
/// GGUF writer.
///
/// `license` overrides [`DEFAULT_LICENSE`] (`"mit"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when the implementation
/// is clean-room but the redistributed checkpoint carries a different
/// SPDX (e.g. `cc-by-4.0`).
pub fn convert_emotion2vec_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Emotion2vecReport, ConvertError> {
    // Whole-file read: emotion2vec+ Large ships as a single small
    // `model.safetensors` (order of MB, not GB) — no need for the
    // streaming path the Moshi 15 GB / Voxtral 8.7 GB converters run.
    // Any future 7B-scale emotion sibling would swap this call for
    // `SafetensorsFileReader::open` + `GgufStreamWriter::begin` per the
    // moshi.rs / qwen3_tts.rs ADR (docs/adr/qwen3-tts-bf16.md, strategy
    // A_passthrough).
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Self-describing redistribution: the artifact carries its own
    // licence. emotion2vec+ Large ships MIT end-to-end
    // (huggingface.co/emotion2vec/emotion2vec_plus_large model card
    // `license: mit`, fetched 2026-07-25 — CLAUDE.md「ハルシネーション厳禁」).
    // The `license` override lets a downstream repackager stamp a
    // different SPDX if they redistribute under stricter terms (the
    // same knob `convert_file_licensed` exposes in `lib.rs`).
    let effective_license = license.unwrap_or(DEFAULT_LICENSE);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        effective_license,
        Some(NAME),
        Some(UPSTREAM_HF),
    );

    let mut report = Emotion2vecReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `qwen3_tts::convert` / `vibevoice::convert` / `voxcpm2::convert`.
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

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Parse(e.to_string()))?;
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Per-process, per-test scratch path in the system temp dir
    /// (moshi test pattern — no external `tempfile` dep, preserving
    /// zero-dep NFR-DS-02). The nanosecond suffix separates the two
    /// tests in this module so a parallel `cargo test` run cannot
    /// clobber files across them.
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-emotion2vec-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    /// Builds a synthetic safetensors buffer with a single BF16 tensor.
    ///
    /// The payload is chosen from a known set of non-zero BF16 bit
    /// patterns (`1.0`, `-2.5`, `0.15625`, `3.5`, `-0.5`, `42.0`) so a
    /// byte-identity assert catches any silent widen / downcast attempt
    /// — the raw zeroed payload would round-trip trivially through
    /// F32 / F16 widen and defeat the pin.
    fn synthetic_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        let header = r#"{"encoder.embed_tokens.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&bf16);
        (buf, bf16)
    }

    /// Builds a synthetic safetensors buffer with one F32 tensor
    /// (`shape=[2,3]`, 24 B) followed by one F16 tensor
    /// (`shape=[1,4]`, 8 B). The offsets are chosen so the tensors are
    /// contiguous in the data region.
    fn synthetic_f32_and_f16_safetensors() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        // F32 payload: 6 non-zero floats so a silent widen would flip a
        // fence rather than trivially round-trip a zero buffer.
        let f32_vals: [f32; 6] = [1.0, -2.0, 3.5, -0.25, 100.0, 0.001];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 24, "6 elements × 4 bytes F32 payload");
        // F16 payload: 4 half-floats with known non-zero bit patterns.
        let f16_patterns: [u16; 4] = [0x3C00, 0xC000, 0x4200, 0x0001];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 8, "4 elements × 2 bytes F16 payload");
        // Header declares F32 first, then F16 in the data region.
        let header = r#"{"encoder.layer0.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"encoder.layer1.weight":{"dtype":"F16","shape":[1,4],"data_offsets":[24,32]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&f32_bytes);
        buf.extend_from_slice(&f16_bytes);
        (buf, f32_bytes, f16_bytes)
    }

    /// STEP 1 RED (BF16 pass-through): the upstream BF16 checkpoint
    /// must survive the file-based converter round-trip with its dtype
    /// preserved (GGUF type 30 = `GgmlType::BF16`) and its payload
    /// byte-identical to the input. Mirrors qwen3_tts / vibevoice /
    /// voxcpm2 / moshi / voxtral. Fails today with `unimplemented!()`.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input_bytes, bf16_payload) = synthetic_bf16_safetensors();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_emotion2vec_file(&input, &output, None).expect("convert");

        // Counters: single BF16 tensor read + written + BF16 subset.
        assert_eq!(report.read, 1, "one tensor visible in safetensors header");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of qwen3_tts)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );

        // Round-trip: dtype preserved, payload byte-identical (no silent widen).
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let info = file
            .tensor_info("encoder.embed_tokens.weight")
            .expect("BF16 tensor present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16_payload.as_slice(),
            "BF16 payload must be byte-identical to input"
        );

        // Provenance + category chunks pinned on the artifact itself.
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
            Some(DEFAULT_LICENSE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category chunk pins the first `emotion` model in the converter tree"
        );
        // Schema stamp is written unconditionally by the GGUF writer.
        assert!(
            file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
            "vokra.schema.version must be stamped"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
            "vokra.schema.producer must be stamped"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// STEP 1 RED (F32 + F16 pass-through): two float tensors of
    /// distinct dtypes in the same input must both reach the
    /// pass-through arm without collapsing into a single dtype branch,
    /// and the BF16 counter must remain 0 (default). Guards against a
    /// naive `if bf16 { ... } else` refactor.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let (input_bytes, f32_payload, f16_payload) = synthetic_f32_and_f16_safetensors();
        let input = scratch_path("f32f16-in");
        let output = scratch_path("f32f16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_emotion2vec_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 2, "two tensors visible in header");
        assert_eq!(report.written, 2, "both F32 and F16 must pass through");
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32+F16-only input must leave the BF16 subset counter at the Default 0"
        );

        // Both tensors survive the round-trip with their upstream names
        // and dtypes preserved.
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let f32_info = file
            .tensor_info("encoder.layer0.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(f32_info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(f32_info), f32_payload.as_slice());

        let f16_info = file
            .tensor_info("encoder.layer1.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(f16_info.dimensions, vec![1, 4]);
        assert_eq!(file.tensor_bytes(f16_info), f16_payload.as_slice());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
