//! **NVIDIA Canary-1B-Flash**: safetensors checkpoint → GGUF conversion
//! (coverage-audit wave-b, 2026-08-03).
//!
//! Input: the upstream Canary-1B-Flash release from
//! `huggingface.co/nvidia/canary-1b-flash` — an 883M-parameter
//! multilingual multi-task ASR / AST checkpoint across four European
//! languages (English / German / French / Spanish). Output: a GGUF
//! carrying every float tensor plus the `vokra.provenance.*` /
//! `vokra.model.*` / `vokra.schema.*` metadata chunks a future native
//! `vokra-models::canary_1b_flash::*` implementation will read.
//!
//! # Model class
//!
//! Canary-1B-Flash is the **flash-tuned** distillation of the Canary
//! family (FastConformer encoder + Transformer decoder AED). Family
//! shape summary (per the upstream model card at
//! `huggingface.co/nvidia/canary-1b-flash`, fetched 2026-08-03 —
//! CLAUDE.md「ハルシネーション厳禁」):
//!
//! - Encoder: FastConformer, **32 layers** — same encoder depth as
//!   Canary-1B-v2 (`models::canary`).
//! - Decoder: Transformer AED, **4 layers** — the Flash-specific
//!   shrinkage (Canary-1B-v2: 8 layers, Canary-1B-v1: 24 layers). This
//!   is the axis that unlocks the "1000+ RTFx" inference throughput
//!   claim on the model card.
//! - Sample rate: 16 kHz, mono `.wav` / `.flac`.
//! - Task tokens: 4-language subset of the unified Canary SentencePiece
//!   (`<source_lang>` / `<target_lang>` / `<taskname>` / `<pnc>` /
//!   `<itn>` / `<timestamp>` / `<diarize>` / `<emotion>`).
//!
//! Distinct arch tag from [`models::canary`] (Canary-1B-v2) because the
//! decoder-layer axis is genuinely different (8 → 4); silently sharing
//! `"canary"` would misroute the runtime dispatch (the future
//! `Canary1bFlashWeights::from_gguf` walks a different tensor manifest
//! than `CanaryWeights::from_gguf` — a shorter decoder stack).
//!
//! # License
//!
//! Both weights and code ship **CC-BY 4.0** end-to-end
//! (`huggingface.co/nvidia/canary-1b-flash` model card `license:
//! cc-by-4.0`, primary-source-verified 2026-08-03 — CLAUDE.md
//! 「ハルシネーション厳禁」). CC-BY 4.0 is an `AttributionRequired`
//! license class — the FR-MD-09 attribution surface activates and a
//! downstream must show the NVIDIA attribution alongside the model
//! output (mirror of the `models::canary` / `models::parakeet` /
//! `models::parakeet_ctc` / `models::kyutai_stt` posture).
//!
//! # BF16 pass-through
//!
//! Every F32 / F16 / BF16 tensor passes through **verbatim** as the
//! matching GGUF type (BF16 emits type 30 = `GgmlType::BF16`, no
//! convert-time widening — the runtime widens BF16 → f32 losslessly at
//! load via the single choke point `crates/vokra-core/src/gguf/quant/
//! mod.rs decode_bf16`). Mirror of `qwen3_tts` / `vibevoice` /
//! `voxcpm2` / `wespeaker` / `emotion2vec` / `frcrn` — the landed
//! sibling posture that keeps the CI cache footprint at the smallest
//! tensor payload while preserving the exact upstream bit pattern.
//! The `.nemo` tarball for Canary-1B-Flash is typically BF16, so the
//! BF16 arm is the primary consumer path (mirror of the `models::canary`
//! BF16 pass-through added 2026-07-25).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / WeSpeaker / emotion2vec / frcrn contract). Real-weight
//! parity binding is a follow-up wave gated on the `.nemo` extraction
//! plus the license §3.1 sign-off (`docs/license-audit.md`); this
//! converter passes every float tensor through unchanged so a future
//! `Canary1bFlashWeights::from_gguf` can walk the same names.
//!
//! # Prep script bridge
//!
//! The Canary-1B-Flash release ships as a `.nemo` tarball (NeMo
//! Toolkit format — tar-of-yaml+ckpt). A downstream user runs the
//! generic `tools/parity/nemo_pt_to_safetensors.py` (uv-managed,
//! Python 3.12) to flatten it to safetensors before invoking this
//! converter — the same script `models::canary` / `models::parakeet` /
//! `models::parakeet_ctc` consume, no new prep-script fork required
//! (the ticket's "既 `nemo_pt_to_safetensors.py` 流用" prescription).
//! The runtime never sees Python / torch / `.nemo` (FR-LD-05).
//!
//! # No ONNX (permanent)
//!
//! Canary-1B-Flash is distributed as a `.nemo` tarball / Python
//! pipeline; this converter **never** touches ONNX (FR-LD-05); the
//! ASR / AST pipeline is re-implemented natively in a future
//! `crates/vokra-models/src/canary_1b_flash/` module (whisper.cpp 型
//! self re-implementation, CLAUDE.md 設計判断 4).
//!
//! # Wiring status
//!
//! Fully wired: `ModelKind::Canary1bFlash` + `from_arg("canary-1b-flash"
//! | …)` + `as_arg() == "canary-1b-flash"` + `convert_file` dispatch
//! arm all land with this module (mirror of the frcrn / ecapa_tdnn /
//! wespeaker / speaker_3d / emotion2vec wiring pattern).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Canary-1B-Flash GGUFs. Distinct from
/// [`models::canary`]'s `"canary"` arch tag because the decoder-layer
/// axis differs (8 → 4); silently sharing would misroute the runtime
/// dispatch (a Canary-1B-v2 loader would try to bind 4 decoder-layer
/// weights against an 8-layer tensor manifest).
pub const ARCH: &str = "canary-1b-flash";

/// `vokra.model.name` value written for the canonical Canary-1B-Flash
/// GGUF.
pub const NAME: &str = "canary-1b-flash";

/// `vokra.model.category` value — the model is a multilingual ASR / AST
/// checkpoint (the `"asr"` tier — same category as `models::canary` /
/// `models::parakeet` / `models::parakeet_ctc` / `models::kyutai_stt`).
pub const CATEGORY: &str = "asr";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) until a sibling `category`
/// consumer lands in `vokra-core` — mirror of the wespeaker /
/// speaker_3d / emotion2vec / frcrn local constant.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Upstream HuggingFace repository slug (`org/name`) recorded under
/// `vokra.provenance.upstream_hf` so a downstream consumer can trace
/// the artifact back to its serving location.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub const UPSTREAM_HF: &str = "nvidia/canary-1b-flash";

/// Canonical weight license SPDX (`cc-by-4.0`). Overrides via the
/// [`convert_canary_1b_flash_file`] `license` parameter — the standing
/// mechanism for "implementation is clean-room MIT but the upstream
/// distributed checkpoint is another license" scenarios (mirror of
/// `convert_file_licensed` in `lib.rs` and the `license` arg on
/// `convert_frcrn_file` / `convert_wespeaker_file`).
pub const DEFAULT_LICENSE: &str = "cc-by-4.0";

/// FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with `NOTICE` and
/// the `docs/license-audit.md` NVIDIA Canary row (final legal
/// sufficiency = §3.1 owner sign-off; this converter records the
/// attribution but the owner-facing publish gate can add / edit before
/// release). Same posture as [`models::canary::CANARY_ATTRIBUTION_TEXT`].
pub const CANARY_1B_FLASH_ATTRIBUTION_TEXT: &str = "This application uses NVIDIA Canary-1B-Flash \
     (multilingual multi-task ASR / AST — English / German / French / \
     Spanish; FastConformer encoder + Transformer AED decoder — the \
     flash-tuned variant of Canary-1B-v2 with a shrunk 4-layer decoder \
     for 1000+ RTFx inference). Model weights are licensed under CC-BY \
     4.0 (attribution required; commercial use permitted). Copyright \
     (c) NVIDIA. Source: https://huggingface.co/nvidia/canary-1b-flash";

/// Outcome of a Canary-1B-Flash conversion.
///
/// All counters are additive and default to zero — a zero-tensor
/// checkpoint returns `Canary1bFlashReport::default()` and the caller
/// remains responsible for surfacing the "no float tensors" loud note
/// (mirror of the qwen3_tts / vibevoice / voxcpm2 / wespeaker /
/// emotion2vec / frcrn `Report` pattern). `read == written +
/// skipped_non_float` is an invariant preserved by
/// [`convert_canary_1b_flash_file`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Canary1bFlashReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all go through
    /// the same byte-copy path since the BF16 pass-through landed
    /// 2026-07-25).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm signals a reader change upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes a
/// Canary-1B-Flash GGUF to `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.provenance.*` + `vokra.model.*` chunk groups pin
/// the upstream slug, weight license, and model category so the zoo
/// manifest + model-card generator can gate on the artifact alone (no
/// side-car lookup); the FR-MD-09 attribution text lands under
/// `vokra.provenance.attribution`. `vokra.schema.*` is written
/// unconditionally by the GGUF writer.
///
/// `license` overrides `DEFAULT_LICENSE` (`"cc-by-4.0"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when the
/// implementation is clean-room but the redistributed checkpoint
/// carries a different SPDX.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_canary_1b_flash_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Canary1bFlashReport, ConvertError> {
    // Whole-file read: Canary-1B-Flash is ~1.8 GB safetensors (BF16),
    // well within the whole-file range on any development host — no
    // need for the streaming path the Moshi 15 GB / Voxtral 8.7 GB
    // converters run. Any future >8 GB Canary sibling would swap this
    // call for `SafetensorsFileReader::open` + `GgufStreamWriter::begin`
    // per the moshi.rs / qwen3_tts.rs ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough).
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = cc-by-4.0 (upstream `nvidia/canary-1b-flash`
    // model card, primary-source-verified 2026-08-03). `license`
    // overrides for callers who obtained the weight under a different
    // SPDX (see `convert_file_licensed` in `lib.rs`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (
            DEFAULT_LICENSE.to_owned(),
            LicenseClass::AttributionRequired,
        ),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "nvidia/canary-1b-flash (FastConformer encoder + Transformer \
             AED decoder, flash-tuned 4-layer decoder for 1000+ RTFx, \
             cc-by-4.0 attribution required)",
        ),
    );
    // FR-MD-09 attribution: only stamp on the built-in stamp path (a
    // license override to a Permissive SPDX would make the attribution
    // text misleading — the same guard `stamp_attribution` itself
    // provides via its empty-text short-circuit, but keeping the
    // stamp call itself gated on the built-in path makes the intent
    // explicit at the call site).
    if license.is_none() {
        vokra_core::stamp_attribution(&mut b, CANARY_1B_FLASH_ATTRIBUTION_TEXT);
    }

    let mut report = Canary1bFlashReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `qwen3_tts::convert` / `vibevoice::convert` / `voxcpm2::convert` /
    // `wespeaker::convert` / `emotion2vec::convert` / `frcrn::convert`.
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
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Per-process, per-test scratch path in the system temp dir (moshi
    /// / emotion2vec / wespeaker / frcrn test pattern — no external
    /// `tempfile` dep, preserving zero-dep NFR-DS-02). The nanosecond
    /// suffix separates parallel `cargo test` runs so they cannot
    /// clobber each other's files.
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-canary-1b-flash-{}-{}-{}.bin",
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
    /// patterns so a byte-identity assert catches any silent widen /
    /// downcast attempt — the raw zeroed payload would round-trip
    /// trivially through F32 / F16 widen and defeat the pin (mirror of
    /// emotion2vec / frcrn fixture).
    fn synthetic_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        let header = r#"{"encoder.blocks.0.attn.qkv_proj.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&bf16);
        (buf, bf16)
    }

    /// Builds a synthetic safetensors buffer with one F32 tensor
    /// (`shape=[2,3]`, 24 B) followed by one F16 tensor
    /// (`shape=[1,4]`, 8 B). The offsets are chosen so the tensors are
    /// contiguous in the data region — mirror of emotion2vec / frcrn
    /// fixture.
    fn synthetic_f32_and_f16_safetensors() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let f32_vals: [f32; 6] = [1.0, -2.0, 3.5, -0.25, 100.0, 0.001];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 24, "6 elements × 4 bytes F32 payload");
        let f16_patterns: [u16; 4] = [0x3C00, 0xC000, 0x4200, 0x0001];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 8, "4 elements × 2 bytes F16 payload");
        let header = r#"{"encoder.blocks.0.attn.qkv_proj.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"decoder.blocks.0.self_attn.qkv.weight":{"dtype":"F16","shape":[1,4],"data_offsets":[24,32]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&f32_bytes);
        buf.extend_from_slice(&f16_bytes);
        (buf, f32_bytes, f16_bytes)
    }

    /// BF16 pass-through: the upstream BF16 checkpoint (the primary
    /// upstream posture — the `.nemo` tarball is BF16) must survive the
    /// file-based converter round-trip with its dtype preserved (GGUF
    /// type 30 = `GgmlType::BF16`) and its payload byte-identical to
    /// the input. Mirror of the emotion2vec / wespeaker / frcrn
    /// equivalent.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input_bytes, bf16_payload) = synthetic_bf16_safetensors();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_canary_1b_flash_file(&input, &output, None).expect("convert");

        // Counters: single BF16 tensor read + written + BF16 subset.
        assert_eq!(report.read, 1, "one tensor visible in safetensors header");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of frcrn / emotion2vec)"
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
            .tensor_info("encoder.blocks.0.attn.qkv_proj.weight")
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
            Some(LicenseClass::AttributionRequired.as_str()),
            "CC-BY 4.0 must resolve to AttributionRequired (same as canary-1b-v2)"
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category chunk pins the `asr` tier (same as canary / parakeet)"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF),
            "upstream slug pins traceability back to nvidia/canary-1b-flash"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
            "vokra.schema.version must be stamped"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
            "vokra.schema.producer must be stamped"
        );
        // Attribution text is non-empty and NVIDIA-named (FR-MD-09).
        let attr = file
            .get(chunks::KEY_PROVENANCE_ATTRIBUTION)
            .and_then(|v| v.as_str())
            .expect("attribution present on the default license path");
        assert!(
            attr.contains("NVIDIA") && attr.contains("CC-BY 4.0"),
            "attribution must name NVIDIA + CC-BY 4.0: {attr}"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// F32 + F16 pass-through: two float tensors of distinct dtypes in
    /// the same input must both reach the pass-through arm without
    /// collapsing into a single dtype branch, and the BF16 counter must
    /// remain 0. Guards against a naive `if bf16 { … } else` refactor.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let (input_bytes, f32_payload, f16_payload) = synthetic_f32_and_f16_safetensors();
        let input = scratch_path("f32f16-in");
        let output = scratch_path("f32f16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_canary_1b_flash_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 2, "two tensors visible in header");
        assert_eq!(report.written, 2, "both F32 and F16 must pass through");
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32+F16-only input must leave the BF16 subset counter at Default 0"
        );

        // Both tensors survive the round-trip with their upstream names
        // and dtypes preserved.
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let f32_info = file
            .tensor_info("encoder.blocks.0.attn.qkv_proj.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(f32_info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(f32_info), f32_payload.as_slice());

        let f16_info = file
            .tensor_info("decoder.blocks.0.self_attn.qkv.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(f16_info.dimensions, vec![1, 4]);
        assert_eq!(file.tensor_bytes(f16_info), f16_payload.as_slice());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// License override: the caller-supplied SPDX must replace the
    /// default `cc-by-4.0` stamp on the artifact, and the attribution
    /// text must NOT land (a Permissive override would make the CC-BY
    /// attribution wording misleading — mirror of the wespeaker /
    /// frcrn license-override test posture with the added attribution
    /// guard).
    #[test]
    fn license_override_replaces_default_stamp_and_suppresses_attribution() {
        let (input_bytes, _) = synthetic_bf16_safetensors();
        let input = scratch_path("license-in");
        let output = scratch_path("license-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        // Override with `apache-2.0` (Permissive) — the SPDX must land
        // in the license stamp, the class must re-derive to Permissive,
        // and the FR-MD-09 attribution text must NOT be stamped (a
        // CC-BY attribution on a Permissive artifact would misinform
        // the downstream).
        convert_canary_1b_flash_file(&input, &output, Some("apache-2.0"))
            .expect("convert with license override");

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
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
            "apache-2.0 must resolve to Permissive"
        );
        assert!(
            file.get(chunks::KEY_PROVENANCE_ATTRIBUTION).is_none(),
            "override to a Permissive SPDX must suppress the CC-BY \
             attribution text (otherwise a Permissive artifact would \
             carry misleading CC-BY wording)"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
