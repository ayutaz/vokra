//! **FireRedTeam/FireRedASR-LLM-L** — Chinese-LLM ASR (Conformer + audio-to-text
//! adapter + Qwen2 LM decoder) safetensors → GGUF conversion
//! (coverage-audit-2026-08-03 Wave B fast-track, post-audit CC-gap 2026-08-13).
//!
//! Input: the upstream `FireRedTeam/FireRedASR-LLM-L` release —
//! `huggingface.co/FireRedTeam/FireRedASR-LLM-L`, an Apache-2.0 Chinese-LLM
//! ASR model (~16.6 GB BF16, 8.3B params = Conformer encoder + linear/MLP
//! audio-to-text adapter + Qwen2 ~7B LM decoder). AISHELL-1 SoTA per the
//! upstream release notes. Output: a GGUF carrying every float tensor plus
//! the `vokra.provenance.*`, `vokra.model.*` and `vokra.schema.*` metadata
//! chunks a future native `vokra-models::firered_asr_llm_l::*`
//! implementation will read.
//!
//! # Model class
//!
//! FireRedASR-LLM-L is a **Conformer + Qwen2 LLM decoder** topology —
//! same "encoder + audio-text adapter + LLM decoder" mold the sibling
//! Canary-Qwen-2.5B (`crates/vokra-convert/src/models/canary_qwen.rs`)
//! and Voxtral use. Category is `"asr"` — same tier as the sibling
//! `firered_asr_aed_l` (Whisper-topology AED) plus Canary-Qwen /
//! Voxtral / Whisper family. The two FireRedASR variants ship distinct
//! arch tags because their topologies differ substantively:
//!
//! - `firered_asr_aed_l` = Whisper-topology AED (mel + Transformer
//!   encoder + Transformer decoder). ~2.2 GB. Sibling module.
//! - `firered_asr_llm_l` = Conformer encoder + linear/MLP audio-text
//!   adapter + Qwen2 LM decoder (~1 GB encoder + ~7 GB LM). ~16.6 GB.
//!
//! Silently sharing an arch tag would mis-route runtime dispatch (an
//! AED Whisper loader would try to interpret a Qwen2 LM decoder
//! checkpoint, or vice versa) — every Wave B model declares its own
//! `vokra.model.arch` per the wave-B uniform posture recorded in
//! `crates/vokra-convert/src/models/mod.rs` header comment.
//!
//! # License
//!
//! Both code and weights ship **Apache-2.0** end-to-end per the model
//! card at `huggingface.co/FireRedTeam/FireRedASR-LLM-L` (recorded in
//! the coverage-audit-2026-08-03 wave-b ticket
//! `docs/tickets/coverage-audit-2026-08-03/wave-b/firered-asr-llm-l.md`).
//! Apache-2.0 is a `Permissive` license class — no runtime-side
//! attribution obligation (unlike NVIDIA's CC-BY 4.0 Parakeet-CTC /
//! Canary / Canary-Qwen which stamp FR-MD-09 attribution text). The
//! `license` override parameter to [`convert_firered_asr_llm_l_file`]
//! follows the standing "implementation is clean-room MIT but the
//! redistributed checkpoint carries a distinct SPDX" precedent
//! (mirror of `convert_firered_asr_aed_l_file`'s license arg).
//!
//! # Owner primary-source verification pending
//!
//! Audit ticket §Owner critical path lists: (1) HF card
//! `license: apache-2.0` primary-source confirmation, (2) FireRedTeam
//! GitHub `github.com/FireRedTeam/FireRedASR` LICENSE cross-check,
//! (3) Chinese ASR training-corpus commercial-use audit (WenetSpeech /
//! KeSpeech 混成疑義 possible — the two most common Chinese ASR
//! corpora with divergent commercial redistribution clauses),
//! (4) `docs/license-audit.md` §3.1 row landing with ☑ Commercial or
//! ☑ Research-only mark. This converter + its BF16 pass-through are
//! green now; publish is fail-closed until the §3.1 row acquires a ☑
//! from the owner (memory
//! `[[feedback-license-signoff-primary-source]]`).
//!
//! # vast.ai required (~16.6 GB)
//!
//! Per memory `[[feedback-large-models-on-vast-ai]]` (2 GB CC-workflow
//! local-convert owner threshold; the M1 iMac 16 GB machine ran into
//! OS-level swap when mmap-ing 8 GB Voxtral / 48 GB Voxtral-Small-24B),
//! the actual weight fetch + convert + publish runs on a rented
//! vast.ai GPU box via `docs/handoff/vast-ai-large-model-publish.md`
//! — CC lands only the converter code + prepare-checkpoint sidecar +
//! tests here.
//!
//! # BF16 pass-through (mirror of firered_asr_aed_l / higgs_audio_v3_tts_4b
//! # / canary_qwen / magpietts_v2602 / owsm_v4_medium_1b /
//! # parakeet_tdt_1_1b / sortformer_diar_4spk_v1 / speaker_3d /
//! # ecapa_tdnn / qwen3_tts / voxcpm2 / vibevoice / moshi / emotion2vec
//! # / wespeaker / frcrn / nkf_aec)
//!
//! Every F32 / F16 / BF16 tensor passes through **verbatim** as the
//! matching GGUF type (BF16 emits type 30 = `GgmlType::BF16`, no
//! convert-time widening — the runtime widens BF16 → f32 losslessly at
//! load via the single choke point `crates/vokra-core/src/gguf/quant/
//! mod.rs decode_bf16`). Mirror of the landed sibling posture that
//! keeps the CI cache footprint at the smallest tensor payload while
//! preserving the exact upstream bit pattern.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / WeSpeaker / emotion2vec / FRCRN / MagpieTTS-v2602 /
//! FireRedASR-AED-L / Higgs-Audio contract). Real-weight parity
//! binding is a follow-up wave gated on the upstream tensor-name
//! manifest fetch + license §3.1 sign-off (`docs/license-audit.md`);
//! this converter passes every float tensor through unchanged so a
//! future `FireredAsrLlmLWeights::from_gguf` can walk the same names.
//!
//! A shared `vokra_ops::qwen2` op is a PROPOSED consolidation, not a
//! landed module — no such module exists today, and this model has no
//! runtime binder of its own yet. The only landed Qwen2-family forward
//! is the inline one in `vokra-models/src/voxtral/text_decoder.rs`
//! (GQA + RoPE + SwiGLU + RMSNorm, via the public `rms_norm` /
//! `silu_inplace` / `rope_apply` helpers). `canary_qwen` reuses that
//! module; `kyutai_stt` does not (it re-implements nothing yet — its
//! `transcribe` is still `NotImplemented`). Consolidating the three
//! into one op is the follow-up wave this converter is written against.
//!
//! # Prep script bridge — sharded safetensors merge
//!
//! The upstream FireRedASR-LLM-L release ships **sharded safetensors**
//! (`model-00001-of-000NN.safetensors` + `model.safetensors.index.json`,
//! ~16.6 GB total in BF16 for the 8.3B backbone). This Rust converter
//! consumes a **single** safetensors file, so a downstream user runs
//! the sidecar `tools/parity/firered_asr_llm_l/prepare_checkpoint.py`
//! (uv-managed Python 3.12, mirror of `higgs_audio_v3_tts_4b/`) to
//! merge the shards, dedupe tied tensors (data_ptr collision → clone
//! + audit trail), and strip non-float training scaffold before
//! invoking this converter — the same posture higgs_audio_v3_tts_4b /
//! DFN3 / DAC / Kokoro / UTMOS / SBV2 / FRCRN converters use. The
//! runtime never sees Python / torch (FR-LD-05).
//!
//! # No ONNX (permanent)
//!
//! The upstream FireRedASR-LLM-L release ships PyTorch sharded
//! safetensors + a Python inference pipeline; this converter **never**
//! touches ONNX (FR-LD-05). The ASR pipeline is re-implemented natively
//! in a future `crates/vokra-models/src/firered_asr_llm_l/` module
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for FireRedASR-LLM-L GGUFs. Intentionally
/// distinct from every sibling ASR arch tag (`whisper` /
/// `distil-whisper` / `kotoba-whisper` / `canary` / `canary-qwen` /
/// `parakeet` / `parakeet_ctc` / `omniasr_ctc` / `kyutai_stt` /
/// `voxtral` / `firered_asr_aed_l`) — silently aliasing any of these
/// would mis-route the runtime dispatch (FireRedTeam's LLM release has
/// its own tensor manifest / tokenizer / hparam contract; a future
/// `FireredAsrLlmLWeights::from_gguf` will diverge from every sibling
/// loader). Specifically distinct from the FireRedTeam AED sibling
/// (`firered_asr_aed_l`, Whisper-topology) because the LLM release
/// swaps the AED decoder for a Qwen2 LLM decoder + linear/MLP
/// audio-text adapter, which is a completely different tensor layout.
pub const ARCH: &str = "firered_asr_llm_l";

/// `vokra.model.name` value written for the canonical
/// `FireRedTeam/FireRedASR-LLM-L` release.
pub const NAME: &str = "firered-asr-llm-l";

/// `vokra.model.category` value — `"asr"`, same tier as the sibling
/// `firered_asr_aed_l` (Whisper-topology AED) plus Whisper / Canary /
/// Canary-Qwen / Voxtral / Kotoba-Whisper / distil-Whisper family.
/// Consumed by the model-card generator + zoo manifest tier gate.
pub const CATEGORY: &str = "asr";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) until a sibling `category`
/// consumer lands in `vokra-core` — mirror of the wespeaker /
/// speaker_3d / emotion2vec / neucodec / frcrn / firered_asr_aed_l /
/// higgs_audio_v3_tts_4b local constant.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Upstream repository slug (`org/name`) recorded under
/// `vokra.provenance.upstream_hf` so a downstream consumer can trace
/// the artifact back to its serving location.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Canonical upstream HF slug for the FireRedASR-LLM-L release.
pub const UPSTREAM_HF: &str = "FireRedTeam/FireRedASR-LLM-L";

/// Canonical weight license SPDX (`apache-2.0`). Overrides via the
/// [`convert_firered_asr_llm_l_file`] `license` parameter — the
/// standing mechanism for "implementation is clean-room MIT but the
/// upstream distributed checkpoint is another license" scenarios
/// (mirror of `convert_file_licensed` in `lib.rs` and the `license`
/// arg on `convert_firered_asr_aed_l_file` /
/// `convert_higgs_audio_v3_tts_4b_file` / `convert_wespeaker_file`).
pub const DEFAULT_LICENSE: &str = "apache-2.0";

/// Outcome of a FireRedASR-LLM-L conversion.
///
/// All counters are additive and default to zero — a zero-tensor
/// checkpoint returns `FireredAsrLlmLReport::default()` and the caller
/// remains responsible for surfacing the "no float tensors" loud note
/// (mirror of the firered_asr_aed_l / higgs_audio_v3_tts_4b /
/// qwen3_tts / vibevoice / voxcpm2 / wespeaker / emotion2vec /
/// neucodec / frcrn `Report` pattern). `read == written +
/// skipped_non_float` is an invariant preserved by
/// [`convert_firered_asr_llm_l_file`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FireredAsrLlmLReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all ride the
    /// same byte-copy pass-through arm).
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

/// Reads a safetensors checkpoint at `input` (as emitted by
/// `tools/parity/firered_asr_llm_l/prepare_checkpoint.py`) and writes a
/// FireRedASR-LLM-L GGUF to `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.provenance.*` + `vokra.model.*` chunk groups pin
/// the upstream slug, weight license, and model category so the zoo
/// manifest + model-card generator can gate on the artifact alone (no
/// side-car lookup). `vokra.schema.*` is written unconditionally by
/// the GGUF writer.
///
/// `license` overrides `DEFAULT_LICENSE` (`"apache-2.0"`) — the
/// same mechanism `lib.rs::convert_file_licensed` uses when the
/// implementation is clean-room but the redistributed checkpoint
/// carries a different SPDX (mirror of the sibling
/// `convert_firered_asr_aed_l_file` / `convert_higgs_audio_v3_tts_4b_file`
/// override convention).
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
///
/// # Memory footprint
///
/// The upstream release is ~16.6 GB (8.3B parameters × 2 bytes BF16).
/// Per memory `[[feedback-large-models-on-vast-ai]]` this exceeds the
/// 2 GB CC-workflow local-convert threshold, so the actual convert
/// runs on vast.ai. The current `std::fs::read` load buffers the
/// entire file into memory — the vast.ai runbook
/// (`docs/handoff/vast-ai-large-model-publish.md`) rents a box with
/// ≥ 32 GB RAM so the whole-file load + `GgufBuilder::to_bytes` peak
/// fits. A future streaming pass-through (per the Moshi 15 GB /
/// Voxtral 8.7 GB `SafetensorsFileReader` + `GgufStreamWriter`
/// posture) would shrink the peak footprint to one tensor payload;
/// that upgrade is a follow-up if the vast.ai box's RAM budget
/// becomes a constraint on a smaller instance class.
pub fn convert_firered_asr_llm_l_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FireredAsrLlmLReport, ConvertError> {
    // Whole-file read: FireRedASR-LLM-L is ~16.6 GB per the audit
    // ticket. Above the Moshi 15 GB / Voxtral 8.7 GB streaming
    // threshold in principle, but the wave-b sibling posture uses
    // whole-file read on vast.ai boxes with ≥ 32 GB RAM per
    // `docs/handoff/vast-ai-large-model-publish.md`; if a rented
    // vast.ai instance OOMs on this file, swap this call for
    // `SafetensorsFileReader::open` + `GgufStreamWriter::begin` per
    // the moshi.rs / qwen3_tts.rs ADR (docs/adr/qwen3-tts-bf16.md,
    // strategy A_passthrough).
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = apache-2.0 (upstream
    // `huggingface.co/FireRedTeam/FireRedASR-LLM-L` model-card
    // header). `license` overrides for callers who obtained the weight
    // under a different SPDX (see `convert_file_licensed` in `lib.rs`).
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
            "FireRedTeam/FireRedASR-LLM-L (Conformer + audio-text adapter + \
             Qwen2 LM decoder Chinese ASR, apache-2.0)",
        ),
    );

    let mut report = FireredAsrLlmLReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `firered_asr_aed_l::convert_firered_asr_aed_l_file` /
    // `higgs_audio_v3_tts_4b::convert_higgs_audio_v3_tts_4b_file` /
    // `qwen3_tts::convert` / `vibevoice::convert` / `voxcpm2::convert` /
    // `wespeaker::convert` / `emotion2vec::convert` / `neucodec::convert`.
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
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch path (PID + nanos + a suffix derived
    /// from the caller — every test in this module uses a distinct
    /// `tag` so concurrent runs do not collide). Mirror of the sibling
    /// firered_asr_aed_l / higgs_audio_v3_tts_4b test-fixture posture
    /// (no external `tempfile` dep, preserving zero-dep NFR-DS-02).
    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-firered-asr-llm-l-{tag}-{}-{}.{ext}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        p
    }

    /// RAII cleanup so failing tests do not leak temp files on disk
    /// (best-effort — a panic mid-cleanup is fine).
    struct TempFileGuard(PathBuf);
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Encodes `values` as BF16 (top 16 bits of each `f32`) little-endian.
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Builds a synthetic safetensors buffer with a single BF16 tensor.
    ///
    /// The payload is chosen from a known set of non-zero BF16 bit
    /// patterns so a byte-identity assert catches any silent widen /
    /// downcast attempt — the raw zeroed payload would round-trip
    /// trivially through F32 / F16 widen and defeat the pin (mirror of
    /// firered_asr_aed_l / higgs_audio_v3_tts_4b / emotion2vec / frcrn
    /// / neucodec fixtures).
    fn synthetic_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = bf16_bytes(&values);
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        // A plausible Qwen2 LM-decoder embed-tokens tensor name — the
        // LLM decoder half of the FireRedASR-LLM-L topology.
        let header = r#"{"decoder.model.embed_tokens.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&bf16);
        (buf, bf16)
    }

    /// Pins the arch / name / category / upstream_hf constants so a
    /// silent rename cannot slip past review. Every downstream reader
    /// (compliance gate / model-card generator / zoo manifest) keys off
    /// these exact strings; drifting them mis-routes the runtime
    /// dispatch (FR-EX-08). Mirrors the sibling `firered_asr_aed_l` /
    /// `higgs_audio_v3_tts_4b` posture and asserts distinctness against
    /// every sibling ASR arch tag.
    #[test]
    fn arch_and_name_pin_matches_publish_repo() {
        assert_eq!(ARCH, "firered_asr_llm_l");
        assert_eq!(NAME, "firered-asr-llm-l");
        assert_eq!(CATEGORY, "asr");
        assert_eq!(UPSTREAM_HF, "FireRedTeam/FireRedASR-LLM-L");
        assert_eq!(DEFAULT_LICENSE, "apache-2.0");
        // Sibling ASR arch tags — silently sharing would mis-route
        // runtime dispatch. Assert distinctness on the closest siblings
        // (Whisper family + Canary + Canary-Qwen + Voxtral + Kyutai +
        // Parakeet + the FireRedTeam AED sibling).
        for sibling in [
            "whisper",
            "distil-whisper",
            "kotoba-whisper",
            "crisperwhisper",
            "canary",
            "canary-qwen",
            "parakeet",
            "parakeet_ctc",
            "omniasr_ctc",
            "kyutai_stt",
            "voxtral",
            "firered_asr_aed_l",
            "firered_vad",
        ] {
            assert_ne!(ARCH, sibling, "arch tag must not collide with {sibling}");
        }
    }

    /// Pins the BF16 pass-through end-to-end: the tensor survives the
    /// converter's `convert_firered_asr_llm_l_file` file → file
    /// round-trip with its dtype preserved (`GgmlType::BF16`, GGUF
    /// type 30) and its payload byte-identical. Mirrors
    /// `firered_asr_aed_l::tests::bf16_tensor_passes_through_verbatim`
    /// / `higgs_audio_v3_tts_4b::tests::bf16_tensor_passes_through_verbatim`.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input_bytes, bf16_payload) = synthetic_bf16_safetensors();
        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report =
            convert_firered_asr_llm_l_file(&input_path, &output_path, None).expect("convert BF16");

        // Counters: single BF16 tensor read + written + BF16 subset.
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of firered_asr_aed_l / higgs_audio_v3_tts_4b)"
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
        let out_bytes = std::fs::read(&output_path).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let info = file
            .tensor_info("decoder.model.embed_tokens.weight")
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

        // Provenance + category + upstream_hf chunks pinned on the artifact itself.
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
            "category chunk pins the ASR-family membership"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF),
            "upstream slug pins traceability back to FireRedTeam/FireRedASR-LLM-L"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
            "vokra.schema.version must be stamped"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
            "vokra.schema.producer must be stamped"
        );
    }

    /// F32 + F16 pass-through: two float tensors of distinct dtypes in
    /// the same input must both reach the pass-through arm without
    /// collapsing into a single dtype branch, and the BF16 counter
    /// must remain 0. Guards against a naive `if bf16 { … } else`
    /// refactor. Mirror of the sibling firered_asr_aed_l /
    /// higgs_audio_v3_tts_4b equivalent.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        // Two tensors in one safetensors file:
        //   encoder.conformer.blocks.0.mhsa.qkv.weight — F32, [2, 3] → 24 B @ [0..24)
        //   adapter.linear.bias                        — F16, [1, 4] →  8 B @ [24..32)
        // Both dtypes must reach the pass-through arm and neither must
        // increment `bf16_passthrough`.
        let f32_vals: [f32; 6] = [1.0, -2.0, 3.5, -0.25, 100.0, 0.001];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 24, "6 elements × 4 bytes F32 payload");
        let f16_patterns: [u16; 4] = [0x3C00, 0xC000, 0x4200, 0x0001];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 8, "4 elements × 2 bytes F16 payload");
        let header = r#"{"encoder.conformer.blocks.0.mhsa.qkv.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"adapter.linear.bias":{"dtype":"F16","shape":[1,4],"data_offsets":[24,32]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16_bytes);

        let input_path = scratch_path("f32f16-in", "safetensors");
        let output_path = scratch_path("f32f16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_firered_asr_llm_l_file(&input_path, &output_path, None)
            .expect("convert F32 + F16");

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
        let out_bytes = std::fs::read(&output_path).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let f32_info = file
            .tensor_info("encoder.conformer.blocks.0.mhsa.qkv.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(f32_info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("adapter.linear.bias")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(f16_info.dimensions, vec![1, 4]);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());
    }

    /// License override: the caller-supplied SPDX must replace the
    /// default `apache-2.0` stamp on the artifact (mirror of the
    /// firered_asr_aed_l / higgs_audio_v3_tts_4b / wespeaker /
    /// emotion2vec / frcrn / neucodec test — proves the standing
    /// `convert_file_licensed` override reaches this arm).
    #[test]
    fn license_override_replaces_default_stamp() {
        let (input_bytes, _) = synthetic_bf16_safetensors();
        let input_path = scratch_path("license-in", "safetensors");
        let output_path = scratch_path("license-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        // Override with `mit` (a Permissive alternative to apache-2.0)
        // — the SPDX must land in the license stamp and the class
        // must re-derive to Permissive.
        convert_firered_asr_llm_l_file(&input_path, &output_path, Some("mit"))
            .expect("convert with license override");

        let out_bytes = std::fs::read(&output_path).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "override SPDX must land in vokra.provenance.license"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "MIT still resolves to Permissive"
        );
    }

    /// Empty-string license override falls through to the apache-2.0
    /// default (matches the sibling higgs_audio_v3_tts_4b /
    /// magpietts_v2602 convention where `Some("") => default`; the
    /// wave-B uniform posture guards against a CLI operator passing
    /// `--license ""` and accidentally stamping an empty SPDX).
    #[test]
    fn empty_license_override_falls_back_to_default() {
        let (input_bytes, _) = synthetic_bf16_safetensors();
        let input_path = scratch_path("emptylic-in", "safetensors");
        let output_path = scratch_path("emptylic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_firered_asr_llm_l_file(&input_path, &output_path, Some(""))
            .expect("convert with empty override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE),
            "empty override string must fall through to apache-2.0 default"
        );
    }

    /// Pins the `read == written + skipped_non_float` invariant on a
    /// mixed three-dtype input (F32 + F16 + BF16). The current
    /// safetensors reader admits only these three float dtypes, so an
    /// int tensor makes the parse itself fail; this test therefore
    /// asserts the invariant on an all-float mixed input, which is
    /// the intended contract shape. Mirror of the sibling BF16-
    /// passthrough report-counter posture.
    #[test]
    fn report_read_written_invariant_holds() {
        // Three tensors, all floats: F32 + F16 + BF16 — the report
        // must show read = written = 3 and skipped_non_float = 0 with
        // bf16_passthrough = 1.
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_bytes: Vec<u8> = [0x3C00u16, 0x4000]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bf16_payload = bf16_bytes(&[3.5, -1.25]);
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);
        assert_eq!(bf16_payload.len(), 4);

        let header = format!(
            r#"{{"a":{{"dtype":"F32","shape":[2],"data_offsets":[0,{}]}},"b":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}},"c":{{"dtype":"BF16","shape":[2],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
            f32_bytes.len() + f16_bytes.len() + bf16_payload.len(),
        );
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16_bytes);
        input_bytes.extend_from_slice(&bf16_payload);

        let input_path = scratch_path("invariant-in", "safetensors");
        let output_path = scratch_path("invariant-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_firered_asr_llm_l_file(&input_path, &output_path, None)
            .expect("convert 3-float mix");
        assert_eq!(report.read, 3, "three tensors observed on input");
        assert_eq!(
            report.written, 3,
            "all three floats must ride the pass-through arm"
        );
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must equal input BF16 count (1)"
        );
        // Invariant.
        assert_eq!(report.read, report.written + report.skipped_non_float);
    }

    /// LicenseClass hard-map: the default SPDX `"apache-2.0"` must
    /// resolve to `LicenseClass::Permissive` via
    /// `LicenseClass::from_license_str`. Pins the CLAUDE.md /
    /// license-audit.md classification chain — a future
    /// `from_license_str` refactor that dropped apache-2.0 → Permissive
    /// would silently downgrade this converter's default stamp and
    /// break the publish gate (`LicenseClass::redistributable()` +
    /// `publish-one.sh` check-catalog-reality → check-redistributable
    /// chain).
    #[test]
    fn default_license_resolves_to_permissive() {
        // Mirror of the sibling firered_asr_aed_l / higgs_audio_v3_tts_4b
        // license-class pin so a shared refactor would fire multiple
        // tests, not just this one.
        let class = LicenseClass::from_license_str(DEFAULT_LICENSE);
        assert_eq!(
            class,
            LicenseClass::Permissive,
            "apache-2.0 must resolve to Permissive; a future refactor \
             that dropped this mapping would silently downgrade every \
             FireRedASR-LLM-L artifact's weight_license stamp"
        );
        // Also assert that the class is redistributable (T1 tier) so a
        // future publish gate would not silently refuse for this
        // license class.
        assert!(
            class.redistributable(),
            "Permissive class must be redistributable per LicenseClass::redistributable()"
        );
        assert!(
            class.commercial_ok(),
            "Permissive class must allow commercial use per LicenseClass::commercial_ok()"
        );
    }
}
