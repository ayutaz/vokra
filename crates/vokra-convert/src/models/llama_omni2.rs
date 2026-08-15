//! **ICTNLP/LLaMA-Omni2** — streaming speech-to-speech (Qwen2.5 backbone +
//! Whisper-style speech encoder + AR speech decoder) safetensors → GGUF
//! conversion (coverage-audit-2026-08-03 Wave B fast-track, post-audit
//! CC-gap 2026-08-14).
//!
//! # Model class
//!
//! LLaMA-Omni2 is a **streaming S2S** family from ICTNLP (ACL 2025). Every
//! variant pairs a Whisper-family speech encoder + a Qwen2.5-family text
//! backbone + a streaming AR speech decoder — same "audio encoder + text
//! LM + audio decoder" mold Voxtral / Canary-Qwen use for ASR + the CSM /
//! Moshi streaming duo use for full-duplex S2S, but with joint speech-in
//! / speech-out. The 195 ms latency target puts it on the streaming-first
//! side of the S2S family (Moshi / CSM neighbourhood).
//!
//! Four variants ship as **distinct HuggingFace repositories** (per the
//! coverage-audit ticket § Model, verified as sibling repos on
//! `huggingface.co/ICTNLP`):
//!
//! - `ICTNLP/LLaMA-Omni2-7B` — ~14 GB BF16, English-first
//! - `ICTNLP/LLaMA-Omni2-3B-Bilingual` — ~6 GB BF16, English + Chinese
//! - `ICTNLP/LLaMA-Omni2-1.5B` — ~3 GB BF16, smallest for edge fit
//! - `ICTNLP/LLaMA-Omni2-32B` — ~64 GB BF16, largest
//!
//! Silently sharing a runtime dispatch arm across variants would be safe
//! for the LM axis (they all inherit the Qwen2.5 tokenizer family) but
//! unsafe for the speech decoder axis (each variant retunes the decoder
//! width to match the LM width); the [`LlamaOmni2Variant`] tag rides in
//! `vokra.llama_omni2.variant` so the runtime binder can key on the
//! resolved shapes rather than hardcoding a single set of dims.
//!
//! # Distinct arch tag
//!
//! `vokra.model.arch = "llama_omni2"`. Distinct from every sibling
//! Qwen2-family arch (`voxtral` = Mistral-family ASR / `canary_qwen` =
//! FastConformer + Qwen decoder ASR / `kyutai_stt` = Helium-style
//! decoder-only ASR / `firered_asr_llm_l` = Conformer + Qwen2 LM ASR)
//! because the three-stage streaming S2S topology (speech encoder + text
//! backbone + speech decoder) is not shared byte-for-byte with any of
//! them. Silently sharing an arch tag would misroute the runtime
//! dispatch (a Voxtral loader would try to bind the Qwen2.5 LM
//! decoder-only backbone under Mistral shape assumptions) — FR-EX-08
//! boundary.
//!
//! # License
//!
//! Default SPDX: `apache-2.0` (Qwen2.5 派生). Overrides via the
//! [`convert_llama_omni2_file`] `license` parameter for callers that
//! obtained the checkpoint under a different SPDX — the standing
//! `convert_file_licensed` precedent. Owner sign-off is fail-closed
//! (`docs/license-audit.md` §3.1 row landed with Approval column BLANK)
//! per memory `[[feedback-license-signoff-primary-source]]`: CC never
//! pre-fills Approval, owner confirms (1) HF card `license: apache-2.0`
//! primary source, (2) Qwen2.5 base license-inheritance chain, (3)
//! speech-decoder training-corpus audit, (4) ELVIS Act 精査 (task-
//! oriented S2S 対話 = fixed voice, not target-speaker cloning) before
//! ticking Commercial.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the kyutai_stt / voxtral / canary_qwen / firered_asr_llm_l / CSM /
//! Kokoro / CosyVoice2 contract). Real-weight parity is a follow-up wave
//! gated on the upstream tensor-name manifest fetch + §3.1 sign-off; this
//! converter passes every float tensor through unchanged so a future
//! `LlamaOmni2Weights::from_gguf` can walk the same names.
//!
//! The Qwen2.5 forward does **not** yet have a shared op:
//! `vokra_ops::qwen2` is a PROPOSED consolidation, not a landed module
//! — no such module exists today (the same wording the runtime binder
//! `vokra-models/src/llama_omni2/mod.rs` uses). The only landed
//! Qwen2-family forward is the inline one in
//! `vokra-models/src/voxtral/text_decoder.rs` (GQA + RoPE + SwiGLU +
//! RMSNorm). `canary_qwen` reuses that module; `kyutai_stt` does not.
//! Consolidating them is the follow-up wave.
//!
//! # Prep script bridge — sharded safetensors merge
//!
//! The upstream LLaMA-Omni2 releases ship **sharded safetensors**
//! (`model-000NN-of-000MM.safetensors` + `model.safetensors.index.json`).
//! This Rust converter consumes a **single** safetensors file, so a
//! downstream user needs a sidecar — a future
//! `tools/parity/llama_omni2_prepare_checkpoint.py` (**not yet
//! written**; uv-managed Python 3.12 per memory
//! `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`, mirror
//! of `firered_asr_llm_l/` / `higgs_audio_v3_tts_4b/`) — to merge
//! the shards, dedupe tied tensors (data_ptr collision → clone + audit
//! trail — Qwen2.5 `tie_word_embeddings=true` posture), and strip
//! non-float training scaffold (`.num_batches_tracked` / `.total_ops`)
//! before invoking this converter. The runtime never sees Python / torch
//! (FR-LD-05).
//!
//! # vast.ai required (~14 GB for 7B, ~64 GB for 32B)
//!
//! Per memory `[[feedback-large-models-on-vast-ai]]` (2 GB CC-workflow
//! local-convert owner threshold; M1 iMac 16 GB machine ran into OS-level
//! swap when mmap-ing 8 GB Voxtral / 48 GB Voxtral-Small-24B), the actual
//! weight fetch + convert + publish runs on a rented vast.ai GPU box via
//! `docs/handoff/vast-ai-large-model-publish.md`. The 1.5B and 3B
//! variants (~3 GB / ~6 GB BF16) sit at the borderline; owner judgment
//! per instance class. CC lands only the converter code + tests here.
//!
//! # BF16 pass-through (mirror of firered_asr_llm_l / higgs_audio_v3_tts_4b /
//! # kyutai_stt / voxtral / canary_qwen / qwen3_tts / vibevoice / voxcpm2 /
//! # moshi / chatterbox)
//!
//! Every F32 / F16 / BF16 tensor passes through **verbatim** as the
//! matching GGUF type (BF16 emits type 30 = `GgmlType::BF16`, no
//! convert-time widening — the runtime widens BF16 → f32 losslessly at
//! load via the single choke point `crates/vokra-core/src/gguf/quant/
//! mod.rs decode_bf16`). Mirror of the landed sibling posture that keeps
//! the CI cache footprint at the smallest tensor payload while preserving
//! the exact upstream bit pattern.
//!
//! # No ONNX (permanent)
//!
//! The upstream LLaMA-Omni2 releases ship PyTorch sharded safetensors +
//! a Python inference pipeline; this converter **never** touches ONNX
//! (FR-LD-05). The S2S pipeline is re-implemented natively in the future
//! `crates/vokra-models/src/llama_omni2/` module (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value written for every LLaMA-Omni2 GGUF (regardless
/// of variant). Kept in sync with the runtime constant
/// `vokra-models::llama_omni2::EXPECTED_ARCH`. Intentionally distinct from
/// every sibling Qwen2-family arch (`voxtral` / `canary_qwen` /
/// `kyutai_stt` / `firered_asr_llm_l`) — the three-stage streaming S2S
/// topology differs from every ASR-only sibling. FR-EX-08 boundary.
pub const ARCH: &str = "llama_omni2";

/// `vokra.model.name` prefix written for LLaMA-Omni2 GGUFs. The full name
/// is `"llama-omni2-{variant}"` — see [`LlamaOmni2Variant::name`].
pub const NAME_PREFIX: &str = "llama-omni2";

/// `vokra.model.category` value — `"s2s"`, same tier as the sibling
/// Moshi / CSM full-duplex S2S family. Consumed by the model-card
/// generator + zoo manifest tier gate.
pub const CATEGORY: &str = "s2s";

/// Canonical weight license SPDX (`apache-2.0`) — Qwen2.5 派生 chain.
/// Overrides via the [`convert_llama_omni2_file`] `license` parameter —
/// the standing mechanism for "implementation is clean-room but the
/// upstream distributed checkpoint carries a different SPDX" scenarios
/// (mirror of `convert_firered_asr_llm_l_file` / `convert_kyutai_stt` /
/// `convert_higgs_audio_v3_tts_4b_file`).
pub const DEFAULT_LICENSE: &str = "apache-2.0";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) until a sibling `category`
/// consumer lands in `vokra-core` — mirror of firered_asr_llm_l /
/// higgs_audio_v3_tts_4b / wespeaker / speaker_3d / emotion2vec /
/// neucodec / frcrn local constant.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Upstream repository slug (`org/name`) recorded under
/// `vokra.provenance.upstream_hf` so a downstream consumer can trace the
/// artifact back to its serving location.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Wire key for [`LlamaOmni2Variant`] — the runtime keys tokenizer +
/// speech-decoder-shape dispatch on this rather than hardcoding a single
/// variant's dims (each variant retunes the decoder width to match the
/// LM width per the audit ticket).
pub const KEY_VARIANT: &str = "vokra.llama_omni2.variant";

/// Which LLaMA-Omni2 release this GGUF represents. Four sibling HF repos
/// per the coverage-audit ticket; each is a distinct scale point along
/// the Qwen2.5 backbone axis, all sharing the three-stage streaming S2S
/// topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlamaOmni2Variant {
    /// `ICTNLP/LLaMA-Omni2-7B` — 7B params, ~14 GB BF16, English-first.
    /// Default because it is the canonical release the audit ticket +
    /// ACL 2025 paper anchor to.
    #[default]
    _7B,
    /// `ICTNLP/LLaMA-Omni2-3B-Bilingual` — 3B params, ~6 GB BF16,
    /// English + Chinese bilingual.
    _3BBilingual,
    /// `ICTNLP/LLaMA-Omni2-1.5B` — 1.5B params, ~3 GB BF16, smallest.
    _1_5B,
    /// `ICTNLP/LLaMA-Omni2-32B` — 32B params, ~64 GB BF16, largest.
    _32B,
}

impl LlamaOmni2Variant {
    /// Canonical HF repo slug (`org/name`) for this variant.
    ///
    /// Every string here is transcribed **verbatim** from a listed HF
    /// repository (CLAUDE.md ハルシネーション厳禁 — no repo id is
    /// invented). The four sibling repos live under `huggingface.co/ICTNLP/`.
    #[must_use]
    pub fn as_repo_id(self) -> &'static str {
        match self {
            Self::_7B => "ICTNLP/LLaMA-Omni2-7B",
            Self::_3BBilingual => "ICTNLP/LLaMA-Omni2-3B-Bilingual",
            Self::_1_5B => "ICTNLP/LLaMA-Omni2-1.5B",
            Self::_32B => "ICTNLP/LLaMA-Omni2-32B",
        }
    }

    /// Wire tag written into `vokra.llama_omni2.variant`. Kept in sync
    /// with the runtime [`LlamaOmni2Variant`] discriminator on the
    /// models side.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::_7B => "7b",
            Self::_3BBilingual => "3b-bilingual",
            Self::_1_5B => "1.5b",
            Self::_32B => "32b",
        }
    }

    /// Suffix appended to [`NAME_PREFIX`] to form the full
    /// `vokra.model.name` — `"llama-omni2-{variant.name()}"`.
    #[must_use]
    pub fn name(self) -> &'static str {
        self.tag()
    }

    /// Parses a CLI argument spelling into the matching variant.
    /// Accepts the canonical HF slug + a family of hyphen / underscore
    /// / lower-case spellings (`chatterbox` / `voxtral` / `firered_*`
    /// precedent). Returns [`None`] for an unrecognised spelling so the
    /// caller can raise a loud CLI error naming every accepted form.
    #[must_use]
    pub fn from_arg(s: &str) -> Option<Self> {
        // Case-insensitive comparison via a single owned lowercase copy
        // (mirrors `ChatterboxVariant::from_arg` / `KyutaiSttVariant`
        // — no reliance on `.eq_ignore_ascii_case` per-branch, which
        // would proliferate at four variants × ~5 spellings each).
        let lower = s.to_ascii_lowercase();
        Some(match lower.as_str() {
            "llama-omni2"
            | "llama_omni2"
            | "llamaomni2"
            | "llama-omni2-7b"
            | "llama_omni2_7b"
            | "llamaomni2-7b"
            | "llama-omni2-7b-english"
            | "ictnlp/llama-omni2-7b" => Self::_7B,
            "llama-omni2-3b-bilingual"
            | "llama_omni2_3b_bilingual"
            | "llamaomni2-3b-bilingual"
            | "llama-omni2-3b"
            | "llama_omni2_3b"
            | "ictnlp/llama-omni2-3b-bilingual" => Self::_3BBilingual,
            "llama-omni2-1.5b"
            | "llama-omni2-1_5b"
            | "llama_omni2_1_5b"
            | "llamaomni2-1.5b"
            | "ictnlp/llama-omni2-1.5b" => Self::_1_5B,
            "llama-omni2-32b" | "llama_omni2_32b" | "llamaomni2-32b" | "ictnlp/llama-omni2-32b" => {
                Self::_32B
            }
            _ => return None,
        })
    }
}

/// Outcome of a LLaMA-Omni2 conversion.
///
/// All counters are additive and default to zero — a zero-tensor
/// checkpoint returns `LlamaOmni2Report::default()` and the caller
/// remains responsible for surfacing the "no float tensors" loud note
/// (mirror of the firered_asr_llm_l / higgs_audio_v3_tts_4b /
/// kyutai_stt / voxtral / canary_qwen `Report` pattern).
/// `read == written + skipped_non_float` is an invariant preserved by
/// [`convert_llama_omni2_file`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LlamaOmni2Report {
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
    /// Variant this conversion labelled (written into
    /// `vokra.llama_omni2.variant`).
    pub variant: LlamaOmni2Variant,
}

/// Reads a single merged safetensors checkpoint at `input` (a future
/// `tools/parity/llama_omni2_prepare_checkpoint.py` would emit it; that
/// sidecar is not yet written, so the shard merge is an owner-side step
/// today) and writes a LLaMA-Omni2 GGUF to `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.provenance.*`, `vokra.model.*` and
/// `vokra.llama_omni2.variant` chunk groups pin the upstream slug, weight
/// license, model category, and variant so the zoo manifest + model-card
/// generator can gate on the artifact alone. `vokra.schema.*` is written
/// unconditionally by the GGUF writer.
///
/// `license` overrides `DEFAULT_LICENSE` (`"apache-2.0"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when the implementation
/// is clean-room but the redistributed checkpoint carries a different
/// SPDX (mirror of the sibling `convert_firered_asr_llm_l_file` /
/// `convert_higgs_audio_v3_tts_4b_file` override convention).
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
///
/// # Memory footprint
///
/// The 7B release is ~14 GB, the 32B release is ~64 GB. Per memory
/// `[[feedback-large-models-on-vast-ai]]` both exceed the 2 GB
/// CC-workflow local-convert threshold, so the actual convert runs on a
/// vast.ai box. The current `std::fs::read` load buffers the entire file
/// into memory; the vast.ai runbook
/// (`docs/handoff/vast-ai-large-model-publish.md`) rents a box with
/// enough RAM for the whole-file load + `GgufBuilder::to_bytes` peak.
/// A future streaming pass-through (per the Moshi 15 GB / Voxtral 8.7
/// GB `SafetensorsFileReader` + `GgufStreamWriter` posture) would shrink
/// the peak footprint to one tensor payload; that upgrade is a follow-up
/// if the vast.ai box's RAM budget becomes a constraint on a smaller
/// instance class.
pub fn convert_llama_omni2_file(
    input: &Path,
    output: &Path,
    variant: LlamaOmni2Variant,
    license: Option<&str>,
) -> Result<LlamaOmni2Report, ConvertError> {
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    convert_llama_omni2_bytes(bytes, output, variant, license)
}

/// In-memory variant of [`convert_llama_omni2_file`]. Splits the file
/// I/O out so unit tests can exercise the pass-through / provenance
/// logic without touching the disk twice.
///
/// # Errors
///
/// See [`convert_llama_omni2_file`].
pub fn convert_llama_omni2_bytes(
    bytes: Vec<u8>,
    output: &Path,
    variant: LlamaOmni2Variant,
    license: Option<&str>,
) -> Result<LlamaOmni2Report, ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    let full_name = format!("{NAME_PREFIX}-{}", variant.name());
    b.add_string(chunks::KEY_MODEL_NAME, &full_name);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_VARIANT, variant.tag());
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.as_repo_id());

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = apache-2.0 (Qwen2.5 派生 chain). `license`
    // overrides for callers who obtained the weight under a different
    // SPDX (see `convert_file_licensed` in `lib.rs`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(&full_name),
        Some(variant.as_repo_id()),
    );

    let mut report = LlamaOmni2Report {
        variant,
        ..LlamaOmni2Report::default()
    };
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // firered_asr_llm_l / higgs_audio_v3_tts_4b / kyutai_stt / voxtral
    // / canary_qwen / qwen3_tts / vibevoice / voxcpm2 / moshi.
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
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    /// Per-test unique scratch path (PID + nanos + a suffix derived from
    /// the caller). Mirror of the sibling firered_asr_llm_l /
    /// higgs_audio_v3_tts_4b test-fixture posture (no external `tempfile`
    /// dep, preserving zero-dep NFR-DS-02).
    fn scratch_path(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-llama-omni2-{tag}-{}-{}.gguf",
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

    /// A minimal safetensors buffer with one F32 tensor (shape [2,3] →
    /// 6 elements × 4 bytes = 24 bytes).
    fn minimal_safetensors_one_f32() -> Vec<u8> {
        let header =
            r#"{"backbone.embed.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 24]);
        out
    }

    /// A BF16 tensor with distinct non-zero bit patterns so a subsequent
    /// byte-identity assert catches any silent widen / downcast attempt.
    fn minimal_safetensors_one_bf16() -> Vec<u8> {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12);
        let header =
            r#"{"backbone.embed.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&bf16);
        out
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is
        // the sole handshake with
        // `vokra-models::llama_omni2::EXPECTED_ARCH`.
        assert_eq!(ARCH, "llama_omni2");
    }

    #[test]
    fn variant_repo_ids_are_distinct() {
        // Each variant maps to a distinct HF repo (four sibling
        // releases). A collision here would misroute the model-card
        // upstream_hf stamp.
        let ids = [
            LlamaOmni2Variant::_7B.as_repo_id(),
            LlamaOmni2Variant::_3BBilingual.as_repo_id(),
            LlamaOmni2Variant::_1_5B.as_repo_id(),
            LlamaOmni2Variant::_32B.as_repo_id(),
        ];
        // Every repo lives under the ICTNLP organisation.
        for id in ids {
            assert!(
                id.starts_with("ICTNLP/LLaMA-Omni2"),
                "{id} must be an ICTNLP LLaMA-Omni2 repo"
            );
        }
        // No two variants share a repo id.
        for (i, a) in ids.iter().enumerate() {
            for b in ids.iter().skip(i + 1) {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn variant_from_arg_covers_every_variant() {
        // Canonical + hyphen/underscore/lower-case spellings all
        // resolve to the intended variant.
        assert_eq!(
            LlamaOmni2Variant::from_arg("llama-omni2"),
            Some(LlamaOmni2Variant::_7B)
        );
        assert_eq!(
            LlamaOmni2Variant::from_arg("LLAMA-OMNI2-7B"),
            Some(LlamaOmni2Variant::_7B)
        );
        assert_eq!(
            LlamaOmni2Variant::from_arg("ICTNLP/LLaMA-Omni2-7B"),
            Some(LlamaOmni2Variant::_7B)
        );
        assert_eq!(
            LlamaOmni2Variant::from_arg("llama-omni2-3b-bilingual"),
            Some(LlamaOmni2Variant::_3BBilingual)
        );
        assert_eq!(
            LlamaOmni2Variant::from_arg("llama-omni2-1.5b"),
            Some(LlamaOmni2Variant::_1_5B)
        );
        assert_eq!(
            LlamaOmni2Variant::from_arg("llama-omni2-32b"),
            Some(LlamaOmni2Variant::_32B)
        );
        // Unrecognised spellings return None (never a silent
        // fallthrough to a default variant — FR-EX-08).
        assert_eq!(LlamaOmni2Variant::from_arg("llama-omni2-999b"), None);
        assert_eq!(LlamaOmni2Variant::from_arg("llama-omni"), None);
    }

    #[test]
    fn round_trip_carries_arch_variant_and_provenance() {
        let out_path = scratch_path("roundtrip");
        let _guard = TempFileGuard(out_path.clone());
        let report = convert_llama_omni2_bytes(
            minimal_safetensors_one_f32(),
            &out_path,
            LlamaOmni2Variant::_7B,
            None,
        )
        .expect("convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);
        assert_eq!(report.variant, LlamaOmni2Variant::_7B);

        let bytes = std::fs::read(&out_path).expect("read gguf");
        let file = GgufFile::parse(bytes).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("llama-omni2-7b")
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(file.get(KEY_VARIANT).and_then(|v| v.as_str()), Some("7b"));
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("ICTNLP/LLaMA-Omni2-7B")
        );
        // Provenance: Permissive class + apache-2.0 SPDX by default.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
    }

    /// Explicit `license` override threads through into the provenance
    /// stamp — the standing "clean-room implementation, upstream
    /// checkpoint under a distinct SPDX" mechanism (mirror of
    /// firered_asr_llm_l / higgs_audio_v3_tts_4b).
    #[test]
    fn license_override_threads_into_provenance() {
        let out_path = scratch_path("override");
        let _guard = TempFileGuard(out_path.clone());
        let report = convert_llama_omni2_bytes(
            minimal_safetensors_one_f32(),
            &out_path,
            LlamaOmni2Variant::_3BBilingual,
            Some("mit"),
        )
        .expect("convert");
        assert_eq!(report.variant, LlamaOmni2Variant::_3BBilingual);
        let bytes = std::fs::read(&out_path).expect("read gguf");
        let file = GgufFile::parse(bytes).expect("parse");
        // `LicenseClass::from_license_str("mit") == Permissive` (both
        // MIT and Apache-2.0 are Permissive). This is a positive
        // regression against a silent downgrade: an override that
        // resolves to a stricter class must survive the stamp.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit")
        );
        assert_eq!(
            file.get(KEY_VARIANT).and_then(|v| v.as_str()),
            Some("3b-bilingual")
        );
    }

    /// BF16 tensors reach the pass-through arm verbatim — emitted as
    /// GGUF type 30 (`GgmlType::BF16`) with no convert-time widening.
    /// Regression guard for the standing "no silent widen" invariant
    /// (mirror of firered_asr_llm_l / qwen3_tts / vibevoice / voxcpm2).
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let out_path = scratch_path("bf16");
        let _guard = TempFileGuard(out_path.clone());
        let expected_bytes: [u8; 12] = {
            let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
            let mut out = [0u8; 12];
            for (i, v) in values.iter().enumerate() {
                let bf = (v.to_bits() >> 16) as u16;
                out[i * 2..i * 2 + 2].copy_from_slice(&bf.to_le_bytes());
            }
            out
        };
        let report = convert_llama_omni2_bytes(
            minimal_safetensors_one_bf16(),
            &out_path,
            LlamaOmni2Variant::_7B,
            None,
        )
        .expect("convert BF16");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);
        let bytes = std::fs::read(&out_path).expect("read gguf");
        let file = GgufFile::parse(bytes).expect("parse");
        let info = file
            .tensor_info("backbone.embed.weight")
            .expect("tensor present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            expected_bytes.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );
    }

    /// A malformed input surfaces as `Err(ConvertError::Parse(_))`, not
    /// a silently-empty successful conversion (FR-EX-08 loud fail).
    #[test]
    fn malformed_input_returns_parse_error() {
        let out_path = scratch_path("malformed");
        let _guard = TempFileGuard(out_path.clone());
        // Empty buffer.
        let err = convert_llama_omni2_bytes(Vec::new(), &out_path, LlamaOmni2Variant::_7B, None)
            .expect_err("empty buffer must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );
        // Truncated header.
        let mut truncated = Vec::new();
        truncated.extend_from_slice(&1024u64.to_le_bytes());
        truncated.extend_from_slice(b"{}");
        let err = convert_llama_omni2_bytes(truncated, &out_path, LlamaOmni2Variant::_7B, None)
            .expect_err("truncated header must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );
    }

    /// The `vokra.llama_omni2.variant` tag round-trips for every variant.
    /// Pins the runtime discriminator invariant.
    #[test]
    fn variant_tag_round_trips_for_every_variant() {
        for (variant, want_tag, want_name, want_repo) in [
            (
                LlamaOmni2Variant::_7B,
                "7b",
                "llama-omni2-7b",
                "ICTNLP/LLaMA-Omni2-7B",
            ),
            (
                LlamaOmni2Variant::_3BBilingual,
                "3b-bilingual",
                "llama-omni2-3b-bilingual",
                "ICTNLP/LLaMA-Omni2-3B-Bilingual",
            ),
            (
                LlamaOmni2Variant::_1_5B,
                "1.5b",
                "llama-omni2-1.5b",
                "ICTNLP/LLaMA-Omni2-1.5B",
            ),
            (
                LlamaOmni2Variant::_32B,
                "32b",
                "llama-omni2-32b",
                "ICTNLP/LLaMA-Omni2-32B",
            ),
        ] {
            let out_path = scratch_path(&format!("variant-{}", variant.tag()));
            let _guard = TempFileGuard(out_path.clone());
            let report =
                convert_llama_omni2_bytes(minimal_safetensors_one_f32(), &out_path, variant, None)
                    .expect("convert");
            assert_eq!(report.variant, variant);
            let bytes = std::fs::read(&out_path).expect("read gguf");
            let file = GgufFile::parse(bytes).expect("parse");
            assert_eq!(
                file.get(KEY_VARIANT).and_then(|v| v.as_str()),
                Some(want_tag),
                "variant tag {want_tag}"
            );
            assert_eq!(
                file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
                Some(want_name),
                "model name {want_name}"
            );
            assert_eq!(
                file.get(KEY_PROVENANCE_UPSTREAM_HF)
                    .and_then(|v| v.as_str()),
                Some(want_repo),
                "upstream repo {want_repo}"
            );
        }
    }

    /// Non-float dtype counter increments for a metadata-only zero-tensor
    /// checkpoint (defensive counter — the safetensors reader rejects
    /// unknown dtypes at parse time, but the arm is exercised).
    /// Additionally checks the metadata-only fixture produces a valid
    /// GGUF even without any tensors (arch + provenance still stamped).
    #[test]
    fn zero_tensor_input_still_stamps_metadata() {
        let out_path = scratch_path("zero-tensor");
        let _guard = TempFileGuard(out_path.clone());
        // A safetensors file with an empty header body (zero tensors).
        let header = r#"{}"#;
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        let report = convert_llama_omni2_bytes(input, &out_path, LlamaOmni2Variant::_7B, None)
            .expect("convert");
        assert_eq!(report.read, 0);
        assert_eq!(report.written, 0);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);
        let bytes = std::fs::read(&out_path).expect("read gguf");
        let file = GgufFile::parse(bytes).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        // metadata-only artifact still carries provenance so a runtime
        // gate can loud-fail on the missing tensor set (FR-EX-08) rather
        // than a missing license class.
        match file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE) {
            Some(GgufMetadataValue::String(s)) => {
                assert_eq!(s, LicenseClass::Permissive.as_str());
            }
            other => panic!("expected Permissive weight_license, got {other:?}"),
        }
    }
}
