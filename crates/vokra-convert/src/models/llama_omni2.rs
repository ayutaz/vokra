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
//! # The 2026-08-15 handshake repair
//!
//! Until 2026-08-15 this converter stamped five strings — arch, name,
//! category, `vokra.llama_omni2.variant`, upstream_hf — plus provenance,
//! and nothing else. The runtime binder
//! `crates/vokra-models/src/llama_omni2/mod.rs` declares **eleven**
//! `vokra.llama_omni2.*` keys, of which the converter stamped exactly
//! one. The other ten read back through `read_u32_or_zero` /
//! `read_f32_or`, so every one of them decayed to its `0` placeholder,
//! and `LlamaOmni2Config::validate_for_forward` then refused the load
//! with `InvalidArgument("backbone ill-formed (n_layer=0, d_model=0,
//! n_head=0)")`.
//!
//! So **every GGUF this converter produced failed to load in the binder
//! written for it.** The sibling `kyutai_stt`, which the binder names as
//! its precedent, does stamp its full group; the precedent was real and
//! simply was not carried over.
//!
//! The gap survived because both halves were tested against a mock of
//! the other: the binder's unit tests hand-build their GGUF with
//! `GgufBuilder`, and this module's tests asserted only the five strings
//! it did stamp. Neither side ever ran the real converter into the real
//! binder. `crates/vokra-models/tests/llama_omni2_convert_bind.rs` is
//! that missing test, and it is what keeps the two halves from drifting
//! apart again.
//!
//! # Where each of the ten axes comes from
//!
//! Four are **derived from the tensors themselves**, so they cannot
//! drift away from the weights they describe:
//!
//! - `backbone.n_layer` — the length of the contiguous run of
//!   `{layer_prefix}{i}.` groups. A gap (0, 1, 3) is a hard error, not a
//!   silent truncation that would drop a block.
//! - `backbone.vocab` and `backbone.d_model` — dims 0 and 1 of the token
//!   embedding tensor.
//! - `backbone.intermediate_size` — dim 0 of the SwiGLU gate projection
//!   under layer 0, whose dim 1 is **cross-checked** against the
//!   `d_model` the embedding implied. A disagreement is a hard error.
//!
//! Six cannot be read off any tensor shape and are **required from a
//! `--config` side-car**, so none of them is invented here:
//!
//! - `backbone.n_head` — a `[d_model, d_model]` attention projection has
//!   the same shape for every head count, so the split is simply not
//!   recoverable from the weights. Cross-checked against the derived
//!   `d_model` (must divide it, and yield an even `head_dim` for RoPE
//!   pairs) so a wrong value fails at convert time rather than at load.
//! - `backbone.rope_max_period` and `backbone.rms_norm_eps` — scalar
//!   hyper-parameters that live only in the upstream `config.json`.
//! - `speech_encoder.dim` and `speech_decoder.dim` — the audit ticket
//!   records that each variant retunes the decoder width to match the LM
//!   width, but does not disclose the values, and the binder treats both
//!   as independent axes.
//! - `sample_rate` — the binder carries a documented
//!   `LLAMA_OMNI2_SAMPLE_RATE` constant, but it is deliberately **not**
//!   mirrored as a silent default here: unlike the openWakeWord front-end
//!   axes, nothing downstream re-checks the rate at push time, so a wrong
//!   value would ride into a resampler unchallenged.
//!
//! # Why the tensor-name knobs are safe, and what would make them unsafe
//!
//! The upstream tensor manifest has **not** been fetched — the coverage
//! audit ticket lists the architecture under 参考 and the prepare script
//! it calls for does not exist yet. So the layer / embedding / gate names
//! this converter searches for are **side-car knobs**, defaulting to the
//! bare HuggingFace `Qwen2ForCausalLM` spelling that the sibling
//! `canary_qwen` and `firered_asr_llm_l` fixtures also carry (under their
//! own `decoder.` prefix).
//!
//! A default search key is a different thing from a default model axis,
//! and only the first is admissible here: **a wrong key cannot produce a
//! plausible wrong number.** Prefix matching is exact and anchored at the
//! start of the tensor name, so a name that does not match contributes
//! nothing, and a prefix that matches nothing at all yields zero layers,
//! which is a hard error naming the knob to set. The one residual hazard
//! — matching some *other* subnetwork's layer stack, e.g. the speech
//! encoder's — is what the `d_model` cross-check between the embedding
//! and the gate projection is there to catch.
//!
//! If a future wave transcribes the real manifest, the honest move is to
//! replace these defaults with the transcribed names and cite the source,
//! not to widen the search into fuzzy matching.
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
use vokra_core::json::{self, JsonValue};

use crate::ConvertError;
use crate::safetensors::{SafeTensorInfo, SafetensorsFile};

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

// ---- vokra.llama_omni2.* metadata keys ----------------------------------
//
// Duplicated from the runtime binder
// (`crates/vokra-models/src/llama_omni2/mod.rs`) rather than imported:
// `vokra-convert` depends only on `vokra-core` / `vokra-ops` /
// `vokra-mmap`, and adding a converter → models edge would invert the
// dependency direction the crate split exists to keep clean. This is the
// same cross-crate constant-duplication convention every sibling
// converter uses; what keeps the two copies honest is
// `crates/vokra-models/tests/llama_omni2_convert_bind.rs`, which runs
// this converter into that binder, so a typo here produces a GGUF the
// binder rejects and the test goes red.

/// Wire key for [`LlamaOmni2Variant`] — the runtime keys tokenizer +
/// speech-decoder-shape dispatch on this rather than hardcoding a single
/// variant's dims (each variant retunes the decoder width to match the
/// LM width per the audit ticket).
pub const KEY_VARIANT: &str = "vokra.llama_omni2.variant";

/// GGUF metadata key: PCM sample rate at the speech-encoder boundary
/// (u32 Hz). Required from the side-car — see the module doc.
pub const KEY_SAMPLE_RATE: &str = "vokra.llama_omni2.sample_rate";

/// GGUF metadata key: Qwen2.5 backbone depth (u32). Derived from the
/// contiguous run of layer-prefixed tensors.
pub const KEY_BB_N_LAYER: &str = "vokra.llama_omni2.arch.backbone.n_layer";

/// GGUF metadata key: Qwen2.5 backbone residual width (u32). Derived
/// from dim 1 of the token embedding tensor.
pub const KEY_BB_D_MODEL: &str = "vokra.llama_omni2.arch.backbone.d_model";

/// GGUF metadata key: attention head count (u32). Required from the
/// side-car — not recoverable from any tensor shape.
pub const KEY_BB_N_HEAD: &str = "vokra.llama_omni2.arch.backbone.n_head";

/// GGUF metadata key: tokenizer vocabulary size (u32). Derived from dim
/// 0 of the token embedding tensor.
pub const KEY_BB_VOCAB: &str = "vokra.llama_omni2.arch.backbone.vocab";

/// GGUF metadata key: SwiGLU FFN inner width (u32). Derived from dim 0
/// of the gate projection under layer 0.
pub const KEY_BB_INTERMEDIATE_SIZE: &str = "vokra.llama_omni2.arch.backbone.intermediate_size";

/// GGUF metadata key: RoPE base period (f32). Required from the
/// side-car — a scalar that lives only in the upstream `config.json`.
pub const KEY_BB_ROPE_MAX_PERIOD: &str = "vokra.llama_omni2.arch.backbone.rope_max_period";

/// GGUF metadata key: RMSNorm ε (f32). Required from the side-car — a
/// scalar that lives only in the upstream `config.json`.
pub const KEY_BB_RMS_NORM_EPS: &str = "vokra.llama_omni2.arch.backbone.rms_norm_eps";

/// GGUF metadata key: Whisper-family speech-encoder projection width
/// (u32). Required from the side-car.
pub const KEY_ENC_DIM: &str = "vokra.llama_omni2.arch.speech_encoder.dim";

/// GGUF metadata key: streaming AR speech-decoder width (u32). Required
/// from the side-car.
pub const KEY_DEC_DIM: &str = "vokra.llama_omni2.arch.speech_decoder.dim";

// ---- default tensor-name search keys ------------------------------------
//
// See the module doc section "Why the tensor-name knobs are safe". These
// are SEARCH KEYS, not model axes: every value actually stamped is read
// off a tensor that matched. A key that matches nothing is a hard error.

/// Default tensor-name prefix for the backbone's per-layer groups. Bare
/// HuggingFace `Qwen2ForCausalLM` spelling; override with
/// `"layer_prefix"` in the `--config` side-car.
pub const DEFAULT_LAYER_PREFIX: &str = "model.layers.";

/// Default tensor name for the token embedding table, expected to carry
/// dims `[vocab, d_model]`. Override with `"embedding_tensor"` in the
/// `--config` side-car.
pub const DEFAULT_EMBEDDING_TENSOR: &str = "model.embed_tokens.weight";

/// Default per-layer suffix for the SwiGLU gate projection, expected to
/// carry dims `[intermediate_size, d_model]`. Joined to the layer prefix
/// as `{layer_prefix}{i}.{gate_proj_suffix}`. Override with
/// `"gate_proj_suffix"` in the `--config` side-car.
pub const DEFAULT_GATE_PROJ_SUFFIX: &str = "mlp.gate_proj.weight";

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
/// [`convert`].
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
    /// Backbone depth derived from the layer-prefixed tensor run, and
    /// therefore the value stamped into [`KEY_BB_N_LAYER`].
    pub n_layer: usize,
    /// Residual width derived from the token embedding, and therefore
    /// the value stamped into [`KEY_BB_D_MODEL`].
    pub d_model: usize,
    /// Vocabulary size derived from the token embedding, and therefore
    /// the value stamped into [`KEY_BB_VOCAB`].
    pub vocab: usize,
    /// SwiGLU inner width derived from the gate projection, and
    /// therefore the value stamped into [`KEY_BB_INTERMEDIATE_SIZE`].
    pub intermediate_size: usize,
}

/// Parsed LLaMA-Omni2 `--config` side-car.
///
/// Six model axes are required because they cannot be read off any
/// tensor shape; three tensor-name search keys are optional and default
/// to the bare HuggingFace Qwen2 spelling. See the module doc for why
/// that asymmetry is the honest one.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LlamaOmni2ConvertConfig {
    /// Attention head count. Cross-checked against the derived
    /// `d_model`.
    pub(crate) n_head: u32,
    /// RoPE base period.
    pub(crate) rope_max_period: f32,
    /// RMSNorm ε.
    pub(crate) rms_norm_eps: f32,
    /// PCM sample rate at the speech-encoder boundary, in Hz.
    pub(crate) sample_rate: u32,
    /// Whisper-family speech-encoder projection width.
    pub(crate) speech_encoder_dim: u32,
    /// Streaming AR speech-decoder width.
    pub(crate) speech_decoder_dim: u32,
    /// Tensor-name prefix for the backbone's per-layer groups.
    pub(crate) layer_prefix: String,
    /// Tensor name of the token embedding table.
    pub(crate) embedding_tensor: String,
    /// Per-layer suffix of the SwiGLU gate projection.
    pub(crate) gate_proj_suffix: String,
}

impl LlamaOmni2ConvertConfig {
    /// Parses the JSON side-car.
    ///
    /// Schema (the six model axes are required, the three name keys are
    /// optional):
    ///
    /// ```json
    /// {
    ///   "n_head": 32,
    ///   "rope_max_period": 1000000.0,
    ///   "rms_norm_eps": 1e-6,
    ///   "sample_rate": 16000,
    ///   "speech_encoder_dim": 1280,
    ///   "speech_decoder_dim": 3584,
    ///   "layer_prefix": "model.layers.",
    ///   "embedding_tensor": "model.embed_tokens.weight",
    ///   "gate_proj_suffix": "mlp.gate_proj.weight"
    /// }
    /// ```
    ///
    /// Every value here is transcribed by the operator from the
    /// upstream `config.json`; none is defaulted, because a wrong
    /// hyper-parameter on a composite S2S model produces a GGUF that
    /// loads and is silently wrong.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, ConvertError> {
        let root = json::parse(bytes).map_err(|e| ConvertError::Parse(e.to_string()))?;

        let req_u32 = |key: &str| -> Result<u32, ConvertError> {
            let v = root.get(key).ok_or_else(|| {
                ConvertError::Parse(format!(
                    "llama-omni2 config: required field `{key}` is missing. It cannot be \
                     derived from any tensor shape, and this converter does not invent \
                     model axes — transcribe it from the upstream config.json."
                ))
            })?;
            let raw = v.as_u64().ok_or_else(|| {
                ConvertError::Parse(format!(
                    "llama-omni2 config: `{key}` must be a positive integer"
                ))
            })?;
            let narrowed = u32::try_from(raw).map_err(|_| {
                ConvertError::Parse(format!(
                    "llama-omni2 config: `{key}` = {raw} does not fit in u32"
                ))
            })?;
            if narrowed == 0 {
                return Err(ConvertError::Parse(format!(
                    "llama-omni2 config: `{key}` must be > 0 (the runtime binder's \
                     `validate_for_forward` refuses a 0-sentinel on every hparam)"
                )));
            }
            Ok(narrowed)
        };

        let req_f32 = |key: &str| -> Result<f32, ConvertError> {
            let v = root.get(key).ok_or_else(|| {
                ConvertError::Parse(format!(
                    "llama-omni2 config: required field `{key}` is missing. It is a scalar \
                     that lives only in the upstream config.json — transcribe it rather \
                     than letting the binder fall back to a placeholder."
                ))
            })?;
            let out = match v {
                JsonValue::Int(i) => *i as f32,
                JsonValue::Float(f) => *f as f32,
                _ => {
                    return Err(ConvertError::Parse(format!(
                        "llama-omni2 config: `{key}` must be a number"
                    )));
                }
            };
            if !out.is_finite() || out <= 0.0 {
                return Err(ConvertError::Parse(format!(
                    "llama-omni2 config: `{key}` = {out} must be finite and > 0"
                )));
            }
            Ok(out)
        };

        let opt_name = |key: &str, default: &str| -> Result<String, ConvertError> {
            let Some(v) = root.get(key) else {
                return Ok(default.to_owned());
            };
            let s = v.as_str().ok_or_else(|| {
                ConvertError::Parse(format!("llama-omni2 config: `{key}` must be a string"))
            })?;
            if s.is_empty() {
                return Err(ConvertError::Parse(format!(
                    "llama-omni2 config: `{key}` is empty — an empty tensor-name key would \
                     match every tensor in the checkpoint"
                )));
            }
            Ok(s.to_owned())
        };

        Ok(Self {
            n_head: req_u32("n_head")?,
            rope_max_period: req_f32("rope_max_period")?,
            rms_norm_eps: req_f32("rms_norm_eps")?,
            sample_rate: req_u32("sample_rate")?,
            speech_encoder_dim: req_u32("speech_encoder_dim")?,
            speech_decoder_dim: req_u32("speech_decoder_dim")?,
            layer_prefix: opt_name("layer_prefix", DEFAULT_LAYER_PREFIX)?,
            embedding_tensor: opt_name("embedding_tensor", DEFAULT_EMBEDDING_TENSOR)?,
            gate_proj_suffix: opt_name("gate_proj_suffix", DEFAULT_GATE_PROJ_SUFFIX)?,
        })
    }
}

/// Backbone axes read off the tensors, never off a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BackboneAxes {
    n_layer: usize,
    d_model: usize,
    vocab: usize,
    intermediate_size: usize,
}

/// True when the dtype rides the verbatim pass-through arm, i.e. when
/// the runtime can widen it back to f32.
fn is_passthrough_float(dtype: GgmlType) -> bool {
    matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

/// Looks up a tensor the derivation depends on, failing loudly with the
/// side-car knob that would fix a name mismatch.
///
/// A non-float tensor here is an error rather than a skip: the generic
/// pass-through loop would drop it, and the GGUF would then fail in the
/// binder with a confusing "missing tensor" for something the input
/// carried all along.
fn require_shape_tensor<'a>(
    st: &'a SafetensorsFile,
    name: &str,
    knob: &str,
) -> Result<&'a SafeTensorInfo, ConvertError> {
    let info = st.tensor_info(name).ok_or_else(|| {
        ConvertError::Parse(format!(
            "llama-omni2: tensor `{name}` is not in the safetensors, so the backbone axes \
             it carries cannot be derived. The upstream LLaMA-Omni2 tensor manifest has not \
             been transcribed into this tree, so the name is a `--config` side-car knob: set \
             `{knob}` to the spelling this checkpoint actually uses. Refusing rather than \
             guessing a shape (FR-EX-08)."
        ))
    })?;
    if !is_passthrough_float(info.dtype) {
        return Err(ConvertError::Parse(format!(
            "llama-omni2: tensor `{name}` has dtype {:?}, which is not one of F32 / F16 / \
             BF16. The pass-through arm would skip it and the emitted GGUF would fail to \
             load with a misleading `missing tensor` error.",
            info.dtype
        )));
    }
    Ok(info)
}

/// Derives `n_layer` / `d_model` / `vocab` / `intermediate_size` from
/// the tensors, cross-checking the two independent `d_model` readings
/// against each other.
///
/// Doing the binder's own shape reasoning here means a conversion either
/// produces a loadable GGUF or fails at convert time with a message
/// naming the offending tensor — the failure is never deferred to the
/// operator's first `from_gguf`.
fn derive_backbone_axes(
    st: &SafetensorsFile,
    cfg: &LlamaOmni2ConvertConfig,
) -> Result<BackboneAxes, ConvertError> {
    // `vocab` and `d_model` from the embedding: [vocab, d_model].
    let emb_name = &cfg.embedding_tensor;
    let emb = require_shape_tensor(st, emb_name, "embedding_tensor")?;
    if emb.shape.len() != 2 {
        return Err(ConvertError::Parse(format!(
            "llama-omni2: embedding tensor `{emb_name}` has rank {}, expected rank 2 \
             [vocab, d_model]",
            emb.shape.len()
        )));
    }
    let vocab = emb.shape[0];
    let d_model = emb.shape[1];
    if vocab == 0 || d_model == 0 {
        return Err(ConvertError::Parse(format!(
            "llama-omni2: embedding tensor `{emb_name}` has dims [{vocab}, {d_model}]; both \
             must be > 0"
        )));
    }

    // `n_layer` from the contiguous run of layer groups, each of which
    // must carry the gate projection the next step reads.
    let prefix = &cfg.layer_prefix;
    let gate_name = |i: usize| format!("{prefix}{i}.{}", cfg.gate_proj_suffix);
    let mut n_layer = 0usize;
    while st.tensor_info(&gate_name(n_layer)).is_some() {
        n_layer += 1;
    }
    if n_layer == 0 {
        let probe = gate_name(0);
        return Err(ConvertError::Parse(format!(
            "llama-omni2: no tensor named `{probe}` was found, so the backbone depth cannot \
             be derived. This converter searches for layer groups by exact prefix and does \
             not guess: set `layer_prefix` and/or `gate_proj_suffix` in the --config \
             side-car to the spelling this checkpoint uses. (The defaults are the bare \
             HuggingFace Qwen2 spelling; a composite release commonly nests the backbone \
             under its own prefix, as the sibling canary_qwen / firered_asr_llm_l \
             checkpoints do with `decoder.`.)"
        )));
    }

    // A gap (0, 1, 3) would otherwise be silently truncated to the run
    // length, dropping a block the operator supplied.
    for t in st.tensors() {
        let Some(rest) = t.name.strip_prefix(prefix.as_str()) else {
            continue;
        };
        let idx_str = rest.split('.').next().unwrap_or("");
        let Ok(idx) = idx_str.parse::<usize>() else {
            // Not an indexed group under this prefix (e.g. a sibling
            // `model.norm.weight` when the prefix is `model.`): not a
            // gap, just not a layer.
            continue;
        };
        if idx >= n_layer {
            return Err(ConvertError::Parse(format!(
                "llama-omni2: tensor `{}` carries layer index {idx}, but the contiguous run \
                 of groups ends at {n_layer}. The indices must be dense from 0 — otherwise \
                 this block would be dropped without a word.",
                t.name
            )));
        }
    }

    // `intermediate_size` from layer 0's gate projection, whose second
    // axis is an independent reading of `d_model`. If the layer run
    // matched some other subnetwork's stack, this is where it shows.
    let gate0_name = gate_name(0);
    let gate0 = require_shape_tensor(st, &gate0_name, "gate_proj_suffix")?;
    if gate0.shape.len() != 2 {
        return Err(ConvertError::Parse(format!(
            "llama-omni2: gate projection `{gate0_name}` has rank {}, expected rank 2 \
             [intermediate_size, d_model]",
            gate0.shape.len()
        )));
    }
    let intermediate_size = gate0.shape[0];
    let gate_d_model = gate0.shape[1];
    if intermediate_size == 0 {
        return Err(ConvertError::Parse(format!(
            "llama-omni2: gate projection `{gate0_name}` has intermediate_size 0"
        )));
    }
    if gate_d_model != d_model {
        return Err(ConvertError::Parse(format!(
            "llama-omni2: `{gate0_name}` implies d_model={gate_d_model}, but `{emb_name}` \
             implies d_model={d_model}. Two independent readings of the residual width \
             disagree, which most often means `layer_prefix` matched a different \
             subnetwork's layer stack (the speech encoder's, say) rather than the text \
             backbone's. Refusing rather than stamping either number."
        )));
    }

    let to_usize = |v: u64, what: &str| -> Result<usize, ConvertError> {
        usize::try_from(v).map_err(|_| {
            ConvertError::Parse(format!("llama-omni2: {what}={v} does not fit in usize"))
        })
    };
    Ok(BackboneAxes {
        n_layer,
        d_model: to_usize(d_model, "d_model")?,
        vocab: to_usize(vocab, "vocab")?,
        intermediate_size: to_usize(intermediate_size, "intermediate_size")?,
    })
}

/// Converts a parsed safetensors buffer plus its config side-car into a
/// populated [`GgufBuilder`].
///
/// Split out from the file-level entry point the way `models::crepe` and
/// `models::openwakeword_op` split theirs, so the caller owns the I/O
/// boundary.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.provenance.*`, `vokra.model.*` and full
/// `vokra.llama_omni2.*` chunk groups pin the upstream slug, weight
/// license, model category, variant **and every hparam the runtime
/// binder reads**, so the artifact both loads and gates on its own.
/// `vokra.schema.*` is written unconditionally by the GGUF writer.
///
/// `license` overrides `DEFAULT_LICENSE` (`"apache-2.0"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when the implementation
/// is clean-room but the redistributed checkpoint carries a different
/// SPDX (mirror of the sibling `convert_firered_asr_llm_l_file` /
/// `convert_higgs_audio_v3_tts_4b_file` override convention).
///
/// # Errors
///
/// [`ConvertError::Parse`] for malformed safetensors input, for a
/// derivation whose tensors are absent or mis-shaped, or for a side-car
/// `n_head` that does not satisfy the binder's own head-split rules;
/// [`ConvertError::Gguf`] if a tensor cannot be appended.
pub(crate) fn convert(
    bytes: Vec<u8>,
    cfg: &LlamaOmni2ConvertConfig,
    variant: LlamaOmni2Variant,
    license: Option<&str>,
) -> Result<(GgufBuilder, LlamaOmni2Report), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;
    let axes = derive_backbone_axes(&st, cfg)?;

    // The binder's own `validate_for_forward` rules, applied here so a
    // mismatch names the offending field at convert time instead of
    // failing on the operator's first `from_gguf`.
    let n_head = cfg.n_head as usize;
    // `parse` already rejects a 0, but this struct is constructible
    // in-crate, and `% 0` panics rather than erroring. Guard the divisor
    // before it is used as one.
    if n_head == 0 {
        return Err(ConvertError::Parse(
            "llama-omni2: side-car n_head must be > 0".to_owned(),
        ));
    }
    if axes.d_model % n_head != 0 {
        return Err(ConvertError::Parse(format!(
            "llama-omni2: side-car n_head={n_head} does not divide the derived \
             d_model={}. The runtime binder requires `d_model % n_head == 0` and refuses \
             the load otherwise, so stamping the mismatch would only move the failure \
             downstream.",
            axes.d_model
        )));
    }
    let head_dim = axes.d_model / n_head;
    if head_dim % 2 != 0 {
        return Err(ConvertError::Parse(format!(
            "llama-omni2: side-car n_head={n_head} against the derived d_model={} yields \
             head_dim={head_dim}, which is odd. RoPE rotates coordinate pairs, so the \
             binder requires an even head_dim.",
            axes.d_model
        )));
    }

    let to_u32 = |v: usize, what: &str| -> Result<u32, ConvertError> {
        u32::try_from(v).map_err(|_| {
            ConvertError::Parse(format!("llama-omni2: {what}={v} does not fit in u32"))
        })
    };

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    let full_name = format!("{NAME_PREFIX}-{}", variant.name());
    b.add_string(chunks::KEY_MODEL_NAME, &full_name);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_VARIANT, variant.tag());
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.as_repo_id());

    // The `vokra.llama_omni2.*` chunk group the runtime binder reads.
    // Ten keys beyond the variant tag: omitting any one of them lets it
    // decay to a `0` placeholder, and `validate_for_forward` then
    // refuses the load. Before 2026-08-15 all ten were omitted.
    b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
    b.add_u32(KEY_BB_N_LAYER, to_u32(axes.n_layer, "n_layer")?);
    b.add_u32(KEY_BB_D_MODEL, to_u32(axes.d_model, "d_model")?);
    b.add_u32(KEY_BB_N_HEAD, cfg.n_head);
    b.add_u32(KEY_BB_VOCAB, to_u32(axes.vocab, "vocab")?);
    b.add_u32(
        KEY_BB_INTERMEDIATE_SIZE,
        to_u32(axes.intermediate_size, "intermediate_size")?,
    );
    b.add_f32(KEY_BB_ROPE_MAX_PERIOD, cfg.rope_max_period);
    b.add_f32(KEY_BB_RMS_NORM_EPS, cfg.rms_norm_eps);
    b.add_u32(KEY_ENC_DIM, cfg.speech_encoder_dim);
    b.add_u32(KEY_DEC_DIM, cfg.speech_decoder_dim);

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
        n_layer: axes.n_layer,
        d_model: axes.d_model,
        vocab: axes.vocab,
        intermediate_size: axes.intermediate_size,
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

    Ok((b, report))
}

/// The plain (`--config`-less) path, which **refuses**.
///
/// Six of the eleven `vokra.llama_omni2.*` axes the runtime binder reads
/// cannot be recovered from any tensor shape, and the binder's
/// `validate_for_forward` refuses a `0` on every one of them. So there is
/// no such thing as a loadable LLaMA-Omni2 GGUF built without the
/// side-car: this entry point used to emit one anyway, and every artifact
/// it produced failed at the binder's first load.
///
/// Rather than keep emitting artifacts that cannot load, this mirrors the
/// [`crate::ModelKind::Crepe`] / `openwakeword_op` precedent and routes
/// the caller to `convert_llama_omni2_file_with_config`.
///
/// # Errors
///
/// Always [`ConvertError::Usage`].
pub fn convert_llama_omni2_file(
    input: &Path,
    output: &Path,
    variant: LlamaOmni2Variant,
    license: Option<&str>,
) -> Result<LlamaOmni2Report, ConvertError> {
    let _ = (input, output, variant, license);
    Err(ConvertError::Usage(REFUSAL.to_owned()))
}

/// In-memory sibling of [`convert_llama_omni2_file`], which **refuses**
/// for the same reason: without the side-car there is no set of metadata
/// this function could stamp that the runtime binder would accept.
///
/// # Errors
///
/// Always [`ConvertError::Usage`].
pub fn convert_llama_omni2_bytes(
    bytes: Vec<u8>,
    output: &Path,
    variant: LlamaOmni2Variant,
    license: Option<&str>,
) -> Result<LlamaOmni2Report, ConvertError> {
    let _ = (bytes, output, variant, license);
    Err(ConvertError::Usage(REFUSAL.to_owned()))
}

/// Shared refusal text for both `--config`-less entry points.
const REFUSAL: &str = "llama-omni2 needs a --config config.json carrying `n_head`, \
     `rope_max_period`, `rms_norm_eps`, `sample_rate`, `speech_encoder_dim` and \
     `speech_decoder_dim` (optionally `layer_prefix` / `embedding_tensor` / \
     `gate_proj_suffix` when the checkpoint does not use the bare HuggingFace Qwen2 tensor \
     spelling); use convert_llama_omni2_file_with_config. Those six axes cannot be read off \
     any tensor shape and are not invented here — the runtime binder refuses a 0 on every \
     one of them, so a GGUF built without them cannot load at all.";

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

    /// Assembles a safetensors buffer from `(name, dtype, shape, bytes)`
    /// entries, laid out in the order given.
    fn safetensors_from(entries: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
        let mut fields = Vec::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in entries {
            let start = payload.len();
            payload.extend_from_slice(bytes);
            let dims: Vec<String> = shape.iter().map(u64::to_string).collect();
            fields.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{start},{}]}}",
                dims.join(","),
                payload.len()
            ));
        }
        let header = format!("{{{}}}", fields.join(","));
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&payload);
        out
    }

    /// A tiny but structurally complete backbone: 2 layers, d_model 4,
    /// vocab 8, intermediate_size 6, under the default tensor spelling.
    ///
    /// Every axis the converter derives is therefore knowable from this
    /// fixture alone, and a derivation bug shows up as a wrong number
    /// rather than as a load failure.
    fn tiny_backbone_safetensors() -> Vec<u8> {
        let zeros = |n: usize| vec![0u8; n * 4];
        safetensors_from(&[
            ("model.embed_tokens.weight", "F32", vec![8, 4], zeros(8 * 4)),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "F32",
                vec![6, 4],
                zeros(6 * 4),
            ),
            (
                "model.layers.1.mlp.gate_proj.weight",
                "F32",
                vec![6, 4],
                zeros(6 * 4),
            ),
            ("model.norm.weight", "F32", vec![4], zeros(4)),
        ])
    }

    /// The side-car matching [`tiny_backbone_safetensors`]: n_head 2
    /// against d_model 4 gives head_dim 2, which is even.
    fn tiny_config() -> LlamaOmni2ConvertConfig {
        LlamaOmni2ConvertConfig {
            n_head: 2,
            rope_max_period: 1_000_000.0,
            rms_norm_eps: 1e-6,
            sample_rate: 16_000,
            speech_encoder_dim: 12,
            speech_decoder_dim: 10,
            layer_prefix: DEFAULT_LAYER_PREFIX.to_owned(),
            embedding_tensor: DEFAULT_EMBEDDING_TENSOR.to_owned(),
            gate_proj_suffix: DEFAULT_GATE_PROJ_SUFFIX.to_owned(),
        }
    }

    /// Runs [`convert`] and serializes, so a test can inspect the GGUF.
    fn convert_to_gguf(
        bytes: Vec<u8>,
        cfg: &LlamaOmni2ConvertConfig,
        variant: LlamaOmni2Variant,
        license: Option<&str>,
    ) -> (GgufFile, LlamaOmni2Report) {
        let (b, report) = convert(bytes, cfg, variant, license).expect("convert");
        let out = b.to_bytes().expect("serialize");
        (GgufFile::parse(out).expect("parse"), report)
    }

    /// The same backbone with a BF16 embedding carrying distinct
    /// non-zero bit patterns, so a byte-identity assert catches any
    /// silent widen / downcast attempt.
    fn tiny_backbone_bf16_embedding() -> (Vec<u8>, Vec<u8>) {
        // 8 * 4 = 32 values, cycling a distinctive set.
        let pattern: [f32; 8] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0, -0.03125, 7.0];
        let values: Vec<f32> = (0..32).map(|i| pattern[i % pattern.len()]).collect();
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 64);
        let zeros = |n: usize| vec![0u8; n * 4];
        let buf = safetensors_from(&[
            (
                "model.embed_tokens.weight",
                "BF16",
                vec![8, 4],
                bf16.clone(),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "F32",
                vec![6, 4],
                zeros(6 * 4),
            ),
        ]);
        (buf, bf16)
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

    /// THE fence on this side of the pair: every one of the eleven keys
    /// the runtime binder declares is actually stamped, with the value
    /// the checkpoint or the side-car implied.
    ///
    /// Before 2026-08-15 only `variant` was written and the other ten
    /// decayed to `0` in the binder, so this test is what would have
    /// caught the defect from the converter's side.
    #[test]
    fn all_eleven_runtime_metadata_keys_are_stamped() {
        let cfg = tiny_config();
        let (file, report) = convert_to_gguf(
            tiny_backbone_safetensors(),
            &cfg,
            LlamaOmni2Variant::_7B,
            None,
        );

        for key in [
            KEY_VARIANT,
            KEY_SAMPLE_RATE,
            KEY_BB_N_LAYER,
            KEY_BB_D_MODEL,
            KEY_BB_N_HEAD,
            KEY_BB_VOCAB,
            KEY_BB_INTERMEDIATE_SIZE,
            KEY_BB_ROPE_MAX_PERIOD,
            KEY_BB_RMS_NORM_EPS,
            KEY_ENC_DIM,
            KEY_DEC_DIM,
        ] {
            assert!(file.get(key).is_some(), "missing metadata key `{key}`");
        }

        // Derived from the tensors.
        assert_eq!(file.get(KEY_BB_N_LAYER).and_then(|v| v.as_u64()), Some(2));
        assert_eq!(file.get(KEY_BB_D_MODEL).and_then(|v| v.as_u64()), Some(4));
        assert_eq!(file.get(KEY_BB_VOCAB).and_then(|v| v.as_u64()), Some(8));
        assert_eq!(
            file.get(KEY_BB_INTERMEDIATE_SIZE).and_then(|v| v.as_u64()),
            Some(6)
        );
        // Taken from the side-car — the axes that cannot be derived and
        // are never invented.
        assert_eq!(file.get(KEY_BB_N_HEAD).and_then(|v| v.as_u64()), Some(2));
        assert_eq!(
            file.get(KEY_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(16_000)
        );
        assert_eq!(file.get(KEY_ENC_DIM).and_then(|v| v.as_u64()), Some(12));
        assert_eq!(file.get(KEY_DEC_DIM).and_then(|v| v.as_u64()), Some(10));
        match file.get(KEY_BB_ROPE_MAX_PERIOD) {
            Some(GgufMetadataValue::F32(v)) => assert!((*v - 1_000_000.0).abs() < 1e-3),
            other => panic!("rope_max_period must be an F32, got {other:?}"),
        }
        match file.get(KEY_BB_RMS_NORM_EPS) {
            Some(GgufMetadataValue::F32(v)) => assert!((*v - 1e-6).abs() < 1e-12),
            other => panic!("rms_norm_eps must be an F32, got {other:?}"),
        }

        // The report echoes what was derived, so the CLI note is not a
        // second, independently-computed claim.
        assert_eq!(report.n_layer, 2);
        assert_eq!(report.d_model, 4);
        assert_eq!(report.vocab, 8);
        assert_eq!(report.intermediate_size, 6);
    }

    #[test]
    fn round_trip_carries_arch_variant_and_provenance() {
        let cfg = tiny_config();
        let (file, report) = convert_to_gguf(
            tiny_backbone_safetensors(),
            &cfg,
            LlamaOmni2Variant::_7B,
            None,
        );
        assert_eq!(report.read, 4);
        assert_eq!(report.written, 4);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);
        assert_eq!(report.variant, LlamaOmni2Variant::_7B);

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
        let cfg = tiny_config();
        let (file, report) = convert_to_gguf(
            tiny_backbone_safetensors(),
            &cfg,
            LlamaOmni2Variant::_3BBilingual,
            Some("mit"),
        );
        assert_eq!(report.variant, LlamaOmni2Variant::_3BBilingual);
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
        let (input, expected_bytes) = tiny_backbone_bf16_embedding();
        let cfg = tiny_config();
        let (file, report) = convert_to_gguf(input, &cfg, LlamaOmni2Variant::_7B, None);
        assert_eq!(report.written, 2);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);
        let info = file
            .tensor_info("model.embed_tokens.weight")
            .expect("tensor present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16"
        );
        assert_eq!(info.dimensions, vec![8, 4]);
        assert_eq!(
            file.tensor_bytes(info),
            expected_bytes.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );
        // A BF16 embedding still yields the same derived axes: the
        // derivation reads shapes, not payloads.
        assert_eq!(report.vocab, 8);
        assert_eq!(report.d_model, 4);
    }

    /// A malformed input surfaces as `Err(ConvertError::Parse(_))`, not
    /// a silently-empty successful conversion (FR-EX-08 loud fail).
    #[test]
    fn malformed_input_returns_parse_error() {
        let cfg = tiny_config();
        // Empty buffer.
        let Err(err) = convert(Vec::new(), &cfg, LlamaOmni2Variant::_7B, None) else {
            panic!("empty buffer must be rejected");
        };
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );
        // Truncated header.
        let mut truncated = Vec::new();
        truncated.extend_from_slice(&1024u64.to_le_bytes());
        truncated.extend_from_slice(b"{}");
        let Err(err) = convert(truncated, &cfg, LlamaOmni2Variant::_7B, None) else {
            panic!("truncated header must be rejected");
        };
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );
    }

    /// Both `--config`-less entry points refuse rather than emitting a
    /// GGUF the binder cannot load, and each names the working route.
    #[test]
    fn plain_paths_refuse_and_name_the_config_route() {
        let out_path = scratch_path("refusal");
        let _guard = TempFileGuard(out_path.clone());
        let in_path = scratch_path("refusal-in");
        let _in_guard = TempFileGuard(in_path.clone());
        std::fs::write(&in_path, tiny_backbone_safetensors()).expect("write input");

        let Err(err) = convert_llama_omni2_file(&in_path, &out_path, LlamaOmni2Variant::_7B, None)
        else {
            panic!("the --config-less file path must refuse");
        };
        let ConvertError::Usage(msg) = err else {
            panic!("expected ConvertError::Usage, got {err:?}");
        };
        assert!(
            msg.contains("convert_llama_omni2_file_with_config"),
            "the refusal must name the working entry point: {msg}"
        );
        assert!(
            msg.contains("n_head"),
            "the refusal must name the required fields: {msg}"
        );

        let Err(err) = convert_llama_omni2_bytes(
            tiny_backbone_safetensors(),
            &out_path,
            LlamaOmni2Variant::_7B,
            None,
        ) else {
            panic!("the --config-less bytes path must refuse");
        };
        assert!(matches!(err, ConvertError::Usage(_)));
        assert!(
            !out_path.exists(),
            "a refusing path must not leave an unloadable GGUF behind"
        );
    }

    /// A layer prefix that matches nothing is a hard error naming the
    /// side-car knob — never a silent `n_layer = 0`, which is exactly
    /// what shipped before this repair.
    #[test]
    fn unmatched_layer_prefix_is_refused_not_defaulted() {
        let cfg = LlamaOmni2ConvertConfig {
            layer_prefix: "language_model.model.layers.".to_owned(),
            ..tiny_config()
        };
        let Err(err) = convert(
            tiny_backbone_safetensors(),
            &cfg,
            LlamaOmni2Variant::_7B,
            None,
        ) else {
            panic!("a prefix matching no tensor must be refused");
        };
        let ConvertError::Parse(msg) = err else {
            panic!("expected ConvertError::Parse, got {err:?}");
        };
        assert!(
            msg.contains("layer_prefix"),
            "the refusal must name the knob that fixes it: {msg}"
        );
    }

    /// A missing embedding tensor is refused with the knob that fixes
    /// it, rather than a guessed `[vocab, d_model]`.
    #[test]
    fn unmatched_embedding_tensor_is_refused() {
        let cfg = LlamaOmni2ConvertConfig {
            embedding_tensor: "model.wte.weight".to_owned(),
            ..tiny_config()
        };
        let Err(err) = convert(
            tiny_backbone_safetensors(),
            &cfg,
            LlamaOmni2Variant::_7B,
            None,
        ) else {
            panic!("a missing embedding tensor must be refused");
        };
        let ConvertError::Parse(msg) = err else {
            panic!("expected ConvertError::Parse, got {err:?}");
        };
        assert!(msg.contains("embedding_tensor"), "{msg}");
    }

    /// A gap in the layer indices is a hard error, not a silent
    /// truncation that would drop a block.
    #[test]
    fn gapped_layer_indices_are_refused() {
        let zeros = |n: usize| vec![0u8; n * 4];
        let input = safetensors_from(&[
            ("model.embed_tokens.weight", "F32", vec![8, 4], zeros(8 * 4)),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "F32",
                vec![6, 4],
                zeros(6 * 4),
            ),
            // index 1 missing, index 2 present
            (
                "model.layers.2.mlp.gate_proj.weight",
                "F32",
                vec![6, 4],
                zeros(6 * 4),
            ),
        ]);
        let cfg = tiny_config();
        let Err(err) = convert(input, &cfg, LlamaOmni2Variant::_7B, None) else {
            panic!("a gapped layer run must be refused");
        };
        let ConvertError::Parse(msg) = err else {
            panic!("expected ConvertError::Parse, got {err:?}");
        };
        assert!(msg.contains("layer index 2"), "{msg}");
    }

    /// The two independent readings of `d_model` must agree. This is the
    /// check that catches a `layer_prefix` which matched some other
    /// subnetwork's layer stack.
    #[test]
    fn disagreeing_d_model_readings_are_refused() {
        let zeros = |n: usize| vec![0u8; n * 4];
        let input = safetensors_from(&[
            ("model.embed_tokens.weight", "F32", vec![8, 4], zeros(8 * 4)),
            // gate projection implies d_model = 5, not 4
            (
                "model.layers.0.mlp.gate_proj.weight",
                "F32",
                vec![6, 5],
                zeros(6 * 5),
            ),
        ]);
        let cfg = tiny_config();
        let Err(err) = convert(input, &cfg, LlamaOmni2Variant::_7B, None) else {
            panic!("disagreeing d_model readings must be refused");
        };
        let ConvertError::Parse(msg) = err else {
            panic!("expected ConvertError::Parse, got {err:?}");
        };
        assert!(msg.contains("d_model"), "{msg}");
    }

    /// The binder's own head-split rules are enforced at convert time,
    /// so a bad `n_head` fails here rather than at the operator's first
    /// `from_gguf`.
    #[test]
    fn n_head_is_cross_checked_against_the_derived_d_model() {
        // d_model = 4; 3 does not divide it.
        let cfg = LlamaOmni2ConvertConfig {
            n_head: 3,
            ..tiny_config()
        };
        let Err(err) = convert(
            tiny_backbone_safetensors(),
            &cfg,
            LlamaOmni2Variant::_7B,
            None,
        ) else {
            panic!("an n_head that does not divide d_model must be refused");
        };
        let ConvertError::Parse(msg) = err else {
            panic!("expected ConvertError::Parse, got {err:?}");
        };
        assert!(msg.contains("does not divide"), "{msg}");

        // d_model = 4, n_head = 4 → head_dim = 1, which is odd.
        let cfg = LlamaOmni2ConvertConfig {
            n_head: 4,
            ..tiny_config()
        };
        let Err(err) = convert(
            tiny_backbone_safetensors(),
            &cfg,
            LlamaOmni2Variant::_7B,
            None,
        ) else {
            panic!("an odd head_dim must be refused");
        };
        let ConvertError::Parse(msg) = err else {
            panic!("expected ConvertError::Parse, got {err:?}");
        };
        assert!(msg.contains("odd"), "{msg}");
    }

    /// Each required side-car field is genuinely required, and the
    /// refusal says so rather than substituting a plausible number.
    #[test]
    fn every_required_config_field_is_demanded() {
        let full = r#"{"n_head":2,"rope_max_period":1000000.0,"rms_norm_eps":0.000001,
                       "sample_rate":16000,"speech_encoder_dim":12,"speech_decoder_dim":10}"#;
        LlamaOmni2ConvertConfig::parse(full.as_bytes()).expect("the complete side-car parses");

        for missing in [
            "n_head",
            "rope_max_period",
            "rms_norm_eps",
            "sample_rate",
            "speech_encoder_dim",
            "speech_decoder_dim",
        ] {
            // Rebuild the JSON without the field under test.
            let mut fields: Vec<&str> = vec![
                r#""n_head":2"#,
                r#""rope_max_period":1000000.0"#,
                r#""rms_norm_eps":0.000001"#,
                r#""sample_rate":16000"#,
                r#""speech_encoder_dim":12"#,
                r#""speech_decoder_dim":10"#,
            ];
            fields.retain(|f| !f.contains(missing));
            let json = format!("{{{}}}", fields.join(","));
            let Err(err) = LlamaOmni2ConvertConfig::parse(json.as_bytes()) else {
                panic!("a side-car missing `{missing}` must be refused");
            };
            let ConvertError::Parse(msg) = err else {
                panic!("expected ConvertError::Parse for `{missing}`, got {err:?}");
            };
            assert!(
                msg.contains(missing),
                "the refusal for `{missing}` must name it: {msg}"
            );
        }
    }

    /// Tensor-name keys default to the bare HuggingFace Qwen2 spelling
    /// and are overridable, so a nested checkpoint is expressible
    /// without touching this file.
    #[test]
    fn tensor_name_keys_default_and_override() {
        let defaults = LlamaOmni2ConvertConfig::parse(
            br#"{"n_head":2,"rope_max_period":1e6,"rms_norm_eps":1e-6,"sample_rate":16000,
                 "speech_encoder_dim":12,"speech_decoder_dim":10}"#,
        )
        .expect("parse");
        assert_eq!(defaults.layer_prefix, DEFAULT_LAYER_PREFIX);
        assert_eq!(defaults.embedding_tensor, DEFAULT_EMBEDDING_TENSOR);
        assert_eq!(defaults.gate_proj_suffix, DEFAULT_GATE_PROJ_SUFFIX);

        let overridden = LlamaOmni2ConvertConfig::parse(
            br#"{"n_head":2,"rope_max_period":1e6,"rms_norm_eps":1e-6,"sample_rate":16000,
                 "speech_encoder_dim":12,"speech_decoder_dim":10,
                 "layer_prefix":"language_model.model.layers.",
                 "embedding_tensor":"language_model.model.embed_tokens.weight",
                 "gate_proj_suffix":"mlp.gate.weight"}"#,
        )
        .expect("parse");
        assert_eq!(overridden.layer_prefix, "language_model.model.layers.");
        assert_eq!(
            overridden.embedding_tensor,
            "language_model.model.embed_tokens.weight"
        );
        assert_eq!(overridden.gate_proj_suffix, "mlp.gate.weight");

        // A nested checkpoint really converts under the overrides.
        let zeros = |n: usize| vec![0u8; n * 4];
        let input = safetensors_from(&[
            (
                "language_model.model.embed_tokens.weight",
                "F32",
                vec![8, 4],
                zeros(8 * 4),
            ),
            (
                "language_model.model.layers.0.mlp.gate.weight",
                "F32",
                vec![6, 4],
                zeros(6 * 4),
            ),
        ]);
        let (_, report) = convert_to_gguf(input, &overridden, LlamaOmni2Variant::_7B, None);
        assert_eq!(report.n_layer, 1);
        assert_eq!(report.d_model, 4);
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
            let cfg = tiny_config();
            let (file, report) = convert_to_gguf(tiny_backbone_safetensors(), &cfg, variant, None);
            assert_eq!(report.variant, variant);
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

    /// A zero-tensor checkpoint is now **refused**, not stamped.
    ///
    /// It used to convert successfully and emit a metadata-only GGUF.
    /// That artifact could never load — the binder derives nothing from
    /// an empty tensor set and refuses every `0` axis — so producing one
    /// only moved the failure to the operator. The derivation now fails
    /// at the first missing tensor and names the knob (FR-EX-08).
    #[test]
    fn zero_tensor_input_is_refused() {
        // A safetensors file with an empty header body (zero tensors).
        let header = r#"{}"#;
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        let cfg = tiny_config();
        let Err(err) = convert(input, &cfg, LlamaOmni2Variant::_7B, None) else {
            panic!("a zero-tensor checkpoint has no derivable axes and must be refused");
        };
        let ConvertError::Parse(msg) = err else {
            panic!("expected ConvertError::Parse, got {err:?}");
        };
        assert!(
            msg.contains("embedding_tensor"),
            "the refusal must name the first thing it could not find: {msg}"
        );
    }

    /// Provenance still rides on a successful conversion — the M2-13
    /// weight-license gate keys on it, so it must survive alongside the
    /// new hparam group.
    #[test]
    fn provenance_survives_alongside_the_hparam_group() {
        let cfg = tiny_config();
        let (file, _) = convert_to_gguf(
            tiny_backbone_safetensors(),
            &cfg,
            LlamaOmni2Variant::_7B,
            None,
        );
        match file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE) {
            Some(GgufMetadataValue::String(s)) => {
                assert_eq!(s, LicenseClass::Permissive.as_str());
            }
            other => panic!("expected Permissive weight_license, got {other:?}"),
        }
    }
}
