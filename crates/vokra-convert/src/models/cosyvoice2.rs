//! CosyVoice2 (Flow Matching + Mimi codec + chunk-aware CFM): safetensors
//! checkpoint → GGUF conversion (M3-09-T03 / T04).
//!
//! Input: the upstream `FunAudioLLM/CosyVoice2-0.5B` LLM checkpoint
//! (`llm.pt` exported to safetensors with verbatim tensor names — upstream
//! ships torch pickles, no monolithic safetensors; the export recipe is in
//! `docs/bench-baselines/m1-real-weight-eval-2026-07-16/report.md` §6-2).
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*` and
//! `vokra.cosyvoice2.*` metadata chunks the native CosyVoice2 implementation
//! (`crates/vokra-models/src/cosyvoice2/`) reads.
//!
//! # Hparam derivation (T04, closed by the 2026-07-16 real-weight eval)
//!
//! The Qwen2-0.5B backbone hparams are written from two sources:
//!
//! - **Shape-derived (always)**: `vocab_size` / `hidden_dim` from the
//!   `llm.model.model.embed_tokens.weight` shape, `n_layer` from the
//!   contiguous `llm.model.model.layers.{i}.*` block count, `ffn_dim` from
//!   the layer-0 `mlp.gate_proj.weight` shape. These are unambiguous.
//! - **`--config` (upstream HF `config.json`)**: the attention head split
//!   (`num_attention_heads` / `num_key_value_heads`) is **not**
//!   shape-derivable — `q_out == hidden` and `kv_out = n_head_kv ×
//!   head_dim` leave `head_dim` free (any divisor of `kv_out` yields a
//!   consistent split, and RoPE θ striding + the softmax scale depend on
//!   it). `rope_theta`, `rms_norm_eps` and `max_position_embeddings` come
//!   from the same file. Without `--config` those keys stay `0`-absent and
//!   the runtime refuses the LLM bind — loud, never guessed (FR-EX-08).
//!
//! Cross-checks between the config and the tensor shapes (hidden size,
//! layer count, FFN width, vocab, GQA algebra) fail the conversion loudly —
//! a config from a different model must not produce a silently-wrong GGUF.
//!
//! # Q/K/V attention biases
//!
//! The Qwen2 family ships attention Q/K/V biases (measured layer-0 max
//! |bias|: q = 51.13, k = 62.49 — eval report §8 row 4) and they are copied
//! verbatim like every other tensor. The converter validates that bias
//! presence is uniform (all three per layer, all layers) so a truncated
//! export fails here instead of at runtime bind.
//!
//! # Tensor naming contract (T03)
//!
//! GGUF tensor names are the **upstream safetensors names verbatim** (same
//! contract Whisper / Kokoro use).
//!
//! # No ONNX (permanent constraint)
//!
//! The converter never touches an ONNX graph — CosyVoice2 ships as torch
//! checkpoints + a Python-side pipeline; the pipeline is re-implemented in
//! Rust by the runtime crate (whisper.cpp 型 self re-implementation,
//! CLAUDE.md 設計判断 4).

use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::json::{self, JsonValue};
use crate::safetensors::{SafeTensorInfo, SafetensorsFile};

/// `vokra.model.arch` value written for CosyVoice2 GGUFs. Kept in sync with
/// the runtime constant `crates/vokra-models/src/cosyvoice2::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "cosyvoice2";
/// `vokra.model.name` value written for the CosyVoice2 GGUF.
pub(crate) const NAME: &str = "cosyvoice2-0.5b";

// --- vokra.cosyvoice2.* metadata keys (T04 chunk design) --------------------
//
// Kept as constants inside this module (mirror the piper-plus / kokoro
// pattern): CosyVoice2-specific keys live with the CosyVoice2 model, not in
// `vokra-core::gguf::chunks`.
//
// The runtime reads back the same keys via
// `crates/vokra-models/src/cosyvoice2/config.rs` (+ `llm.rs` for the
// `arch.n_head_kv` / `arch.rope_base` / `arch.rms_norm_eps` / `arch.n_ctx`
// group); the two crates intentionally duplicate the constant strings (the
// runtime crate cannot depend on `vokra-convert`, and `vokra-convert` cannot
// depend on `vokra-models` — both depend only on `vokra-core`). Round-trip
// tests on both sides catch any drift.

const KEY_SAMPLE_RATE: &str = "vokra.cosyvoice2.sample_rate";
const KEY_VOCAB_SIZE: &str = "vokra.cosyvoice2.arch.vocab_size";
const KEY_HIDDEN_DIM: &str = "vokra.cosyvoice2.arch.hidden_dim";
const KEY_N_LAYER: &str = "vokra.cosyvoice2.arch.n_layer";
const KEY_N_HEAD: &str = "vokra.cosyvoice2.arch.n_head";
const KEY_FFN_DIM: &str = "vokra.cosyvoice2.arch.ffn_dim";
const KEY_N_HEAD_KV: &str = "vokra.cosyvoice2.arch.n_head_kv";
const KEY_ROPE_BASE: &str = "vokra.cosyvoice2.arch.rope_base";
const KEY_RMS_NORM_EPS: &str = "vokra.cosyvoice2.arch.rms_norm_eps";
const KEY_N_CTX: &str = "vokra.cosyvoice2.arch.n_ctx";
const KEY_FLOW_NFE: &str = "vokra.cosyvoice2.flow.nfe";
const KEY_FLOW_SCHEDULE: &str = "vokra.cosyvoice2.flow.schedule";
const KEY_MIMI_N_CODEBOOKS: &str = "vokra.cosyvoice2.mimi.n_codebooks";
const KEY_MIMI_CODEBOOK_SIZE: &str = "vokra.cosyvoice2.mimi.codebook_size";
const KEY_MIMI_D_MODEL: &str = "vokra.cosyvoice2.mimi.d_model";
const KEY_STREAMING_CHUNK_SIZE: &str = "vokra.cosyvoice2.streaming.chunk_size";
const KEY_STREAMING_CHUNK_HOP: &str = "vokra.cosyvoice2.streaming.chunk_hop";

// --- Text tokenizer (T06): raw Qwen2 vocab.json + merges.txt, U8 embed ------
//
// Mirrors the runtime constants in
// `crates/vokra-models/src/cosyvoice2/text_encoder.rs`
// (`KEY_TOKENIZER_VOCAB` / `KEY_TOKENIZER_MERGES`) under the two-crate
// constant rule; a round-trip test on each side catches drift.
const KEY_TOKENIZER_VOCAB: &str = "vokra.cosyvoice2.tokenizer.vocab";
const KEY_TOKENIZER_MERGES: &str = "vokra.cosyvoice2.tokenizer.merges";

// --- Upstream tensor names (eval-recorded, never invented) ------------------
//
// Source of truth: `docs/bench-baselines/m1-real-weight-eval-2026-07-16/`
// (report §4 / §6-2 + `llm-pt-manifest.tsv`) — the deployed
// `FunAudioLLM/CosyVoice2-0.5B` `llm.pt` state-dict names. Duplicated in
// `crates/vokra-models/src/cosyvoice2/llm.rs` (`T_TOKEN_EMB` etc.) under the
// two-crate constant rule above.

const T_TOKEN_EMB: &str = "llm.model.model.embed_tokens.weight";

/// Per-layer tensor-name prefix.
fn layer_prefix(i: usize) -> String {
    format!("llm.model.model.layers.{i}.")
}

/// CosyVoice2 output PCM sample rate (Hz).
///
/// Sourced from the CosyVoice2 model card (24 kHz output); this is the same
/// "model-card invariant" exception Kokoro uses for its 24 kHz value.
const COSYVOICE2_SAMPLE_RATE: u32 = 24_000;

/// Canonical Mimi RVQ shape (8 codebooks × 2048 entries × 512 dim).
///
/// Sourced from the Mimi paper (Kyutai) / M3-06 module documentation —
/// stable model-card invariants, not invented numbers. The runtime rejects
/// a `0` codec shape at load (`MimiBridge::from_config`), so we do
/// **not** emit `0` placeholders on these three axes. NOTE (recorded by the
/// 2026-07-16 eval as an open owner/CC design item): the upstream
/// CosyVoice2-0.5B release does **not** ship Mimi — it ships an FSQ
/// `speech_tokenizer_v2` + `flow.pt` + `hift.pt`; the Mimi bridge is the
/// M3-06 design decision this converter follows until that item is
/// resolved.
const MIMI_N_CODEBOOKS: u32 = 8;
const MIMI_CODEBOOK_SIZE: u32 = 2048;
const MIMI_D_MODEL: u32 = 512;

/// Hparams derived while converting (surfaced through
/// [`CosyVoice2Report`] so the CLI can print what was written).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct DerivedHparams {
    /// Embedding rows (`llm.model.model.embed_tokens.weight` dim 0).
    pub(crate) vocab_size: u32,
    /// Embedding cols / model width (dim 1).
    pub(crate) hidden_dim: u32,
    /// Contiguous transformer block count.
    pub(crate) n_layer: u32,
    /// SwiGLU inner width (`mlp.gate_proj.weight` dim 0).
    pub(crate) ffn_dim: u32,
    /// Query heads — from `--config`; `0` = unknown (not shape-derivable).
    pub(crate) n_head: u32,
    /// KV heads — from `--config`; `0` = unknown.
    pub(crate) n_head_kv: u32,
    /// Max positions — from `--config`; `0` = unknown.
    pub(crate) n_ctx: u32,
    /// True when the checkpoint ships Q/K/V attention biases (Qwen2).
    pub(crate) has_attn_bias: bool,
}

/// Outcome of a CosyVoice2 conversion.
#[derive(Debug, Default)]
pub(crate) struct CosyVoice2Report {
    /// Number of float weight tensors written to the GGUF.
    pub(crate) written: usize,
    /// Tensors whose dtype falls outside the F32/F16 range and were skipped.
    ///
    /// The upstream safetensors reader already rejects unknown dtypes at
    /// parse time (`SafetensorsError::UnsupportedDtype`), so this counter
    /// is defensive/forward-compat (same rationale as Kokoro).
    pub(crate) skipped_non_float: usize,
    /// Shape/config-derived hparams actually written; `None` when the
    /// buffer does not carry the LLM backbone tensors (scaffold inputs).
    pub(crate) derived: Option<DerivedHparams>,
    /// Whether the Qwen2 text tokenizer (`vocab.json` + `merges.txt`) was
    /// embedded as the `vokra.cosyvoice2.tokenizer.*` U8 chunks (T06).
    pub(crate) tokenizer_embedded: bool,
    /// Diagnostic notes surfaced to the CLI operator. The converter never
    /// fails on a note — hard inconsistencies are `ConvertError`s instead —
    /// but a loud warning is printed so the operator does not learn about
    /// a degraded conversion only at load time.
    pub(crate) notes: Vec<String>,
}

/// Raw Qwen2 text-tokenizer side-car files (T06): the upstream `vocab.json`
/// and `merges.txt` bytes, embedded verbatim as U8 arrays under
/// `vokra.cosyvoice2.tokenizer.vocab` / `.merges` (the Whisper / Voxtral /
/// CSM zero-dep embed pattern; the runtime tokenizer is self-implemented in
/// `crates/vokra-models/src/cosyvoice2/text_encoder.rs`).
pub(crate) struct TokenizerFiles<'a> {
    /// Raw `vocab.json` bytes (Qwen2 byte-level BPE vocabulary).
    pub(crate) vocab_json: &'a [u8],
    /// Raw `merges.txt` bytes (BPE merge ranks, one `LEFT RIGHT` per line).
    pub(crate) merges_txt: &'a [u8],
}

/// Converts a CosyVoice2 safetensors buffer (no config / tokenizer side-car)
/// — see [`convert_with_config_and_tokenizer`].
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, CosyVoice2Report), ConvertError> {
    convert_with_config(bytes, None)
}

/// Converts a CosyVoice2 safetensors buffer with no tokenizer side-car — see
/// [`convert_with_config_and_tokenizer`].
pub(crate) fn convert_with_config(
    bytes: Vec<u8>,
    config_json: Option<&[u8]>,
) -> Result<(GgufBuilder, CosyVoice2Report), ConvertError> {
    convert_with_config_and_tokenizer(bytes, config_json, None)
}

/// Converts a CosyVoice2 safetensors buffer into a populated GGUF builder
/// plus a report of what was written vs. skipped.
///
/// Every tensor is written verbatim (bytes, dtype and shape preserved); no
/// FP16 → FP32 widening. `config_json` is the upstream HF `config.json`
/// (Qwen2 schema) supplying the head split + RoPE/eps/n_ctx values that
/// tensor shapes cannot determine; without it those keys are left `0` /
/// unwritten with a loud note and the runtime will refuse the LLM bind.
///
/// `tokenizer` are the raw Qwen2 `vocab.json` + `merges.txt` bytes (T06);
/// when present and non-empty they are embedded verbatim as the
/// `vokra.cosyvoice2.tokenizer.*` U8 chunks. When absent (or empty) the
/// runtime text path (`CosyVoice2Tts::encode`) fails loudly until a
/// tokenizer-carrying GGUF is converted (FR-EX-08 — never a silent stub).
pub(crate) fn convert_with_config_and_tokenizer(
    bytes: Vec<u8>,
    config_json: Option<&[u8]>,
    tokenizer: Option<TokenizerFiles<'_>>,
) -> Result<(GgufBuilder, CosyVoice2Report), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;
    let mut report = CosyVoice2Report::default();

    let shape = derive_shape_hparams(&st)?;
    let config = config_json.map(parse_hf_config).transpose()?;

    // Cross-check config vs shapes; resolve the final hparam set.
    let derived = match (&shape, &config) {
        (Some(s), Some(c)) => {
            cross_check(s, c)?;
            Some(DerivedHparams {
                n_head: c.num_attention_heads,
                n_head_kv: c.num_key_value_heads,
                n_ctx: c.max_position_embeddings,
                ..*s
            })
        }
        (Some(s), None) => {
            report.notes.push(format!(
                "attention head split (n_head / n_head_kv) is not derivable from tensor \
                 shapes (q_out == hidden leaves head_dim free) — pass `--config \
                 <upstream config.json>` to write it; until then \
                 `{KEY_N_HEAD}` stays 0 and the runtime refuses the LLM bind"
            ));
            Some(*s)
        }
        (None, Some(_)) => {
            return Err(ConvertError::Parse(format!(
                "cosyvoice2: --config was passed but the safetensors buffer does not \
                 carry the LLM backbone (`{T_TOKEN_EMB}` missing) — nothing to \
                 cross-check the config against"
            )));
        }
        (None, None) => {
            report.notes.push(format!(
                "`{T_TOKEN_EMB}` not found — no LLM backbone hparams derived; the \
                 numeric hparams are 0-placeholders and the runtime rejects the LLM \
                 bind at load"
            ));
            None
        }
    };
    report.derived = derived;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b, derived.as_ref(), config.as_ref());
    embed_tokenizer(&mut b, tokenizer, &mut report);

    for t in st.tensors() {
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    Ok((b, report))
}

/// The `--config` subset the converter consumes — the HF Qwen2
/// `config.json` schema fields the runtime needs and shapes cannot supply
/// (plus the redundant shape fields used purely for cross-checking).
#[derive(Debug, Clone, Copy)]
struct HfQwen2Config {
    num_attention_heads: u32,
    num_key_value_heads: u32,
    rope_theta: f32,
    rms_norm_eps: f32,
    max_position_embeddings: u32,
    hidden_size: Option<u32>,
    num_hidden_layers: Option<u32>,
    intermediate_size: Option<u32>,
    vocab_size: Option<u32>,
}

/// Parses the upstream HF `config.json`. The five load-bearing fields are
/// required — a config missing them is the wrong file, and inventing a
/// default here would bake a silent wrong value into the GGUF (FR-EX-08).
fn parse_hf_config(bytes: &[u8]) -> Result<HfQwen2Config, ConvertError> {
    let root = json::parse(bytes)
        .map_err(|e| ConvertError::Parse(format!("cosyvoice2 --config: not valid JSON: {e}")))?;
    let req_u32 = |key: &str| -> Result<u32, ConvertError> {
        json_u32(&root, key)?.ok_or_else(|| {
            ConvertError::Parse(format!(
                "cosyvoice2 --config: `{key}` missing — pass the upstream HF config.json \
                 (Qwen2 schema)"
            ))
        })
    };
    let req_f32 = |key: &str| -> Result<f32, ConvertError> {
        json_f32(&root, key)?.ok_or_else(|| {
            ConvertError::Parse(format!(
                "cosyvoice2 --config: `{key}` missing — pass the upstream HF config.json \
                 (Qwen2 schema)"
            ))
        })
    };
    Ok(HfQwen2Config {
        num_attention_heads: req_u32("num_attention_heads")?,
        num_key_value_heads: req_u32("num_key_value_heads")?,
        rope_theta: req_f32("rope_theta")?,
        rms_norm_eps: req_f32("rms_norm_eps")?,
        max_position_embeddings: req_u32("max_position_embeddings")?,
        hidden_size: json_u32(&root, "hidden_size")?,
        num_hidden_layers: json_u32(&root, "num_hidden_layers")?,
        intermediate_size: json_u32(&root, "intermediate_size")?,
        vocab_size: json_u32(&root, "vocab_size")?,
    })
}

/// Reads an optional non-negative integer field; present-with-wrong-type is
/// a loud error, absent is `None`.
fn json_u32(root: &JsonValue, key: &str) -> Result<Option<u32>, ConvertError> {
    match root.get(key) {
        None => Ok(None),
        Some(v) => v
            .as_u64()
            .and_then(|x| u32::try_from(x).ok())
            .map(Some)
            .ok_or_else(|| {
                ConvertError::Parse(format!(
                    "cosyvoice2 --config: `{key}` is not a u32-range integer: {v:?}"
                ))
            }),
    }
}

/// Reads an optional numeric field as f32 (JSON int or float); present-with-
/// wrong-type is a loud error, absent is `None`.
fn json_f32(root: &JsonValue, key: &str) -> Result<Option<f32>, ConvertError> {
    match root.get(key) {
        None => Ok(None),
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        Some(JsonValue::Int(i)) => Ok(Some(*i as f32)),
        #[allow(clippy::cast_possible_truncation)]
        Some(JsonValue::Float(f)) => Ok(Some(*f as f32)),
        Some(other) => Err(ConvertError::Parse(format!(
            "cosyvoice2 --config: `{key}` is not a number: {other:?}"
        ))),
    }
}

/// Derives the unambiguous shape hparams from the tensor set, or `None`
/// when the buffer does not carry the LLM backbone (scaffold inputs keep
/// converting with 0-placeholders + a note).
fn derive_shape_hparams(st: &SafetensorsFile) -> Result<Option<DerivedHparams>, ConvertError> {
    let Some(emb) = find(st, T_TOKEN_EMB) else {
        return Ok(None);
    };
    let [vocab, hidden] = rank2(emb)?;

    // Contiguous layer count + no-gap validation.
    let mut n_layer = 0usize;
    while find(
        st,
        &format!("{}input_layernorm.weight", layer_prefix(n_layer)),
    )
    .is_some()
    {
        n_layer += 1;
    }
    if n_layer == 0 {
        return Err(ConvertError::Parse(format!(
            "cosyvoice2: `{T_TOKEN_EMB}` present but no \
             `llm.model.model.layers.0.input_layernorm.weight` — not a CosyVoice2 \
             LLM checkpoint layout"
        )));
    }
    for t in st.tensors() {
        if let Some(idx) = parse_layer_index(&t.name) {
            if idx >= n_layer {
                return Err(ConvertError::Parse(format!(
                    "cosyvoice2: `{}` implies layer {idx} but the contiguous block \
                     count is {n_layer} (a gap in the layer indices means a broken \
                     export)",
                    t.name
                )));
            }
        }
    }

    // FFN width + per-layer projection / bias consistency.
    let gate0 = require(st, &format!("{}mlp.gate_proj.weight", layer_prefix(0)))?;
    let [ffn, gate_in] = rank2(gate0)?;
    check_eq(&gate0.name, "in width", gate_in, hidden)?;
    let q0 = require(st, &format!("{}self_attn.q_proj.weight", layer_prefix(0)))?;
    let [q_out, q_in] = rank2(q0)?;
    check_eq(&q0.name, "in width", q_in, hidden)?;
    check_eq(
        &q0.name,
        "out width (q_out must equal hidden)",
        q_out,
        hidden,
    )?;
    let k0 = require(st, &format!("{}self_attn.k_proj.weight", layer_prefix(0)))?;
    let [kv_out, k_in] = rank2(k0)?;
    check_eq(&k0.name, "in width", k_in, hidden)?;

    let mut has_attn_bias: Option<bool> = None;
    for i in 0..n_layer {
        let p = layer_prefix(i);
        let mut present = 0usize;
        for (proj, want_out) in [("q", hidden), ("k", kv_out), ("v", kv_out)] {
            let name = format!("{p}self_attn.{proj}_proj.bias");
            if let Some(info) = find(st, &name) {
                present += 1;
                let dims: Vec<u64> = info.shape.clone();
                if dims != [want_out] {
                    return Err(ConvertError::Parse(format!(
                        "cosyvoice2: `{name}` shape {dims:?} != expected [{want_out}]"
                    )));
                }
            }
        }
        let layer_has = match present {
            0 => false,
            3 => true,
            n => {
                return Err(ConvertError::Parse(format!(
                    "cosyvoice2: layer {i} ships {n}/3 Q/K/V bias tensors — a partial \
                     bias set means a broken export (all three or none)"
                )));
            }
        };
        match has_attn_bias {
            None => has_attn_bias = Some(layer_has),
            Some(expected) if expected != layer_has => {
                return Err(ConvertError::Parse(format!(
                    "cosyvoice2: layer {i} bias presence ({layer_has}) differs from \
                     layer 0 ({expected}) — mixed-bias checkpoints are a broken export"
                )));
            }
            Some(_) => {}
        }
    }

    Ok(Some(DerivedHparams {
        vocab_size: u32::try_from(vocab).map_err(|_| overflow(T_TOKEN_EMB))?,
        hidden_dim: u32::try_from(hidden).map_err(|_| overflow(T_TOKEN_EMB))?,
        n_layer: u32::try_from(n_layer).map_err(|_| overflow("n_layer"))?,
        ffn_dim: u32::try_from(ffn).map_err(|_| overflow(&gate0.name))?,
        n_head: 0,
        n_head_kv: 0,
        n_ctx: 0,
        has_attn_bias: has_attn_bias.unwrap_or(false),
    }))
}

/// Hard cross-checks between the `--config` values and the tensor shapes —
/// a config from a different model must not produce a silently-wrong GGUF.
fn cross_check(s: &DerivedHparams, c: &HfQwen2Config) -> Result<(), ConvertError> {
    let pairs = [
        ("hidden_size", c.hidden_size, s.hidden_dim),
        ("num_hidden_layers", c.num_hidden_layers, s.n_layer),
        ("intermediate_size", c.intermediate_size, s.ffn_dim),
        ("vocab_size", c.vocab_size, s.vocab_size),
    ];
    for (key, got, want) in pairs {
        if let Some(got) = got {
            if got != want {
                return Err(ConvertError::Parse(format!(
                    "cosyvoice2: --config `{key}` = {got} disagrees with the tensor \
                     shapes ({want}) — wrong config.json for this checkpoint"
                )));
            }
        }
    }
    // GQA algebra: hidden = n_head × head_dim; the K projection width must
    // be n_head_kv × head_dim; and the head split must divide evenly.
    let n_head = c.num_attention_heads;
    let n_kv = c.num_key_value_heads;
    if n_head == 0 || n_kv == 0 || n_head % n_kv != 0 || s.hidden_dim % n_head != 0 {
        return Err(ConvertError::Parse(format!(
            "cosyvoice2: --config head split (num_attention_heads = {n_head}, \
             num_key_value_heads = {n_kv}) is not GQA-well-formed for hidden_dim {}",
            s.hidden_dim
        )));
    }
    Ok(())
}

fn find<'a>(st: &'a SafetensorsFile, name: &str) -> Option<&'a SafeTensorInfo> {
    st.tensors().iter().find(|t| t.name == name)
}

fn require<'a>(st: &'a SafetensorsFile, name: &str) -> Result<&'a SafeTensorInfo, ConvertError> {
    find(st, name).ok_or_else(|| {
        ConvertError::Parse(format!(
            "cosyvoice2: `{name}` missing — not a CosyVoice2 LLM checkpoint layout"
        ))
    })
}

fn rank2(info: &SafeTensorInfo) -> Result<[u64; 2], ConvertError> {
    match info.shape[..] {
        [a, b] => Ok([a, b]),
        ref other => Err(ConvertError::Parse(format!(
            "cosyvoice2: `{}` rank {} (shape {other:?}) where a rank-2 matrix was \
             expected",
            info.name,
            other.len(),
        ))),
    }
}

fn check_eq(name: &str, what: &str, got: u64, want: u64) -> Result<(), ConvertError> {
    if got == want {
        Ok(())
    } else {
        Err(ConvertError::Parse(format!(
            "cosyvoice2: `{name}` {what} = {got}, expected {want}"
        )))
    }
}

fn overflow(name: &str) -> ConvertError {
    ConvertError::Parse(format!("cosyvoice2: `{name}` dimension exceeds u32 range"))
}

/// Extracts `{i}` from `llm.model.model.layers.{i}.<rest>`, if the name is
/// a layer tensor.
fn parse_layer_index(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("llm.model.model.layers.")?;
    let (idx, _) = rest.split_once('.')?;
    idx.parse().ok()
}

/// Writes the `vokra.cosyvoice2.*` hparam chunk group.
///
/// The LLM arch hparams carry the shape/config-derived values when
/// available; anything unknown stays a `0` placeholder that the runtime
/// rejects at first use (loud fail rather than a silent zero-shape
/// forward). The Flow Matching / streaming keys stay `0` / `"linear"` —
/// they belong to the upstream `flow.pt` pipeline, which this converter
/// does not consume yet.
fn write_hparams(
    b: &mut GgufBuilder,
    derived: Option<&DerivedHparams>,
    cfg: Option<&HfQwen2Config>,
) {
    b.add_u32(KEY_SAMPLE_RATE, COSYVOICE2_SAMPLE_RATE);
    let d = derived.copied().unwrap_or_default();
    b.add_u32(KEY_VOCAB_SIZE, d.vocab_size);
    b.add_u32(KEY_HIDDEN_DIM, d.hidden_dim);
    b.add_u32(KEY_N_LAYER, d.n_layer);
    b.add_u32(KEY_N_HEAD, d.n_head);
    b.add_u32(KEY_FFN_DIM, d.ffn_dim);
    if let Some(c) = cfg {
        // Only written when the upstream config supplied them — the
        // runtime has documented fallbacks for absent keys, and a made-up
        // value here would silently override them.
        b.add_u32(KEY_N_HEAD_KV, c.num_key_value_heads);
        b.add_f32(KEY_ROPE_BASE, c.rope_theta);
        b.add_f32(KEY_RMS_NORM_EPS, c.rms_norm_eps);
        b.add_u32(KEY_N_CTX, c.max_position_embeddings);
    }
    b.add_u32(KEY_FLOW_NFE, 0);
    // The schedule tag has no meaningful `0`-placeholder — a missing
    // schedule tag is what the runtime error is written to catch. We
    // write `"linear"` (the M3-05 default schedule) until the flow.pt
    // pipeline lands in the converter.
    b.add_string(KEY_FLOW_SCHEDULE, "linear");
    b.add_u32(KEY_MIMI_N_CODEBOOKS, MIMI_N_CODEBOOKS);
    b.add_u32(KEY_MIMI_CODEBOOK_SIZE, MIMI_CODEBOOK_SIZE);
    b.add_u32(KEY_MIMI_D_MODEL, MIMI_D_MODEL);
    b.add_u32(KEY_STREAMING_CHUNK_SIZE, 0);
    b.add_u32(KEY_STREAMING_CHUNK_HOP, 0);
}

/// Embeds the Qwen2 text tokenizer (`vocab.json` + `merges.txt`) as the two
/// `vokra.cosyvoice2.tokenizer.*` U8 chunks (T06), or records a loud note
/// when no (usable) tokenizer was supplied.
///
/// Both files are embedded together or not at all: a byte-level BPE needs the
/// vocabulary *and* the merge ranks, so a half-supplied pair is treated as
/// "no tokenizer" (noted) rather than written as a silently-unusable chunk.
fn embed_tokenizer(
    b: &mut GgufBuilder,
    tokenizer: Option<TokenizerFiles<'_>>,
    report: &mut CosyVoice2Report,
) {
    match tokenizer {
        Some(tok) if !tok.vocab_json.is_empty() && !tok.merges_txt.is_empty() => {
            // U8 arrays, bytes verbatim (the M2-06 Whisper / M3-10 Voxtral /
            // M4-05 CSM zero-dep embed pattern).
            b.add_metadata(
                KEY_TOKENIZER_VOCAB,
                GgufMetadataValue::Array(GgufArray {
                    element_type: GgufValueType::U8,
                    values: tok
                        .vocab_json
                        .iter()
                        .map(|&x| GgufMetadataValue::U8(x))
                        .collect(),
                }),
            );
            b.add_metadata(
                KEY_TOKENIZER_MERGES,
                GgufMetadataValue::Array(GgufArray {
                    element_type: GgufValueType::U8,
                    values: tok
                        .merges_txt
                        .iter()
                        .map(|&x| GgufMetadataValue::U8(x))
                        .collect(),
                }),
            );
            report.tokenizer_embedded = true;
        }
        Some(_) => {
            report.notes.push(
                "tokenizer side-car present but vocab.json or merges.txt was empty — \
                 vokra.cosyvoice2.tokenizer.* not embedded"
                    .to_owned(),
            );
        }
        None => {
            report.notes.push(
                "no Qwen2 tokenizer side-car (vocab.json + merges.txt) supplied — \
                 vokra.cosyvoice2.tokenizer.* not embedded; the runtime text path \
                 (CosyVoice2Tts::encode) fails loudly until a tokenizer-carrying GGUF \
                 is converted"
                    .to_owned(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    /// Builds a minimal safetensors buffer with one F32 tensor. Payload is
    /// deliberately trivial (all-zero) — only the header parsing and the
    /// verbatim byte-copy path are exercised.
    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // A single F32 tensor of shape [2, 3] = 6 elements = 24 bytes.
        let header = r#"{"llm.wte":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 24]);
        out
    }

    /// Builds a safetensors buffer with the full (tiny) Qwen2-shaped LLM
    /// backbone: vocab 16, hidden 8, 2 layers, ffn 16, kv_out 4. With
    /// `with_bias`, every layer ships the three Q/K/V bias tensors.
    fn backbone_safetensors(with_bias: bool) -> Vec<u8> {
        let (vocab, d, ffn, kv) = (16u64, 8u64, 16u64, 4u64);
        let mut entries: Vec<(String, Vec<u64>)> = vec![(
            "llm.model.model.embed_tokens.weight".to_owned(),
            vec![vocab, d],
        )];
        for i in 0..2 {
            let p = format!("llm.model.model.layers.{i}.");
            entries.push((format!("{p}input_layernorm.weight"), vec![d]));
            entries.push((format!("{p}self_attn.q_proj.weight"), vec![d, d]));
            entries.push((format!("{p}self_attn.k_proj.weight"), vec![kv, d]));
            entries.push((format!("{p}self_attn.v_proj.weight"), vec![kv, d]));
            if with_bias {
                entries.push((format!("{p}self_attn.q_proj.bias"), vec![d]));
                entries.push((format!("{p}self_attn.k_proj.bias"), vec![kv]));
                entries.push((format!("{p}self_attn.v_proj.bias"), vec![kv]));
            }
            entries.push((format!("{p}self_attn.o_proj.weight"), vec![d, d]));
            entries.push((format!("{p}post_attention_layernorm.weight"), vec![d]));
            entries.push((format!("{p}mlp.gate_proj.weight"), vec![ffn, d]));
            entries.push((format!("{p}mlp.up_proj.weight"), vec![ffn, d]));
            entries.push((format!("{p}mlp.down_proj.weight"), vec![d, ffn]));
        }
        entries.push(("llm.model.model.norm.weight".to_owned(), vec![d]));
        build_safetensors(&entries)
    }

    /// Serializes `entries` (name, shape) as an all-zero F32 safetensors
    /// buffer.
    fn build_safetensors(entries: &[(String, Vec<u64>)]) -> Vec<u8> {
        let mut header = String::from("{");
        let mut offset = 0u64;
        for (i, (name, shape)) in entries.iter().enumerate() {
            let n: u64 = shape.iter().product();
            let end = offset + n * 4;
            if i > 0 {
                header.push(',');
            }
            let dims = shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                r#""{name}":{{"dtype":"F32","shape":[{dims}],"data_offsets":[{offset},{end}]}}"#
            ));
            offset = end;
        }
        header.push('}');
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.resize(out.len() + offset as usize, 0u8);
        out
    }

    /// The Qwen2-style config.json matching `backbone_safetensors` shapes
    /// (head split 2/1, head_dim 4).
    const TINY_CONFIG: &str = r#"{
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 16,
        "vocab_size": 16,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-06,
        "max_position_embeddings": 32768
    }"#;

    fn get_u32(file: &GgufFile, key: &str) -> u32 {
        match file.get(key) {
            Some(GgufMetadataValue::U32(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    fn get_f32(file: &GgufFile, key: &str) -> f32 {
        match file.get(key) {
            Some(GgufMetadataValue::F32(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    #[test]
    fn round_trip_carries_arch_and_cosyvoice2_chunk_group() {
        // A scaffold buffer (no backbone tensors) keeps converting with
        // 0-placeholders + a note — the chunk group must round-trip so the
        // runtime constants read the same values back.
        let bytes = minimal_safetensors_one_f32();
        let (builder, report) = convert(bytes).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert!(report.derived.is_none());
        assert!(
            report.notes.iter().any(|n| n.contains("not found")),
            "scaffold path must note the missing backbone: {:?}",
            report.notes
        );

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");

        // Arch / name.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );

        // Sample rate: model-card invariant.
        assert_eq!(get_u32(&file, KEY_SAMPLE_RATE), COSYVOICE2_SAMPLE_RATE);

        // Mimi shape: canonical Kyutai defaults.
        for (key, expected) in [
            (KEY_MIMI_N_CODEBOOKS, MIMI_N_CODEBOOKS),
            (KEY_MIMI_CODEBOOK_SIZE, MIMI_CODEBOOK_SIZE),
            (KEY_MIMI_D_MODEL, MIMI_D_MODEL),
        ] {
            assert_eq!(get_u32(&file, key), expected, "{key}");
        }

        // Placeholder hparams: `0` (no backbone to derive from).
        for key in [
            KEY_VOCAB_SIZE,
            KEY_HIDDEN_DIM,
            KEY_N_LAYER,
            KEY_N_HEAD,
            KEY_FFN_DIM,
            KEY_FLOW_NFE,
            KEY_STREAMING_CHUNK_SIZE,
            KEY_STREAMING_CHUNK_HOP,
        ] {
            assert_eq!(get_u32(&file, key), 0, "{key}");
        }
        // The config-only keys are unwritten without --config.
        for key in [KEY_N_HEAD_KV, KEY_ROPE_BASE, KEY_RMS_NORM_EPS, KEY_N_CTX] {
            assert!(file.get(key).is_none(), "{key} must be absent");
        }

        // Schedule tag: `linear` default.
        assert_eq!(
            file.get(KEY_FLOW_SCHEDULE).and_then(|v| v.as_str()),
            Some("linear")
        );
    }

    #[test]
    fn shape_hparams_derive_without_config() {
        let (builder, report) = convert(backbone_safetensors(true)).expect("convert");
        let d = report.derived.expect("backbone present → derived");
        assert_eq!(
            d,
            DerivedHparams {
                vocab_size: 16,
                hidden_dim: 8,
                n_layer: 2,
                ffn_dim: 16,
                n_head: 0,
                n_head_kv: 0,
                n_ctx: 0,
                has_attn_bias: true,
            }
        );
        assert!(
            report.notes.iter().any(|n| n.contains("head split")),
            "must warn that the head split needs --config: {:?}",
            report.notes
        );
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(get_u32(&file, KEY_VOCAB_SIZE), 16);
        assert_eq!(get_u32(&file, KEY_HIDDEN_DIM), 8);
        assert_eq!(get_u32(&file, KEY_N_LAYER), 2);
        assert_eq!(get_u32(&file, KEY_FFN_DIM), 16);
        assert_eq!(get_u32(&file, KEY_N_HEAD), 0, "not shape-derivable");
        assert!(file.get(KEY_N_HEAD_KV).is_none());
        // Every tensor rides along verbatim: embed + 2 layers × (9 weights
        // + 3 biases) + final norm.
        assert_eq!(report.written, 1 + 2 * (9 + 3) + 1);
    }

    #[test]
    fn config_supplies_head_split_and_rope_group() {
        let (builder, report) =
            convert_with_config(backbone_safetensors(true), Some(TINY_CONFIG.as_bytes()))
                .expect("convert");
        let d = report.derived.expect("derived");
        assert_eq!(d.n_head, 2);
        assert_eq!(d.n_head_kv, 1);
        assert_eq!(d.n_ctx, 32_768);
        assert!(d.has_attn_bias);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(get_u32(&file, KEY_N_HEAD), 2);
        assert_eq!(get_u32(&file, KEY_N_HEAD_KV), 1);
        assert_eq!(get_u32(&file, KEY_N_CTX), 32_768);
        assert!((get_f32(&file, KEY_ROPE_BASE) - 1_000_000.0).abs() < 1e-1);
        assert!((get_f32(&file, KEY_RMS_NORM_EPS) - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn biasless_backbone_derives_has_attn_bias_false() {
        let (_, report) =
            convert_with_config(backbone_safetensors(false), Some(TINY_CONFIG.as_bytes()))
                .expect("convert");
        assert!(!report.derived.expect("derived").has_attn_bias);
    }

    #[test]
    fn config_shape_mismatch_fails_loudly() {
        let bad = TINY_CONFIG.replace("\"hidden_size\": 8", "\"hidden_size\": 896");
        let err = convert_with_config(backbone_safetensors(true), Some(bad.as_bytes()))
            .expect_err("wrong config must fail");
        assert!(
            err.to_string().contains("hidden_size"),
            "must name the field: {err}"
        );
    }

    #[test]
    fn config_bad_gqa_split_fails_loudly() {
        // 3 kv heads do not divide 2 query heads.
        let bad = TINY_CONFIG.replace("\"num_key_value_heads\": 1", "\"num_key_value_heads\": 3");
        let err = convert_with_config(backbone_safetensors(true), Some(bad.as_bytes()))
            .expect_err("bad GQA split must fail");
        assert!(err.to_string().contains("GQA"), "{err}");
    }

    #[test]
    fn config_missing_required_field_fails_loudly() {
        let bad = TINY_CONFIG.replace("\"num_attention_heads\": 2,", "");
        let err = convert_with_config(backbone_safetensors(true), Some(bad.as_bytes()))
            .expect_err("missing head count must fail");
        assert!(err.to_string().contains("num_attention_heads"), "{err}");
    }

    #[test]
    fn config_without_backbone_tensors_fails_loudly() {
        let err = convert_with_config(minimal_safetensors_one_f32(), Some(TINY_CONFIG.as_bytes()))
            .expect_err("nothing to cross-check");
        assert!(err.to_string().contains("does not carry"), "{err}");
    }

    #[test]
    fn partial_bias_export_fails_loudly() {
        // Rebuild the biased layout but drop layer 1's k/v biases.
        let (vocab, d, ffn, kv) = (16u64, 8u64, 16u64, 4u64);
        let mut entries: Vec<(String, Vec<u64>)> = vec![(
            "llm.model.model.embed_tokens.weight".to_owned(),
            vec![vocab, d],
        )];
        for i in 0..2 {
            let p = format!("llm.model.model.layers.{i}.");
            entries.push((format!("{p}input_layernorm.weight"), vec![d]));
            entries.push((format!("{p}self_attn.q_proj.weight"), vec![d, d]));
            entries.push((format!("{p}self_attn.k_proj.weight"), vec![kv, d]));
            entries.push((format!("{p}self_attn.v_proj.weight"), vec![kv, d]));
            entries.push((format!("{p}self_attn.q_proj.bias"), vec![d]));
            if i == 0 {
                entries.push((format!("{p}self_attn.k_proj.bias"), vec![kv]));
                entries.push((format!("{p}self_attn.v_proj.bias"), vec![kv]));
            }
            entries.push((format!("{p}self_attn.o_proj.weight"), vec![d, d]));
            entries.push((format!("{p}post_attention_layernorm.weight"), vec![d]));
            entries.push((format!("{p}mlp.gate_proj.weight"), vec![ffn, d]));
            entries.push((format!("{p}mlp.up_proj.weight"), vec![ffn, d]));
            entries.push((format!("{p}mlp.down_proj.weight"), vec![d, ffn]));
        }
        entries.push(("llm.model.model.norm.weight".to_owned(), vec![d]));
        let err = convert(build_safetensors(&entries)).expect_err("partial bias set");
        assert!(err.to_string().contains("bias"), "{err}");
    }

    #[test]
    fn gapped_layer_indices_fail_loudly() {
        let (vocab, d) = (16u64, 8u64);
        // Layer 0 complete-ish, then a layers.5 stray.
        let entries: Vec<(String, Vec<u64>)> = vec![
            (
                "llm.model.model.embed_tokens.weight".to_owned(),
                vec![vocab, d],
            ),
            (
                "llm.model.model.layers.0.input_layernorm.weight".to_owned(),
                vec![d],
            ),
            (
                "llm.model.model.layers.0.mlp.gate_proj.weight".to_owned(),
                vec![16, d],
            ),
            (
                "llm.model.model.layers.0.self_attn.q_proj.weight".to_owned(),
                vec![d, d],
            ),
            (
                "llm.model.model.layers.0.self_attn.k_proj.weight".to_owned(),
                vec![4, d],
            ),
            (
                "llm.model.model.layers.5.input_layernorm.weight".to_owned(),
                vec![d],
            ),
        ];
        let err = convert(build_safetensors(&entries)).expect_err("gap must fail");
        assert!(err.to_string().contains("layer 5"), "{err}");
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // Hard-coded sanity: the runtime's EXPECTED_ARCH is `cosyvoice2`;
        // this file's ARCH constant must be identical. A drift is caught
        // here rather than at load time.
        assert_eq!(ARCH, "cosyvoice2");
    }

    #[test]
    fn llm_key_strings_match_runtime_constants() {
        // The runtime duplicates these strings in
        // `vokra-models/src/cosyvoice2/llm.rs` (two-crate constant rule);
        // pin them here so a drift is a test failure, not a load-time
        // mystery.
        assert_eq!(KEY_N_HEAD_KV, "vokra.cosyvoice2.arch.n_head_kv");
        assert_eq!(KEY_ROPE_BASE, "vokra.cosyvoice2.arch.rope_base");
        assert_eq!(KEY_RMS_NORM_EPS, "vokra.cosyvoice2.arch.rms_norm_eps");
        assert_eq!(KEY_N_CTX, "vokra.cosyvoice2.arch.n_ctx");
    }

    #[test]
    fn tokenizer_key_strings_match_runtime_constants() {
        // Mirror of the runtime's KEY_TOKENIZER_VOCAB / KEY_TOKENIZER_MERGES
        // in `vokra-models/src/cosyvoice2/text_encoder.rs` (two-crate rule).
        assert_eq!(KEY_TOKENIZER_VOCAB, "vokra.cosyvoice2.tokenizer.vocab");
        assert_eq!(KEY_TOKENIZER_MERGES, "vokra.cosyvoice2.tokenizer.merges");
    }

    /// Reads a `U8` GGUF array metadata value back into bytes (test-side
    /// mirror of the runtime reader).
    fn read_u8_array(file: &GgufFile, key: &str) -> Vec<u8> {
        match file.get(key) {
            Some(GgufMetadataValue::Array(arr)) => arr
                .values
                .iter()
                .map(|v| match v {
                    GgufMetadataValue::U8(x) => *x,
                    other => panic!("{key}: non-U8 element {other:?}"),
                })
                .collect(),
            other => panic!("{key}: expected U8 array, got {other:?}"),
        }
    }

    #[test]
    fn tokenizer_files_are_embedded_verbatim() {
        // A scaffold buffer converts fine; the tokenizer chunks ride along
        // verbatim so the runtime reads the exact upstream bytes back.
        let vocab = br#"{"a":0,"b":1,"ab":2}"#.to_vec();
        let merges = b"#version: 0.2\na b\n".to_vec();
        let (builder, report) = convert_with_config_and_tokenizer(
            minimal_safetensors_one_f32(),
            None,
            Some(TokenizerFiles {
                vocab_json: &vocab,
                merges_txt: &merges,
            }),
        )
        .expect("convert");
        assert!(report.tokenizer_embedded, "tokenizer must be embedded");

        let file = GgufFile::parse(builder.to_bytes().expect("serialize")).expect("parse");
        assert_eq!(read_u8_array(&file, KEY_TOKENIZER_VOCAB), vocab);
        assert_eq!(read_u8_array(&file, KEY_TOKENIZER_MERGES), merges);
    }

    #[test]
    fn no_tokenizer_side_car_is_noted_and_not_embedded() {
        let (builder, report) = convert(minimal_safetensors_one_f32()).expect("convert");
        assert!(!report.tokenizer_embedded);
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("no Qwen2 tokenizer")),
            "missing tokenizer must be a loud note: {:?}",
            report.notes
        );
        let file = GgufFile::parse(builder.to_bytes().expect("serialize")).expect("parse");
        assert!(file.get(KEY_TOKENIZER_VOCAB).is_none());
        assert!(file.get(KEY_TOKENIZER_MERGES).is_none());
    }

    #[test]
    fn half_supplied_tokenizer_is_not_embedded() {
        // A byte-level BPE needs both files; an empty merges half is treated
        // as "no tokenizer" (noted), never written as an unusable chunk.
        let vocab = br#"{"a":0}"#.to_vec();
        let (builder, report) = convert_with_config_and_tokenizer(
            minimal_safetensors_one_f32(),
            None,
            Some(TokenizerFiles {
                vocab_json: &vocab,
                merges_txt: b"",
            }),
        )
        .expect("convert");
        assert!(!report.tokenizer_embedded);
        let file = GgufFile::parse(builder.to_bytes().expect("serialize")).expect("parse");
        assert!(file.get(KEY_TOKENIZER_VOCAB).is_none());
    }
}
