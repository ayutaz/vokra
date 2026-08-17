#![allow(clippy::doc_lazy_continuation)]
//! **CT-Transformer punctuation restoration** (`funasr/ct-punc`,
//! **apache-2.0**) — safetensors → GGUF conversion (Wave D 2026-08-15,
//! **brand-new `punctuation` category**).
//!
//! # Why this exists
//!
//! Vokra's ASR fleet (Whisper / Paraformer-lineage / Parakeet / Canary /
//! SenseVoice / …) emits raw token streams. Several of those emit *no*
//! punctuation at all, which makes the transcript technically correct and
//! practically unreadable. CT-Punc is the post-processing stage that turns
//! `我们今天讨论三个议题首先是产品发布` into
//! `我们今天讨论三个议题，首先是产品发布。` — paired with the ITN
//! (inverse text normalisation) stage it is what makes an ASR transcript
//! shippable. Before this converter Vokra had **no** punctuation-restoration
//! model of any kind, so [`CATEGORY`] = `"punctuation"` is the first of its
//! kind in the tree.
//!
//! # Primary sources (all fetched 2026-08-15, transcribed — not recalled)
//!
//! - HF release: <https://huggingface.co/funasr/ct-punc> — the README
//!   front-matter reads `license: apache-2.0`, and the HuggingFace model API
//!   (`https://huggingface.co/api/models/funasr/ct-punc`) reports
//!   `cardData.license = "apache-2.0"` / `tags: [… "license:apache-2.0" …]`.
//!   The repo ships `config.yaml`, `configuration.json`, `model.pt`
//!   (1 125 507 622 bytes per the `x-linked-size` response header) and
//!   `tokens.json`.
//! - Upstream toolkit: <https://github.com/modelscope/FunASR> — repo
//!   `LICENSE` is verbatim `MIT License / Copyright (c) 2025 FunASR`.
//! - Reference implementation (the source this forward is transcribed from):
//!   `funasr/models/ct_transformer/model.py` (`class CTTransformer`, header
//!   `MIT License (https://opensource.org/licenses/MIT)`),
//!   `funasr/models/sanm/encoder.py` (`SANMEncoder` / `EncoderLayerSANM`),
//!   `funasr/models/sanm/attention.py` (`MultiHeadedAttentionSANM`),
//!   `funasr/models/transformer/positionwise_feed_forward.py`,
//!   `funasr/models/transformer/embedding.py` (`SinusoidalPositionEncoder`),
//!   `funasr/models/transformer/layer_norm.py` (`LayerNorm`, `eps=1e-12`).
//! - Paper: Chen et al. 2020, *"Controllable Time-delay Transformer for
//!   Real-time Punctuation Prediction and Disfluency Detection"*
//!   (<https://arxiv.org/pdf/2003.01309.pdf>, cited in the upstream
//!   `CTTransformer` docstring).
//! - ModelScope mirror id (from the repo's own `configuration.json`
//!   `model_name_in_hub.ms`): `iic/punc_ct-transformer_cn-en-common-vocab471067-large`
//!   — the `vocab471067` in that slug matches the 471 067-entry
//!   `tokens.json` this repo ships, so the HF `funasr/ct-punc` release **is**
//!   the `-large` cn-en variant, not the older `vocab272727` zh-only one.
//!
//! # Licence — apache-2.0 on THIS checkpoint (recorded honestly)
//!
//! FunASR's *code* is MIT, but individual FunASR *weight* releases do not
//! all share it: the sibling `sensevoicesmall` converter/binder in this tree
//! correctly fail-close to [`LicenseClass::Unknown`] because
//! `FunAudioLLM/SenseVoiceSmall` ships the bespoke
//! `github.com/modelscope/FunASR/blob/main/MODEL_LICENSE` ("FunASR Model
//! Open Source License Agreement", version 1.1, Alibaba Group) rather than
//! an SPDX id.
//!
//! **For `funasr/ct-punc` specifically that is not the case**: the model
//! card front-matter and the HF API both declare plain `apache-2.0`, and
//! the repo carries no `MODEL_LICENSE` sibling file (its file list is
//! `.gitattributes`, `README.md`, `config.yaml`, `configuration.json`,
//! `example/punc_example.txt`, `fig/struct.png`, `model.pt`, `tokens.json`).
//! So [`DEFAULT_LICENSE_SPDX`] = `"apache-2.0"` →
//! [`LicenseClass::Permissive`]. That is what the primary source says on
//! 2026-08-15; it is **not** a sign-off.
//!
//! `docs/license-audit.md` §3.1 sign-off column stays **BLANK** — CC never
//! signs a licence row (fail-closed, owner-only per memory
//! `[[feedback-license-signoff-primary-source]]`). Runtime binder land is
//! unblocked; *publish* is blocked until the owner signs.
//!
//! # Architecture (transcribed from the reference implementation)
//!
//! ```text
//! token ids                                   [T]
//!   -> embed: nn.Embedding(vocab_size, embed_unit)          [T, D]
//!   -> x * sqrt(output_size)          (SANMEncoder.forward line 1)
//!   -> + SinusoidalPositionEncoder()  (positions are 1-based!)
//!   -> encoders0.0            : EncoderLayerSANM             [T, D]
//!   -> encoders.0 .. .{N-2}   : EncoderLayerSANM x (num_blocks - 1)
//!   -> encoder.after_norm     : LayerNorm(D, eps=1e-12)
//!   -> decoder: nn.Linear(att_unit, punc_size)               [T, P]
//!   -> argmax over P => an index into `punc_list`
//! ```
//!
//! `EncoderLayerSANM` with `normalize_before=true`, `concat_after=false`,
//! `in_size == size` (pre-norm, both residuals live):
//!
//! ```text
//! x = x + self_attn(norm1(x))
//! x = x + w_2(relu(w_1(norm2(x))))
//! ```
//!
//! `MultiHeadedAttentionSANM.forward` — the piece that makes this **not** a
//! plain BERT block, and the reason [`ARCH`] must not alias `bert_base`:
//!
//! ```text
//! q, k, v   = split(linear_q_k_v(x), D)              # ONE fused 3D-wide proj
//! fsmn_mem  = depthwise_conv1d(pad(v), fsmn_block) + v   # FSMN memory branch
//! q        *= d_k ** -0.5
//! attn_out  = linear_out( softmax(q k^T) v )
//! return      attn_out + fsmn_mem                    # <- parallel memory add
//! ```
//!
//! The `fsmn_block` is `nn.Conv1d(D, D, kernel_size, groups=D, bias=False)`
//! (depthwise, no bias) with `ConstantPad1d((left, right), 0.0)` where
//! `left = (kernel_size - 1) // 2 + sanm_shfit` and
//! `right = kernel_size - 1 - left`.
//!
//! # Distinct arch tag from every text-encoder sibling
//!
//! [`ARCH`] = `"ct_punc"` is **deliberately distinct** from every sibling
//! Transformer-over-text arch tag in the tree:
//!
//! - `bert_base` — plain BERT (post-norm, learned absolute position table,
//!   separate `query`/`key`/`value` projections, **no FSMN memory branch**);
//! - `deberta_v2` / `deberta_v3` — disentangled relative position attention
//!   with `rel_embeddings` (a completely different score assembly);
//! - `sensevoicesmall` — also SAN-M, but a *speech* encoder over fbank
//!   frames with four per-task heads, not a per-token punctuation head;
//! - `fsmn-vad` — FSMN memory blocks too, but a feed-forward VAD stack with
//!   no self-attention at all and a 2-class frame output.
//!
//! Silently sharing an arch tag would let runtime dispatch mis-route a
//! CT-Punc checkpoint onto a wrong-topology loader (FR-EX-08 forbids the
//! silent shape misroute).
//!
//! # `vokra.ct_punc.*` chunk group — derived from the checkpoint, not guessed
//!
//! Every axis that *can* be read off the real tensor shapes **is**, so the
//! stamped GGUF describes the checkpoint in front of it rather than the
//! checkpoint the author had in mind:
//!
//! | key | source |
//! |---|---|
//! | `vocab_size` | `embed.weight` dim 0 |
//! | `embed_unit` | `embed.weight` dim 1 |
//! | `att_unit` | `decoder.weight` dim 1 |
//! | `linear_units` | `…feed_forward.w_1.weight` dim 0 |
//! | `num_blocks` | count of `encoder.encoders.{i}.norm1.weight` + 1 |
//! | `kernel_size` | `…self_attn.fsmn_block.weight` last dim |
//! | `punc_list` | [`DEFAULT_PUNC_LIST`], **cross-checked** against `decoder.weight` dim 0 |
//! | `attention_heads` | [`DEFAULT_ATTENTION_HEADS`] (a reshape param — not derivable from any shape), validated `att_unit % heads == 0` |
//! | `sanm_shift` | [`DEFAULT_SANM_SHIFT`] (a padding param — not derivable) |
//! | `sentence_end_id` | [`DEFAULT_SENTENCE_END_ID`], validated `< punc_list.len()` |
//! | `layer_norm_eps` | [`DEFAULT_LAYER_NORM_EPS`] (`torch.nn.LayerNorm` `eps=1e-12` per `layer_norm.py`) |
//!
//! The three non-derivable axes come from the fetched `config.yaml` of
//! `funasr/ct-punc` verbatim (`attention_heads: 12`, `sanm_shfit: 0`,
//! `sentence_end_id: 3`) — CLAUDE.md「ハルシネーション厳禁」.
//!
//! # Punctuation label inventory
//!
//! [`DEFAULT_PUNC_LIST`] is the `model_conf.punc_list` block of the fetched
//! `funasr/ct-punc` `config.yaml`, verbatim and in order:
//! `["<unk>", "_", "，", "。", "？", "、"]`. Index order is **load-bearing**
//! — the decoder head emits one logit per entry and every consumer's
//! `argmax` reads back through this list, so a silent reorder would
//! mis-punctuate every output. `"_"` is the "no punctuation here" label.
//!
//! The list is stamped as an `Array<String>` chunk so the runtime binder
//! **reads the inventory off the artifact** instead of hardcoding it. The
//! converter refuses to write a GGUF whose `decoder.weight` row count
//! disagrees with the list length — a checkpoint with a different label set
//! is a *different* model and must not be silently mislabelled (FR-EX-08).
//!
//! # Token list is NOT embedded (deliberate, documented)
//!
//! The upstream `tokens.json` holds **471 067** entries (~8.3 MB of UTF-8).
//! Embedding it would be a `--config` side-car axis, and the upstream
//! tokenizer is `CharTokenizer` fronted by **jieba** word segmentation
//! (`split_words(text, jieba_usr_dict=…)`) — a dictionary-driven Chinese
//! segmenter that is a separate work package, not something to fake. So
//! this converter emits the *model*, and the runtime binder's forward takes
//! **token ids**, not text. `vocab_size` is stamped from the real embedding
//! shape so a future tokenizer side-car can be validated against it.
//!
//! # Scale
//!
//! `model.pt` is 1 125 507 622 bytes (~1.05 GiB) — dominated by the
//! 471 067 × 516 embedding table. Well under the ≥8 GB vast.ai cutoff
//! (memory `[[feedback-large-models-on-vast-ai]]`), so local conversion on
//! the 16 GB M1 iMac is fine; peak is roughly 3× the checkpoint (input read
//! + per-tensor copy + output serialise).
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 tensors ride the verbatim pass-through arm. BF16 stays
//! GGUF type 30 ([`GgmlType::BF16`]); the runtime widens BF16 → f32
//! losslessly at load through the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream `state_dict` names verbatim**. Those
//! names are not folklore — PyTorch derives `state_dict` keys mechanically
//! from the module attribute path, so `CTTransformer.embed` →
//! `embed.weight`, `CTTransformer.encoder.encoders0[0].self_attn.fsmn_block`
//! → `encoder.encoders0.0.self_attn.fsmn_block.weight`, and so on. Note the
//! `encoders0` (1 block) / `encoders` (`num_blocks - 1` blocks) split is
//! real upstream structure (`SANMEncoder.__init__` builds two `repeat(...)`
//! containers), **not** a typo.
//!
//! # No ONNX / no pickle (permanent)
//!
//! CT-Punc ships as `model.pt` (a torch pickle); this converter **never**
//! touches ONNX or pickle (FR-LD-05 / NFR-DS-02). The `.pt` → safetensors
//! bridge is the existing offline sidecar
//! `tools/parity/nemo_pt_to_safetensors.py` (uv-managed Python 3.12 per
//! memory `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`),
//! which is not part of the runtime — pickle deserialisation inside the
//! Rust runtime would violate the FR-LD-05 "no arbitrary code execution at
//! load" rule.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` = `ct_punc` — distinct from every sibling
/// Transformer-over-text arch tag (`bert_base`, `deberta_v2`, `deberta_v3`)
/// and from the sibling SAN-M *speech* encoder (`sensevoicesmall`) and the
/// sibling FSMN *VAD* stack (`fsmn-vad`). FR-EX-08 forbids the silent
/// shape misroute.
pub const ARCH: &str = "ct_punc";

/// `vokra.model.name` — canonical slug for the `funasr/ct-punc` release
/// (the cn-en `vocab471067-large` CT-Transformer, per the repo's own
/// `configuration.json` ModelScope mirror id).
pub const NAME: &str = "ct-punc";

/// `vokra.model.category` = `punctuation` — **the first entry of this
/// category in the tree**. Consumed by the model-card generator + zoo
/// manifest tier gate so a text post-processing model is never advertised
/// as an ASR / TTS release.
pub const CATEGORY: &str = "punctuation";

/// Upstream HuggingFace repository slug — recorded under
/// `vokra.provenance.upstream_hf`.
pub const UPSTREAM_HF: &str = "funasr/ct-punc";

/// Default weight-licence SPDX (`apache-2.0`) per the `funasr/ct-punc`
/// model-card front-matter and the HF model API `cardData.license`, both
/// read 2026-08-15. **Not** the FunASR `MODEL_LICENSE` that the sibling
/// SenseVoiceSmall release carries — see the module docstring. Overridable
/// through [`convert_ct_punc_file`]'s `license` parameter (the standing
/// `convert_file_licensed` mechanism for "implementation is clean-room but
/// the redistributed checkpoint carries a different SPDX").
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

/// Ad-hoc metadata key for the model category. Converter-side constant
/// (not a `chunks::KEY_*` alias) matching the sibling `fsmn_vad` /
/// `gtcrn` / `emotion2vec` posture until a first-class `category` consumer
/// lands in `vokra-core`.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Ad-hoc metadata key for the upstream HuggingFace slug — sibling to
/// `fsmn_vad::KEY_PROVENANCE_UPSTREAM_HF`.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

// ---- `vokra.ct_punc.*` hparam chunk group --------------------------------

/// GGUF metadata key: token-embedding vocabulary size (u32). Derived from
/// `embed.weight` dim 0 at convert time.
pub const KEY_CT_PUNC_VOCAB_SIZE: &str = "vokra.ct_punc.vocab_size";
/// GGUF metadata key: token-embedding width (u32). Derived from
/// `embed.weight` dim 1.
pub const KEY_CT_PUNC_EMBED_UNIT: &str = "vokra.ct_punc.embed_unit";
/// GGUF metadata key: encoder model width (u32). Derived from
/// `decoder.weight` dim 1. Upstream `config.yaml` has
/// `embed_unit == att_unit == 516`; both are stamped separately anyway so a
/// future variant that decouples them still round-trips.
pub const KEY_CT_PUNC_ATT_UNIT: &str = "vokra.ct_punc.att_unit";
/// GGUF metadata key: self-attention head count (u32). NOT derivable from
/// any tensor shape (it is a reshape parameter), so it comes from the
/// upstream `config.yaml` `encoder_conf.attention_heads` and is validated
/// against `att_unit % heads == 0`.
pub const KEY_CT_PUNC_ATTENTION_HEADS: &str = "vokra.ct_punc.attention_heads";
/// GGUF metadata key: position-wise feed-forward hidden width (u32).
/// Derived from `…feed_forward.w_1.weight` dim 0.
pub const KEY_CT_PUNC_LINEAR_UNITS: &str = "vokra.ct_punc.linear_units";
/// GGUF metadata key: encoder block count (u32), counting the `encoders0`
/// block. Derived by walking `encoder.encoders.{i}.norm1.weight`.
pub const KEY_CT_PUNC_NUM_BLOCKS: &str = "vokra.ct_punc.num_blocks";
/// GGUF metadata key: FSMN memory-branch depthwise kernel width (u32).
/// Derived from `…self_attn.fsmn_block.weight` last dim.
pub const KEY_CT_PUNC_KERNEL_SIZE: &str = "vokra.ct_punc.kernel_size";
/// GGUF metadata key: extra left padding applied to the FSMN memory branch
/// (u32, upstream spells it `sanm_shfit`). NOT derivable from a shape.
pub const KEY_CT_PUNC_SANM_SHIFT: &str = "vokra.ct_punc.sanm_shift";
/// GGUF metadata key: index into `punc_list` that terminates a sentence
/// (u32). Upstream `model_conf.sentence_end_id`.
pub const KEY_CT_PUNC_SENTENCE_END_ID: &str = "vokra.ct_punc.sentence_end_id";
/// GGUF metadata key: LayerNorm epsilon (f32). `funasr` subclasses
/// `torch.nn.LayerNorm` with a fixed `eps=1e-12`
/// (`funasr/models/transformer/layer_norm.py`).
pub const KEY_CT_PUNC_LAYER_NORM_EPS: &str = "vokra.ct_punc.layer_norm_eps";
/// GGUF metadata key: the punctuation label inventory, `Array<String>` in
/// decoder-head order. The runtime binder reads the labels from HERE rather
/// than hardcoding them.
pub const KEY_CT_PUNC_PUNC_LIST: &str = "vokra.ct_punc.punc_list";

/// Self-attention head count from the upstream `funasr/ct-punc`
/// `config.yaml` (`encoder_conf.attention_heads: 12`).
pub const DEFAULT_ATTENTION_HEADS: u32 = 12;
/// Extra FSMN left padding from the upstream `config.yaml`
/// (`encoder_conf.sanm_shfit: 0`).
pub const DEFAULT_SANM_SHIFT: u32 = 0;
/// Sentence-terminating label index from the upstream `config.yaml`
/// (`model_conf.sentence_end_id: 3`, i.e. `"。"` in [`DEFAULT_PUNC_LIST`]).
pub const DEFAULT_SENTENCE_END_ID: u32 = 3;
/// LayerNorm epsilon — `funasr`'s `LayerNorm` calls
/// `super().__init__(nout, eps=1e-12)`.
pub const DEFAULT_LAYER_NORM_EPS: f32 = 1e-12;

/// Punctuation label inventory, verbatim from the `model_conf.punc_list`
/// block of the fetched `funasr/ct-punc` `config.yaml`.
///
/// Index order is load-bearing: the decoder head emits one logit per entry
/// and `argmax` indexes straight back into this list (see the upstream
/// `CTTransformer.inference`, which does `self.punc_list[punctuations[i]]`).
/// `"_"` means "emit no punctuation after this token".
pub const DEFAULT_PUNC_LIST: [&str; 6] = ["<unk>", "_", "，", "。", "？", "、"];

/// Required tensor: the token-embedding table (`[vocab_size, embed_unit]`).
const TENSOR_EMBED: &str = "embed.weight";
/// Required tensor: the punctuation classification head weight
/// (`[punc_size, att_unit]`).
const TENSOR_DECODER_WEIGHT: &str = "decoder.weight";
/// Required tensor: the FSMN depthwise kernel of the first encoder block
/// (`[att_unit, 1, kernel_size]`) — the axis probe for `kernel_size`.
const TENSOR_BLOCK0_FSMN: &str = "encoder.encoders0.0.self_attn.fsmn_block.weight";
/// Required tensor: the first FFN projection of the first encoder block
/// (`[linear_units, att_unit]`) — the axis probe for `linear_units`.
const TENSOR_BLOCK0_FFN_W1: &str = "encoder.encoders0.0.feed_forward.w_1.weight";

const UPSTREAM_SOURCE: &str = "funasr/ct-punc (FunASR CT-Transformer punctuation restoration; \
     SANM encoder 12 blocks x 516 over 471067-token CharTokenizer vocab; \
     Chen et al. 2020 arXiv:2003.01309; apache-2.0)";

/// Outcome of a CT-Punc conversion.
///
/// Counters mirror the sibling `gtcrn` / `fsmn_vad` `Report` shape; the
/// derived-topology fields exist so a caller (and the CLI note) can show
/// what was actually read off the checkpoint rather than what was assumed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CtPuncReport {
    /// Total tensors surfaced by the safetensors reader
    /// (`written + skipped_non_float`).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// Of `written`, how many were BF16 (subset counter).
    pub bf16_passthrough: usize,
    /// Vocabulary size read off `embed.weight` dim 0.
    pub vocab_size: u32,
    /// Encoder width read off `decoder.weight` dim 1.
    pub att_unit: u32,
    /// Encoder block count (including the `encoders0` block).
    pub num_blocks: u32,
    /// Punctuation label count — equals `decoder.weight` dim 0 and
    /// [`DEFAULT_PUNC_LIST`]`.len()` (the converter refuses a mismatch).
    pub punc_size: u32,
}

/// Builds an `Array<String>` metadata chunk.
fn string_array_chunk(values: &[&str]) -> GgufMetadataValue {
    GgufMetadataValue::Array(GgufArray {
        element_type: GgufValueType::String,
        values: values
            .iter()
            .map(|s| GgufMetadataValue::String((*s).to_owned()))
            .collect(),
    })
}

/// Looks up a required tensor and returns its shape, or a loud
/// [`ConvertError::Parse`] naming the tensor and what it is needed for.
fn require_shape<'a>(
    st: &'a SafetensorsFile,
    name: &str,
    needed_for: &str,
) -> Result<&'a [u64], ConvertError> {
    let info = st.tensor_info(name).ok_or_else(|| {
        ConvertError::Parse(format!(
            "ct_punc: required tensor `{name}` is missing — it is the source of truth for \
             {needed_for}. This does not look like a FunASR CT-Transformer checkpoint \
             (upstream reference: github.com/modelscope/FunASR \
             `funasr/models/ct_transformer/model.py`, release \
             huggingface.co/{UPSTREAM_HF}). If the input came from a torch `.pt`, run it \
             through `tools/parity/nemo_pt_to_safetensors.py` first and keep the \
             state_dict names verbatim."
        ))
    })?;
    Ok(&info.shape)
}

/// Counts encoder blocks: the single `encoders0` block plus however many
/// `encoder.encoders.{i}` blocks are present, walked contiguously from 0.
fn count_blocks(st: &SafetensorsFile) -> u32 {
    let mut extra = 0u32;
    while st
        .tensor_info(&format!("encoder.encoders.{extra}.norm1.weight"))
        .is_some()
    {
        extra += 1;
    }
    // `encoders0` contributes exactly one block (SANMEncoder builds it with
    // `repeat(1, ...)`), so total = 1 + extra.
    extra + 1
}

/// Reads a safetensors checkpoint at `input` and writes a CT-Punc GGUF to
/// `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// `state_dict` name. The `vokra.model.*` (arch / name / category),
/// `vokra.provenance.*` and `vokra.ct_punc.*` chunk groups are stamped;
/// `vokra.schema.*` is written unconditionally by the GGUF writer.
///
/// Topology axes are **derived from the checkpoint's own tensor shapes**
/// wherever a shape determines them (see the module docstring table); the
/// three that no shape determines (`attention_heads`, `sanm_shift`,
/// `sentence_end_id`) come from the upstream `config.yaml` and are
/// validated against the derived axes.
///
/// `license` overrides [`DEFAULT_LICENSE_SPDX`] (`"apache-2.0"`).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read / write failure.
/// - [`ConvertError::Parse`] on a malformed safetensors input, on a missing
///   required tensor, on a rank that disagrees with the reference topology,
///   on `att_unit % attention_heads != 0`, on a `decoder.weight` row count
///   that disagrees with [`DEFAULT_PUNC_LIST`], or on a `sentence_end_id`
///   outside the label inventory. Every one of these is a *loud* refusal —
///   writing a mislabelled GGUF would silently mis-punctuate downstream
///   (FR-EX-08).
/// - [`ConvertError::Gguf`] if GGUF serialisation fails.
pub fn convert_ct_punc_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<CtPuncReport, ConvertError> {
    // Whole-file read: ~1.05 GiB for the real release, well under the
    // ≥8 GB vast.ai cutoff. The streaming path (Moshi / Voxtral) is
    // reserved for checkpoints that cannot be materialised at all.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    // ---- Derive the topology from the checkpoint --------------------------

    let embed_shape = require_shape(&st, TENSOR_EMBED, "vocab_size / embed_unit")?;
    if embed_shape.len() != 2 {
        return Err(ConvertError::Parse(format!(
            "ct_punc: `{TENSOR_EMBED}` must be rank-2 [vocab_size, embed_unit], got rank {} \
             (shape {embed_shape:?})",
            embed_shape.len()
        )));
    }
    let vocab_size = embed_shape[0];
    let embed_unit = embed_shape[1];

    let decoder_shape = require_shape(&st, TENSOR_DECODER_WEIGHT, "att_unit / punc_size")?;
    if decoder_shape.len() != 2 {
        return Err(ConvertError::Parse(format!(
            "ct_punc: `{TENSOR_DECODER_WEIGHT}` must be rank-2 [punc_size, att_unit], got rank {} \
             (shape {decoder_shape:?})",
            decoder_shape.len()
        )));
    }
    let punc_size = decoder_shape[0];
    let att_unit = decoder_shape[1];

    let fsmn_shape = require_shape(&st, TENSOR_BLOCK0_FSMN, "kernel_size")?;
    // `nn.Conv1d(D, D, k, groups=D, bias=False).weight` is [D, 1, k].
    if fsmn_shape.len() != 3 {
        return Err(ConvertError::Parse(format!(
            "ct_punc: `{TENSOR_BLOCK0_FSMN}` must be rank-3 [att_unit, 1, kernel_size] \
             (a depthwise nn.Conv1d weight), got rank {} (shape {fsmn_shape:?})",
            fsmn_shape.len()
        )));
    }
    let kernel_size = fsmn_shape[2];

    let ffn_shape = require_shape(&st, TENSOR_BLOCK0_FFN_W1, "linear_units")?;
    if ffn_shape.len() != 2 {
        return Err(ConvertError::Parse(format!(
            "ct_punc: `{TENSOR_BLOCK0_FFN_W1}` must be rank-2 [linear_units, att_unit], got rank \
             {} (shape {ffn_shape:?})",
            ffn_shape.len()
        )));
    }
    let linear_units = ffn_shape[0];

    let num_blocks = count_blocks(&st);

    // ---- Validate the non-derivable axes against the derived ones --------

    // `DEFAULT_ATTENTION_HEADS` is a non-zero primary-source constant (12),
    // pinned by `attention_head_count_is_non_zero` below so the `%` here can
    // never be a division by zero.
    if att_unit == 0 {
        return Err(ConvertError::Parse(format!(
            "ct_punc: att_unit is 0 (from `{TENSOR_DECODER_WEIGHT}` dim 1) — a zero-width \
             encoder is not a loadable checkpoint"
        )));
    }
    if att_unit % u64::from(DEFAULT_ATTENTION_HEADS) != 0 {
        return Err(ConvertError::Parse(format!(
            "ct_punc: att_unit ({att_unit}, from `{TENSOR_DECODER_WEIGHT}` dim 1) is not divisible \
             by attention_heads ({DEFAULT_ATTENTION_HEADS}, from the upstream config.yaml \
             `encoder_conf.attention_heads`). MultiHeadedAttentionSANM asserts \
             `n_feat % n_head == 0`, so this checkpoint is not the `{UPSTREAM_HF}` topology."
        )));
    }
    if punc_size != DEFAULT_PUNC_LIST.len() as u64 {
        return Err(ConvertError::Parse(format!(
            "ct_punc: `{TENSOR_DECODER_WEIGHT}` has {punc_size} rows but the upstream \
             `{UPSTREAM_HF}` config.yaml `model_conf.punc_list` has {} labels \
             ({DEFAULT_PUNC_LIST:?}). Refusing to stamp a label inventory that does not match the \
             classification head — every downstream `argmax` reads back through this list, so a \
             mislabelled artifact would silently mis-punctuate (FR-EX-08). A checkpoint with a \
             different label set needs its own converter arm carrying its own punc_list.",
            DEFAULT_PUNC_LIST.len()
        )));
    }
    if u64::from(DEFAULT_SENTENCE_END_ID) >= punc_size {
        return Err(ConvertError::Parse(format!(
            "ct_punc: sentence_end_id ({DEFAULT_SENTENCE_END_ID}) is out of range for a \
             {punc_size}-label inventory"
        )));
    }

    // Guard the u32 chunk widths (GGUF u32 axes; a checkpoint bigger than
    // 4 Gi rows in any of these is not a CT-Transformer).
    let as_u32 = |v: u64, what: &str| -> Result<u32, ConvertError> {
        u32::try_from(v).map_err(|_| {
            ConvertError::Parse(format!(
                "ct_punc: {what} ({v}) exceeds the u32 metadata chunk width"
            ))
        })
    };
    let vocab_size = as_u32(vocab_size, "vocab_size")?;
    let embed_unit = as_u32(embed_unit, "embed_unit")?;
    let att_unit = as_u32(att_unit, "att_unit")?;
    let linear_units = as_u32(linear_units, "linear_units")?;
    let kernel_size = as_u32(kernel_size, "kernel_size")?;
    let punc_size = as_u32(punc_size, "punc_size")?;

    // ---- Build --------------------------------------------------------------

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    let effective_license = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_license);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_license,
        Some(NAME),
        Some(UPSTREAM_SOURCE),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    b.add_u32(KEY_CT_PUNC_VOCAB_SIZE, vocab_size);
    b.add_u32(KEY_CT_PUNC_EMBED_UNIT, embed_unit);
    b.add_u32(KEY_CT_PUNC_ATT_UNIT, att_unit);
    b.add_u32(KEY_CT_PUNC_ATTENTION_HEADS, DEFAULT_ATTENTION_HEADS);
    b.add_u32(KEY_CT_PUNC_LINEAR_UNITS, linear_units);
    b.add_u32(KEY_CT_PUNC_NUM_BLOCKS, num_blocks);
    b.add_u32(KEY_CT_PUNC_KERNEL_SIZE, kernel_size);
    b.add_u32(KEY_CT_PUNC_SANM_SHIFT, DEFAULT_SANM_SHIFT);
    b.add_u32(KEY_CT_PUNC_SENTENCE_END_ID, DEFAULT_SENTENCE_END_ID);
    b.add_f32(KEY_CT_PUNC_LAYER_NORM_EPS, DEFAULT_LAYER_NORM_EPS);
    b.add_metadata(
        KEY_CT_PUNC_PUNC_LIST,
        string_array_chunk(&DEFAULT_PUNC_LIST),
    );

    let mut report = CtPuncReport {
        vocab_size,
        att_unit,
        num_blocks,
        punc_size,
        ..CtPuncReport::default()
    };
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
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch path (PID + monotonic sequence — the
    /// gtcrn / sepformer pattern; no external `tempfile` dep, preserving
    /// zero-dep NFR-DS-02).
    fn scratch_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-ct-punc-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    /// One entry of the synthetic safetensors fixture.
    struct Entry {
        name: String,
        dtype: &'static str,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    }

    fn f32_entry(name: &str, shape: &[u64]) -> Entry {
        let n: u64 = shape.iter().product();
        // Non-zero, non-uniform payload so a silent widen / reorder cannot
        // hide behind a trivially symmetric buffer.
        let bytes: Vec<u8> = (0..n)
            .flat_map(|i| ((i as f32) * 0.125 - 1.0).to_le_bytes())
            .collect();
        Entry {
            name: name.to_owned(),
            dtype: "F32",
            shape: shape.to_vec(),
            bytes,
        }
    }

    fn bf16_entry(name: &str, shape: &[u64], values: &[f32]) -> Entry {
        let n: u64 = shape.iter().product();
        assert_eq!(values.len() as u64, n, "bf16_entry: shape/value mismatch");
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        Entry {
            name: name.to_owned(),
            dtype: "BF16",
            shape: shape.to_vec(),
            bytes,
        }
    }

    /// Serialises entries into a minimal safetensors buffer.
    fn safetensors(entries: &[Entry]) -> Vec<u8> {
        let mut header = String::from("{");
        let mut offset = 0usize;
        for (i, e) in entries.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let shape_str = e
                .shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let end = offset + e.bytes.len();
            header.push_str(&format!(
                r#""{}":{{"dtype":"{}","shape":[{}],"data_offsets":[{},{}]}}"#,
                e.name, e.dtype, shape_str, offset, end
            ));
            offset = end;
        }
        header.push('}');
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        for e in entries {
            out.extend_from_slice(&e.bytes);
        }
        out
    }

    // Synthetic topology: D = 24 so `D % 12 heads == 0` (d_k = 2), FFN = 8,
    // 2 blocks, kernel 3, vocab 7, 6 punctuation labels (matching the real
    // inventory length, which the converter enforces).
    const D: u64 = 24;
    const FF: u64 = 8;
    const VOCAB: u64 = 7;
    const K: u64 = 3;
    const P: u64 = 6;

    fn block_entries(prefix: &str, out: &mut Vec<Entry>) {
        out.push(f32_entry(&format!("{prefix}.norm1.weight"), &[D]));
        out.push(f32_entry(&format!("{prefix}.norm1.bias"), &[D]));
        out.push(f32_entry(&format!("{prefix}.norm2.weight"), &[D]));
        out.push(f32_entry(&format!("{prefix}.norm2.bias"), &[D]));
        out.push(f32_entry(
            &format!("{prefix}.self_attn.linear_q_k_v.weight"),
            &[D * 3, D],
        ));
        out.push(f32_entry(
            &format!("{prefix}.self_attn.linear_q_k_v.bias"),
            &[D * 3],
        ));
        out.push(f32_entry(
            &format!("{prefix}.self_attn.linear_out.weight"),
            &[D, D],
        ));
        out.push(f32_entry(
            &format!("{prefix}.self_attn.linear_out.bias"),
            &[D],
        ));
        out.push(f32_entry(
            &format!("{prefix}.self_attn.fsmn_block.weight"),
            &[D, 1, K],
        ));
        out.push(f32_entry(
            &format!("{prefix}.feed_forward.w_1.weight"),
            &[FF, D],
        ));
        out.push(f32_entry(&format!("{prefix}.feed_forward.w_1.bias"), &[FF]));
        out.push(f32_entry(
            &format!("{prefix}.feed_forward.w_2.weight"),
            &[D, FF],
        ));
        out.push(f32_entry(&format!("{prefix}.feed_forward.w_2.bias"), &[D]));
    }

    /// A structurally complete 2-block synthetic CT-Punc checkpoint.
    fn full_checkpoint() -> Vec<Entry> {
        let mut e = vec![f32_entry(TENSOR_EMBED, &[VOCAB, D])];
        block_entries("encoder.encoders0.0", &mut e);
        block_entries("encoder.encoders.0", &mut e);
        e.push(f32_entry("encoder.after_norm.weight", &[D]));
        e.push(f32_entry("encoder.after_norm.bias", &[D]));
        e.push(f32_entry(TENSOR_DECODER_WEIGHT, &[P, D]));
        e.push(f32_entry("decoder.bias", &[P]));
        e
    }

    // -----------------------------------------------------------------
    // Test 1 — full round-trip: every stamp lands, axes are DERIVED
    // -----------------------------------------------------------------

    #[test]
    fn full_checkpoint_round_trips_and_derives_topology_from_shapes() {
        let entries = full_checkpoint();
        let expected_tensors = entries.len();
        let input = scratch_path("full-in");
        let output = scratch_path("full-out");
        std::fs::write(&input, safetensors(&entries)).expect("write safetensors input");

        let report = convert_ct_punc_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, expected_tensors);
        assert_eq!(report.written, expected_tensors, "all tensors are float");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);
        // Derived, not assumed.
        assert_eq!(report.vocab_size, VOCAB as u32);
        assert_eq!(report.att_unit, D as u32);
        assert_eq!(report.num_blocks, 2, "encoders0 (1) + encoders.0 (1)");
        assert_eq!(report.punc_size, P as u32);

        let file = GgufFile::parse(std::fs::read(&output).expect("read gguf")).expect("parse gguf");

        // arch / name / category / provenance stamps.
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
            Some("punctuation"),
            "brand-new category tag must land verbatim"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 normalises to LicenseClass::Permissive"
        );
        assert!(file.get(chunks::KEY_SCHEMA_VERSION).is_some());
        assert!(file.get(chunks::KEY_SCHEMA_PRODUCER).is_some());

        // Derived axes.
        for (k, want) in [
            (KEY_CT_PUNC_VOCAB_SIZE, VOCAB),
            (KEY_CT_PUNC_EMBED_UNIT, D),
            (KEY_CT_PUNC_ATT_UNIT, D),
            (KEY_CT_PUNC_LINEAR_UNITS, FF),
            (KEY_CT_PUNC_NUM_BLOCKS, 2),
            (KEY_CT_PUNC_KERNEL_SIZE, K),
        ] {
            assert_eq!(
                file.get(k).and_then(|v| v.as_u64()),
                Some(want),
                "axis `{k}` must be derived from the checkpoint as {want}"
            );
        }
        // Config-sourced axes.
        assert_eq!(
            file.get(KEY_CT_PUNC_ATTENTION_HEADS)
                .and_then(|v| v.as_u64()),
            Some(u64::from(DEFAULT_ATTENTION_HEADS))
        );
        assert_eq!(
            file.get(KEY_CT_PUNC_SANM_SHIFT).and_then(|v| v.as_u64()),
            Some(u64::from(DEFAULT_SANM_SHIFT))
        );
        assert_eq!(
            file.get(KEY_CT_PUNC_SENTENCE_END_ID)
                .and_then(|v| v.as_u64()),
            Some(u64::from(DEFAULT_SENTENCE_END_ID))
        );
        let eps = file
            .get(KEY_CT_PUNC_LAYER_NORM_EPS)
            .and_then(|v| v.as_f64())
            .expect("layer_norm_eps stamped");
        assert!(
            (eps - f64::from(DEFAULT_LAYER_NORM_EPS)).abs() < 1e-20,
            "LayerNorm eps must round-trip as 1e-12 (funasr LayerNorm), got {eps}"
        );

        // Punctuation label inventory, in order.
        let labels = file
            .get(KEY_CT_PUNC_PUNC_LIST)
            .and_then(|v| v.as_array())
            .expect("punc_list array stamped");
        assert_eq!(labels.element_type, GgufValueType::String);
        let got: Vec<&str> = labels.values.iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(
            got,
            DEFAULT_PUNC_LIST.to_vec(),
            "punc_list must be the upstream config.yaml order verbatim"
        );

        // A representative tensor survives byte-identically.
        let want = &entries
            .iter()
            .find(|e| e.name == TENSOR_DECODER_WEIGHT)
            .expect("fixture has decoder.weight")
            .bytes;
        let info = file
            .tensor_info(TENSOR_DECODER_WEIGHT)
            .expect("decoder.weight present");
        assert_eq!(info.dtype, GgmlType::F32);
        assert_eq!(info.dimensions, vec![P, D]);
        assert_eq!(file.tensor_bytes(info), want.as_slice());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 2 — BF16 rides the pass-through arm byte-identically
    // -----------------------------------------------------------------

    #[test]
    fn bf16_tensor_passes_through_byte_identically() {
        let mut entries = full_checkpoint();
        // Swap the decoder bias for a BF16 copy with known bit patterns.
        entries.retain(|e| e.name != "decoder.bias");
        let values: [f32; P as usize] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        entries.push(bf16_entry("decoder.bias", &[P], &values));
        let bf16_bytes = entries
            .last()
            .expect("just pushed the bf16 entry")
            .bytes
            .clone();

        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, safetensors(&entries)).expect("write safetensors input");

        let report = convert_ct_punc_file(&input, &output, None).expect("convert");
        assert_eq!(
            report.bf16_passthrough, 1,
            "the BF16 tensor must land on the pass-through arm"
        );
        assert_eq!(report.skipped_non_float, 0);

        let file = GgufFile::parse(std::fs::read(&output).expect("read gguf")).expect("parse gguf");
        let info = file.tensor_info("decoder.bias").expect("bias present");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays GGUF type 30"
        );
        assert_eq!(file.tensor_bytes(info), bf16_bytes.as_slice());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 3 — malformed / foreign inputs are refused LOUDLY
    // -----------------------------------------------------------------

    #[test]
    fn truncated_safetensors_header_is_refused_loudly() {
        let input = scratch_path("trunc-in");
        let output = scratch_path("trunc-out");
        // A plausible-looking 8-byte length prefix followed by nothing.
        std::fs::write(&input, 4096u64.to_le_bytes()).expect("write truncated input");
        let err = convert_ct_punc_file(&input, &output, None)
            .expect_err("a truncated safetensors header must not convert");
        // The reader's own parse error is surfaced, not swallowed.
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected a loud parse refusal, got {err:?}"
        );
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn checkpoint_without_embed_table_names_the_missing_tensor() {
        let entries: Vec<Entry> = full_checkpoint()
            .into_iter()
            .filter(|e| e.name != TENSOR_EMBED)
            .collect();
        let input = scratch_path("noembed-in");
        let output = scratch_path("noembed-out");
        std::fs::write(&input, safetensors(&entries)).expect("write safetensors input");

        let err = convert_ct_punc_file(&input, &output, None)
            .expect_err("a checkpoint without embed.weight is not a CT-Transformer");
        let msg = err.to_string();
        assert!(
            msg.contains(TENSOR_EMBED),
            "the refusal must name the missing tensor, got: {msg}"
        );
        assert!(
            msg.contains("modelscope/FunASR"),
            "the refusal must cite the upstream reference, got: {msg}"
        );
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn decoder_head_width_disagreeing_with_the_label_inventory_is_refused() {
        // A head with 4 outputs cannot be labelled by a 6-entry inventory.
        let mut entries: Vec<Entry> = full_checkpoint()
            .into_iter()
            .filter(|e| e.name != TENSOR_DECODER_WEIGHT && e.name != "decoder.bias")
            .collect();
        entries.push(f32_entry(TENSOR_DECODER_WEIGHT, &[4, D]));
        entries.push(f32_entry("decoder.bias", &[4]));
        let input = scratch_path("badhead-in");
        let output = scratch_path("badhead-out");
        std::fs::write(&input, safetensors(&entries)).expect("write safetensors input");

        let err = convert_ct_punc_file(&input, &output, None)
            .expect_err("a 4-way head must not be stamped with a 6-label inventory");
        let msg = err.to_string();
        assert!(
            msg.contains('4') && msg.contains('6'),
            "the refusal must name BOTH the head width and the label count, got: {msg}"
        );
        assert!(
            !output.exists(),
            "no GGUF may be written when the label inventory is refused"
        );
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn att_unit_not_divisible_by_head_count_is_refused() {
        // D = 10 is not divisible by 12 heads.
        const BAD_D: u64 = 10;
        let entries = vec![
            f32_entry(TENSOR_EMBED, &[VOCAB, BAD_D]),
            f32_entry(
                "encoder.encoders0.0.self_attn.fsmn_block.weight",
                &[BAD_D, 1, K],
            ),
            f32_entry("encoder.encoders0.0.feed_forward.w_1.weight", &[FF, BAD_D]),
            f32_entry(TENSOR_DECODER_WEIGHT, &[P, BAD_D]),
        ];
        let input = scratch_path("baddiv-in");
        let output = scratch_path("baddiv-out");
        std::fs::write(&input, safetensors(&entries)).expect("write safetensors input");

        let err = convert_ct_punc_file(&input, &output, None)
            .expect_err("MultiHeadedAttentionSANM asserts n_feat % n_head == 0");
        let msg = err.to_string();
        assert!(
            msg.contains("attention_heads"),
            "the refusal must name the divisibility constraint, got: {msg}"
        );
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 4 — licence override re-derives the class
    // -----------------------------------------------------------------

    #[test]
    fn license_override_swaps_spdx_and_class() {
        let entries = full_checkpoint();
        let input = scratch_path("lic-in");
        let output = scratch_path("lic-out");
        std::fs::write(&input, safetensors(&entries)).expect("write safetensors input");

        // cc-by-4.0 -> AttributionRequired: a real class change, so a
        // hard-coded `Permissive` at the stamp site would fail here.
        convert_ct_punc_file(&input, &output, Some("cc-by-4.0")).expect("convert with override");
        let file = GgufFile::parse(std::fs::read(&output).expect("read gguf")).expect("parse gguf");
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

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 5 — arch / category tag distinctness (FR-EX-08 pin)
    // -----------------------------------------------------------------

    #[test]
    fn arch_tag_distinct_from_sibling_text_and_sanm_arches() {
        assert_eq!(ARCH, "ct_punc");
        assert_eq!(NAME, "ct-punc");
        assert_eq!(CATEGORY, "punctuation");
        for sibling in [
            "bert_base",       // plain BERT (post-norm, learned positions)
            "deberta_v2",      // disentangled relative position
            "deberta_v3",      // disentangled relative position
            "sensevoicesmall", // SAN-M, but a speech encoder w/ 4 heads
            "fsmn-vad",        // FSMN memory, but no self-attention
            "w2v_bert2",       // wav2vec2-BERT speech encoder
            "whisper",         // ASR
        ] {
            assert_ne!(
                ARCH, sibling,
                "ct_punc (SANM encoder over TEXT tokens with a per-token punctuation head) and \
                 `{sibling}` are distinct arches — sharing an arch tag would misroute runtime \
                 dispatch (FR-EX-08)"
            );
        }
    }

    // -----------------------------------------------------------------
    // Test 6 — the primary-source label inventory is pinned
    // -----------------------------------------------------------------

    /// Pins the label inventory transcribed from the fetched
    /// `funasr/ct-punc` `config.yaml` `model_conf.punc_list`. A silent
    /// reorder here would mis-punctuate every output, so the order is
    /// asserted element by element rather than as a set.
    #[test]
    fn punc_list_matches_the_upstream_config_yaml_verbatim() {
        assert_eq!(DEFAULT_PUNC_LIST.len(), 6);
        assert_eq!(DEFAULT_PUNC_LIST[0], "<unk>");
        assert_eq!(
            DEFAULT_PUNC_LIST[1], "_",
            "index 1 is the no-punctuation label"
        );
        assert_eq!(DEFAULT_PUNC_LIST[2], "，");
        assert_eq!(DEFAULT_PUNC_LIST[3], "。");
        assert_eq!(DEFAULT_PUNC_LIST[4], "？");
        assert_eq!(DEFAULT_PUNC_LIST[5], "、");
        // `sentence_end_id: 3` in the same config.yaml must select "。".
        assert_eq!(
            DEFAULT_PUNC_LIST[DEFAULT_SENTENCE_END_ID as usize], "。",
            "config.yaml sentence_end_id=3 must index the ideographic full stop"
        );
    }

    /// The divisibility check in [`convert_ct_punc_file`] takes
    /// `att_unit % DEFAULT_ATTENTION_HEADS`. That is only safe because the
    /// head count is a non-zero primary-source constant; pin it here so a
    /// future edit that zeroes it fails a test rather than panicking at
    /// convert time.
    #[test]
    fn attention_head_count_is_non_zero() {
        assert_eq!(
            DEFAULT_ATTENTION_HEADS, 12,
            "upstream funasr/ct-punc config.yaml `encoder_conf.attention_heads: 12`"
        );
        const { assert!(DEFAULT_ATTENTION_HEADS > 0) };
    }
}
