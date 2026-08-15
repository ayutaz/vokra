//! **CT-Transformer punctuation restoration** (`funasr/ct-punc`,
//! **apache-2.0**) — runtime binder with a **real forward** (Wave D
//! 2026-08-15, the first `punctuation`-category model in the tree).
//!
//! # What this is for
//!
//! Vokra's ASR fleet emits raw token streams; several of those models emit
//! no punctuation at all. CT-Punc is the post-processing stage that turns
//! `我们今天讨论三个议题首先是产品发布` into
//! `我们今天讨论三个议题，首先是产品发布。`. Every primitive the forward
//! needs already exists in-tree, so — unlike the `storm` / `gtcrn` /
//! `emotion2vec` skeletons — this binder is **not** loud-partial: the
//! encoder runs for real.
//!
//! # Primary sources (fetched 2026-08-15, transcribed — not recalled)
//!
//! - HF release: <https://huggingface.co/funasr/ct-punc>. Model-card
//!   front-matter `license: apache-2.0`; the HF model API reports
//!   `cardData.license = "apache-2.0"`. Repo contents: `.gitattributes`,
//!   `README.md`, `config.yaml`, `configuration.json`,
//!   `example/punc_example.txt`, `fig/struct.png`, `model.pt`
//!   (1 125 507 622 bytes), `tokens.json` (471 067 entries). The repo's own
//!   `configuration.json` names the ModelScope mirror
//!   `iic/punc_ct-transformer_cn-en-common-vocab471067-large`.
//! - Toolkit: <https://github.com/modelscope/FunASR> — repo `LICENSE` is
//!   verbatim `MIT License / Copyright (c) 2025 FunASR`.
//! - Reference implementation this forward is transcribed from:
//!   `funasr/models/ct_transformer/model.py` (`CTTransformer.punc_forward`),
//!   `funasr/models/sanm/encoder.py` (`SANMEncoder.forward`,
//!   `EncoderLayerSANM.forward`), `funasr/models/sanm/attention.py`
//!   (`MultiHeadedAttentionSANM.forward` / `.forward_fsmn` / `.forward_qkv`
//!   / `.forward_attention`),
//!   `funasr/models/transformer/positionwise_feed_forward.py`
//!   (`PositionwiseFeedForward`),
//!   `funasr/models/transformer/embedding.py` (`SinusoidalPositionEncoder`),
//!   `funasr/models/transformer/layer_norm.py` (`LayerNorm`, `eps=1e-12`),
//!   `funasr/models/transformer/utils/repeat.py` (`MultiSequential`).
//! - Paper: Chen et al. 2020, *"Controllable Time-delay Transformer for
//!   Real-time Punctuation Prediction and Disfluency Detection"*,
//!   <https://arxiv.org/pdf/2003.01309.pdf>.
//!
//! # Licence — recorded honestly, NOT signed
//!
//! FunASR's *code* is MIT, but its *weight* releases do not all share it:
//! the sibling [`crate::sensevoicesmall_runtime`] binder correctly
//! fail-closes to [`LicenseClass::Unknown`] because
//! `FunAudioLLM/SenseVoiceSmall` ships the bespoke `MODEL_LICENSE`
//! ("FunASR Model Open Source License Agreement" v1.1, Alibaba Group)
//! instead of an SPDX id.
//!
//! **`funasr/ct-punc` is different**: its card front-matter and the HF API
//! both declare plain `apache-2.0`, and its file list carries no
//! `MODEL_LICENSE` sibling. So the expected stamped class is
//! [`LicenseClass::Permissive`]. That is what the primary source said on
//! 2026-08-15 — it is **not** a sign-off. `docs/license-audit.md` §3.1
//! sign-off stays **BLANK** (owner-only, fail-closed, per memory
//! `[[feedback-license-signoff-primary-source]]`). Loading is unblocked;
//! *publish* is blocked until the owner signs.
//!
//! # Architecture — exactly what [`CtPunc::logits`] runs
//!
//! ```text
//! token ids                                                  [T]
//!  1. x = embed[id]                                          [T, D]
//!  2. x *= sqrt(att_unit)             SANMEncoder.forward line 1
//!  3. x += sinusoidal_pe(T, D)        positions are 1-BASED (see below)
//!  4. for each of `num_blocks` EncoderLayerSANM blocks (pre-norm):
//!        x += sanm_attention(layer_norm(x, norm1))
//!        x += w_2(relu(w_1(layer_norm(x, norm2))))
//!  5. x = layer_norm(x, encoder.after_norm)
//!  6. logits = x · decoder.weight^T + decoder.bias           [T, P]
//! ```
//!
//! `sanm_attention` is the piece that makes this **not** a plain BERT
//! block:
//!
//! ```text
//! qkv       = h · linear_q_k_v^T + bias           (ONE fused 3D-wide proj)
//! q, k, v   = qkv[..0..D], qkv[..D..2D], qkv[..2D..3D]
//! fsmn_mem  = depthwise_conv1d(pad(v, left, right), fsmn_block) + v
//! q        *= d_k ** -0.5
//! attn      = softmax_over_keys(q · k^T)          per head, d_k = D / heads
//! out       = (attn · v) · linear_out^T + bias
//! return      out + fsmn_mem                      (parallel memory add)
//! ```
//!
//! Three details that are easy to get wrong and are therefore pinned by
//! tests in this module:
//!
//! 1. **Positions are 1-based.** Upstream does
//!    `positions = torch.arange(1, timesteps + 1)`, so the first token gets
//!    `sin(1 · inv[i])`, not `sin(0)`. A 0-based port would silently shift
//!    every position embedding by one step.
//! 2. **The PE layout is `[sin block | cos block]`, not interleaved.**
//!    Upstream is `torch.cat([sin(scaled), cos(scaled)], dim=2)`, so
//!    columns `0..D/2` are sines and `D/2..D` are cosines. (RoPE-style
//!    adjacent-pair interleaving would be wrong here.)
//! 3. **The embedding is scaled by `sqrt(output_size)` BEFORE the PE is
//!    added** (`xs_pad = xs_pad * self.output_size() ** 0.5` precedes
//!    `self.embed(xs_pad)` in `SANMEncoder.forward`).
//!
//! # Tensor names
//!
//! Upstream `state_dict` names verbatim. PyTorch derives those keys
//! mechanically from the module attribute path, so they are determined by
//! the reference source rather than guessed. Note the `encoders0` (exactly
//! one block) / `encoders` (`num_blocks - 1` blocks) split is real upstream
//! structure — `SANMEncoder.__init__` builds two separate `repeat(...)`
//! containers — so block 0 lives at `encoder.encoders0.0.*` and block `i`
//! (for `i >= 1`) at `encoder.encoders.{i - 1}.*`. See [`block_prefix`].
//!
//! # What this binder deliberately does NOT do
//!
//! - **Tokenisation.** Upstream fronts a `CharTokenizer` (471 067-entry
//!   `tokens.json`) with **jieba** word segmentation
//!   (`split_words(text, jieba_usr_dict=…)`). That is a dictionary-driven
//!   Chinese segmenter and a separate work package; faking it would produce
//!   confidently wrong token boundaries. So the forward takes **token
//!   ids**, and [`CtPunc::restore_with_labels`] takes the caller's already
//!   split tokens.
//! - **Mini-sentence chunking.** Upstream `inference` splits into
//!   `split_size = 20` token windows and carries a cache across windows,
//!   re-cutting at the last `。` / `？`. That policy needs the tokenizer to
//!   be meaningful, so it is left to the caller; this binder runs one
//!   window.
//! - **Padded batches.** `MultiHeadedAttentionSANM` takes a padding mask;
//!   this binder runs a single unbatched sequence where every position is
//!   valid, which is what the upstream single-utterance inference path does
//!   too. Batching is a follow-up, not a silent approximation.
//!
//! # Numerics
//!
//! Transcendentals route through `vokra_math` (`exp` / `sqrt` / `sin` /
//! `cos` / `log`) rather than `std`, matching the WP-10 / WP-12 policy for
//! cross-platform bit-identity inside Vokra. `vokra_math::sin` is accurate
//! for `|x| <= 4096·π/2 ≈ 6434`; the largest PE argument is the token
//! position itself (`inv[0] == 1.0`), so accuracy holds for sequences up to
//! ~6434 tokens — far beyond the upstream 20-token mini-sentence window.
//!
//! Real-checkpoint numerical parity against the upstream Python pipeline is
//! an owner task (it needs the 1.05 GiB `model.pt`); the tests here pin the
//! structure analytically instead — see `encoder_stack_is_identity_when_…`,
//! which zeroes the block LayerNorms so the whole encoder provably collapses
//! to the identity and the expected logits can be computed independently.
//!
//! # Memory
//!
//! The real embedding table is 471 067 × 516 f32 ≈ 972 MB, and
//! [`CtPuncWeights::from_gguf`] materialises it (the established
//! `vokra-bert` / `bert_base` posture). That is fine on the 16 GB dev
//! machine but it is the dominant cost of loading this model; an
//! mmap-lazy row gather (mirroring `crate::mapped_weights` / the Voxtral
//! `MappedLazy` port) is the follow-up for low-memory hosts.
//!
//! # No ONNX / no pickle (permanent)
//!
//! Upstream ships `model.pt` (a torch pickle). This runtime **never**
//! touches ONNX or pickle (FR-LD-05 / NFR-DS-02); the offline bridge is
//! `tools/parity/nemo_pt_to_safetensors.py` (uv-managed Python 3.12 per
//! memory `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`).

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirrors of `crates/vokra-convert/src/models/ct_punc.rs`.
//
// Duplicated on purpose so `vokra-models` never gains a dependency edge onto
// `vokra-convert`, preserving the layered convention `vokra-ops -> nothing
// GGUF-aware`, `vokra-core -> GGUF reader`, `vokra-models -> GGUF binder`,
// `vokra-convert -> GGUF writer`.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` — `ct_punc`.
///
/// Deliberately distinct from every sibling Transformer-over-text arch
/// (`bert_base` post-norm + learned positions + split q/k/v; `deberta_v2` /
/// `deberta_v3` disentangled relative position) and from the two sibling
/// arches that share *part* of CT-Punc's machinery but nothing else:
/// `sensevoicesmall` (SAN-M as well, but a speech encoder over fbank frames
/// with four per-task heads) and `fsmn-vad` (FSMN memory blocks, but no
/// self-attention and a 2-class frame output). Silent aliasing would let
/// runtime dispatch bind a CT-Punc checkpoint with a wrong-topology loader
/// (FR-EX-08).
pub const ARCH: &str = "ct_punc";

/// Expected `vokra.model.name` — `ct-punc`.
pub const NAME: &str = "ct-punc";

/// Expected `vokra.model.category` — `punctuation`, the first of its kind
/// in the tree.
pub const CATEGORY: &str = "punctuation";

/// Upstream HuggingFace slug, echoed in diagnostics.
pub const UPSTREAM_HF: &str = "funasr/ct-punc";

/// Primary-source anchor: the HF release.
pub const PRIMARY_SOURCE_HF: &str = "huggingface.co/funasr/ct-punc";
/// Primary-source anchor: the reference implementation.
pub const PRIMARY_SOURCE_CODE: &str =
    "github.com/modelscope/FunASR (funasr/models/ct_transformer/model.py + funasr/models/sanm/)";
/// Primary-source anchor: the paper.
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/pdf/2003.01309.pdf";

// ---- GGUF metadata keys (mirrors of the converter's `KEY_CT_PUNC_*`) ------

/// `vokra.ct_punc.vocab_size` — token-embedding vocabulary size (u32).
pub const KEY_VOCAB_SIZE: &str = "vokra.ct_punc.vocab_size";
/// `vokra.ct_punc.embed_unit` — token-embedding width (u32).
pub const KEY_EMBED_UNIT: &str = "vokra.ct_punc.embed_unit";
/// `vokra.ct_punc.att_unit` — encoder model width (u32).
pub const KEY_ATT_UNIT: &str = "vokra.ct_punc.att_unit";
/// `vokra.ct_punc.attention_heads` — self-attention head count (u32).
pub const KEY_ATTENTION_HEADS: &str = "vokra.ct_punc.attention_heads";
/// `vokra.ct_punc.linear_units` — feed-forward hidden width (u32).
pub const KEY_LINEAR_UNITS: &str = "vokra.ct_punc.linear_units";
/// `vokra.ct_punc.num_blocks` — encoder block count (u32), counting the
/// `encoders0` block.
pub const KEY_NUM_BLOCKS: &str = "vokra.ct_punc.num_blocks";
/// `vokra.ct_punc.kernel_size` — FSMN depthwise kernel width (u32).
pub const KEY_KERNEL_SIZE: &str = "vokra.ct_punc.kernel_size";
/// `vokra.ct_punc.sanm_shift` — extra left padding on the FSMN branch
/// (u32; upstream spells it `sanm_shfit`).
pub const KEY_SANM_SHIFT: &str = "vokra.ct_punc.sanm_shift";
/// `vokra.ct_punc.sentence_end_id` — index into the label inventory that
/// terminates a sentence (u32).
pub const KEY_SENTENCE_END_ID: &str = "vokra.ct_punc.sentence_end_id";
/// `vokra.ct_punc.layer_norm_eps` — LayerNorm epsilon (f32).
pub const KEY_LAYER_NORM_EPS: &str = "vokra.ct_punc.layer_norm_eps";
/// `vokra.ct_punc.punc_list` — the punctuation label inventory,
/// `Array<String>`, in decoder-head order.
///
/// **The binder reads the inventory from here.** It is never hardcoded: a
/// checkpoint that carries a different label set is a different model, and
/// its own artifact is the only honest source for what its head's columns
/// mean.
pub const KEY_PUNC_LIST: &str = "vokra.ct_punc.punc_list";

/// Tensor name: the token-embedding table, `[vocab_size, embed_unit]`.
pub const TENSOR_EMBED: &str = "embed.weight";
/// Tensor name: final encoder LayerNorm gain, `[att_unit]`.
pub const TENSOR_AFTER_NORM_WEIGHT: &str = "encoder.after_norm.weight";
/// Tensor name: final encoder LayerNorm bias, `[att_unit]`.
pub const TENSOR_AFTER_NORM_BIAS: &str = "encoder.after_norm.bias";
/// Tensor name: punctuation head weight, `[punc_size, att_unit]`.
pub const TENSOR_DECODER_WEIGHT: &str = "decoder.weight";
/// Tensor name: punctuation head bias, `[punc_size]`.
pub const TENSOR_DECODER_BIAS: &str = "decoder.bias";

/// The `state_dict` prefix of encoder block `idx`.
///
/// Upstream `SANMEncoder` builds the stack as **two** containers — a
/// `repeat(1, …)` named `encoders0` and a `repeat(num_blocks - 1, …)` named
/// `encoders` — so the flat block index maps as:
///
/// ```text
/// 0 -> "encoder.encoders0.0"
/// 1 -> "encoder.encoders.0"
/// 2 -> "encoder.encoders.1"
/// ```
///
/// This is real upstream structure, not a naming accident; getting it wrong
/// makes block 0 unbindable.
#[must_use]
pub fn block_prefix(idx: usize) -> String {
    if idx == 0 {
        "encoder.encoders0.0".to_owned()
    } else {
        format!("encoder.encoders.{}", idx - 1)
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Topology read from the `vokra.ct_punc.*` chunk group.
///
/// Every axis is required and every `0`-sentinel is refused: a silent
/// default would let a mis-stamped artifact run a wrong-shaped forward
/// (FR-EX-08).
#[derive(Debug, Clone, PartialEq)]
pub struct CtPuncConfig {
    /// Token-embedding vocabulary size.
    pub vocab_size: usize,
    /// Token-embedding width. Upstream has `embed_unit == att_unit == 516`;
    /// they are carried separately so a future variant that decouples them
    /// still loads.
    pub embed_unit: usize,
    /// Encoder model width (`D`).
    pub att_unit: usize,
    /// Self-attention head count. `att_unit % attention_heads == 0` is
    /// enforced (upstream `MultiHeadedAttentionSANM` asserts it).
    pub attention_heads: usize,
    /// Position-wise feed-forward hidden width.
    pub linear_units: usize,
    /// Encoder block count, counting the `encoders0` block.
    pub num_blocks: usize,
    /// FSMN depthwise kernel width.
    pub kernel_size: usize,
    /// Extra left padding on the FSMN branch (upstream `sanm_shfit`). May
    /// legitimately be `0`, so it is the one axis with no non-zero check.
    pub sanm_shift: usize,
    /// Index into [`CtPunc::punc_labels`] that terminates a sentence.
    pub sentence_end_id: usize,
    /// LayerNorm epsilon (upstream `torch.nn.LayerNorm(eps=1e-12)`).
    pub layer_norm_eps: f32,
}

impl CtPuncConfig {
    /// Head width `d_k = att_unit / attention_heads`.
    #[must_use]
    pub fn d_k(&self) -> usize {
        self.att_unit / self.attention_heads
    }

    /// Left / right zero padding applied before the FSMN depthwise
    /// convolution.
    ///
    /// Upstream: `left = (kernel_size - 1) // 2` (`+ sanm_shfit` when that
    /// is positive), `right = kernel_size - 1 - left`.
    #[must_use]
    pub fn fsmn_padding(&self) -> (usize, usize) {
        let mut left = (self.kernel_size - 1) / 2;
        if self.sanm_shift > 0 {
            left += self.sanm_shift;
        }
        let right = (self.kernel_size - 1).saturating_sub(left);
        (left, right)
    }

    /// Reads and validates the config from a GGUF's metadata.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] if any key is missing, is not a u32, is a
    /// `0`-sentinel where zero is meaningless, or if `att_unit` is not
    /// divisible by `attention_heads`.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let u32_key = |k: &str| -> Result<u64> {
            gguf.get(k).and_then(|v| v.as_u64()).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "ct_punc: GGUF is missing required unsigned metadata `{k}` — re-run \
                     `vokra-cli convert --model ct-punc` with a current converter \
                     (primary source: {PRIMARY_SOURCE_HF})"
                ))
            })
        };
        let nonzero = |k: &str, v: u64| -> Result<usize> {
            if v == 0 {
                return Err(VokraError::ModelLoad(format!(
                    "ct_punc: metadata `{k}` is 0 — refusing to substitute a default, a \
                     zero-width axis cannot describe a loadable checkpoint (FR-EX-08)"
                )));
            }
            usize::try_from(v).map_err(|_| {
                VokraError::ModelLoad(format!("ct_punc: metadata `{k}` ({v}) does not fit usize"))
            })
        };

        let vocab_size = nonzero(KEY_VOCAB_SIZE, u32_key(KEY_VOCAB_SIZE)?)?;
        let embed_unit = nonzero(KEY_EMBED_UNIT, u32_key(KEY_EMBED_UNIT)?)?;
        let att_unit = nonzero(KEY_ATT_UNIT, u32_key(KEY_ATT_UNIT)?)?;
        let attention_heads = nonzero(KEY_ATTENTION_HEADS, u32_key(KEY_ATTENTION_HEADS)?)?;
        let linear_units = nonzero(KEY_LINEAR_UNITS, u32_key(KEY_LINEAR_UNITS)?)?;
        let num_blocks = nonzero(KEY_NUM_BLOCKS, u32_key(KEY_NUM_BLOCKS)?)?;
        let kernel_size = nonzero(KEY_KERNEL_SIZE, u32_key(KEY_KERNEL_SIZE)?)?;
        // `sanm_shift` legitimately IS zero upstream, so it gets no
        // non-zero check — only a presence + width check.
        let sanm_shift = usize::try_from(u32_key(KEY_SANM_SHIFT)?).map_err(|_| {
            VokraError::ModelLoad("ct_punc: metadata `sanm_shift` does not fit usize".to_owned())
        })?;
        let sentence_end_id = usize::try_from(u32_key(KEY_SENTENCE_END_ID)?).map_err(|_| {
            VokraError::ModelLoad(
                "ct_punc: metadata `sentence_end_id` does not fit usize".to_owned(),
            )
        })?;

        if att_unit % attention_heads != 0 {
            return Err(VokraError::ModelLoad(format!(
                "ct_punc: att_unit ({att_unit}) is not divisible by attention_heads \
                 ({attention_heads}) — upstream MultiHeadedAttentionSANM asserts \
                 `n_feat % n_head == 0`, so this artifact is mis-stamped"
            )));
        }
        // Upstream computes `right = kernel_size - 1 - left`, which goes
        // NEGATIVE once `sanm_shfit` pushes `left` past `kernel_size - 1` —
        // and a negative `nn.ConstantPad1d` pad *crops* rather than pads.
        // The released `funasr/ct-punc` config has `sanm_shfit: 0`, so that
        // crop branch is unreachable in practice and is deliberately NOT
        // implemented here. Refuse it loudly instead of silently clamping to
        // zero padding, which would shift the whole memory branch (FR-EX-08).
        if (kernel_size - 1) / 2 + sanm_shift > kernel_size - 1 {
            return Err(VokraError::ModelLoad(format!(
                "ct_punc: sanm_shift ({sanm_shift}) with kernel_size ({kernel_size}) puts the FSMN \
                 left padding at {}, past kernel_size - 1 ({}). Upstream would turn the right pad \
                 negative (a crop); this runtime does not implement the crop branch, and the \
                 released {UPSTREAM_HF} config has sanm_shfit = 0. Refusing rather than clamping.",
                (kernel_size - 1) / 2 + sanm_shift,
                kernel_size - 1
            )));
        }

        let layer_norm_eps = gguf
            .get(KEY_LAYER_NORM_EPS)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "ct_punc: GGUF is missing required float metadata `{KEY_LAYER_NORM_EPS}`"
                ))
            })? as f32;
        if !(layer_norm_eps.is_finite() && layer_norm_eps > 0.0) {
            return Err(VokraError::ModelLoad(format!(
                "ct_punc: `{KEY_LAYER_NORM_EPS}` must be a finite positive float, got \
                 {layer_norm_eps}"
            )));
        }

        Ok(Self {
            vocab_size,
            embed_unit,
            att_unit,
            attention_heads,
            linear_units,
            num_blocks,
            kernel_size,
            sanm_shift,
            sentence_end_id,
            layer_norm_eps,
        })
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// One `EncoderLayerSANM` block's parameters, flat and row-major.
#[derive(Debug, Clone)]
pub struct CtPuncBlock {
    /// `norm1.weight`, `[D]`.
    pub norm1_weight: Vec<f32>,
    /// `norm1.bias`, `[D]`.
    pub norm1_bias: Vec<f32>,
    /// `norm2.weight`, `[D]`.
    pub norm2_weight: Vec<f32>,
    /// `norm2.bias`, `[D]`.
    pub norm2_bias: Vec<f32>,
    /// `self_attn.linear_q_k_v.weight`, `[3D, D]` (the ONE fused projection).
    pub qkv_weight: Vec<f32>,
    /// `self_attn.linear_q_k_v.bias`, `[3D]`.
    pub qkv_bias: Vec<f32>,
    /// `self_attn.linear_out.weight`, `[D, D]`.
    pub out_weight: Vec<f32>,
    /// `self_attn.linear_out.bias`, `[D]`.
    pub out_bias: Vec<f32>,
    /// `self_attn.fsmn_block.weight`, `[D, 1, kernel_size]` — a depthwise
    /// `nn.Conv1d(D, D, k, groups=D, bias=False)`, so there is no bias.
    pub fsmn_weight: Vec<f32>,
    /// `feed_forward.w_1.weight`, `[linear_units, D]`.
    pub ffn_w1_weight: Vec<f32>,
    /// `feed_forward.w_1.bias`, `[linear_units]`.
    pub ffn_w1_bias: Vec<f32>,
    /// `feed_forward.w_2.weight`, `[D, linear_units]`.
    pub ffn_w2_weight: Vec<f32>,
    /// `feed_forward.w_2.bias`, `[D]`.
    pub ffn_w2_bias: Vec<f32>,
}

/// Every tensor CT-Punc's forward needs, bound from a GGUF.
#[derive(Debug, Clone)]
pub struct CtPuncWeights {
    /// `embed.weight`, `[vocab_size, embed_unit]`.
    pub embed: Vec<f32>,
    /// The encoder stack, in flat block order (see [`block_prefix`]).
    pub blocks: Vec<CtPuncBlock>,
    /// `encoder.after_norm.weight`, `[D]`.
    pub after_norm_weight: Vec<f32>,
    /// `encoder.after_norm.bias`, `[D]`.
    pub after_norm_bias: Vec<f32>,
    /// `decoder.weight`, `[punc_size, D]`.
    pub decoder_weight: Vec<f32>,
    /// `decoder.bias`, `[punc_size]`.
    pub decoder_bias: Vec<f32>,
}

/// Loads one tensor and checks its element count against the expected shape.
///
/// A missing tensor names itself; a shape mismatch names the tensor, the
/// expected shape and the actual on-disk dimensions. Both are loud
/// (FR-EX-08) — a silently truncated or reshaped weight would produce
/// plausible-looking garbage.
fn bind_tensor(gguf: &GgufFile, name: &str, expect: &[usize]) -> Result<Vec<f32>> {
    let info = gguf.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "ct_punc: GGUF is missing required tensor `{name}` — expected shape {expect:?}. \
             Tensor names are the upstream FunASR state_dict names verbatim; see \
             {PRIMARY_SOURCE_CODE}"
        ))
    })?;
    let want: usize = expect.iter().product();
    let got_dims: Vec<u64> = info.dimensions.clone();
    let got: usize = got_dims.iter().product::<u64>() as usize;
    if got != want {
        return Err(VokraError::ModelLoad(format!(
            "ct_punc: tensor `{name}` has {got} elements (dimensions {got_dims:?}) but the \
             stamped topology requires {want} (shape {expect:?}) — refusing to reinterpret a \
             wrong-shaped weight (FR-EX-08)"
        )));
    }
    gguf.tensor_f32(name).map_err(|e| {
        VokraError::ModelLoad(format!("ct_punc: tensor `{name}` failed to decode: {e}"))
    })
}

impl CtPuncWeights {
    /// Binds every tensor named by `cfg` from `gguf`.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming the offending tensor if any is
    /// missing or wrongly shaped.
    pub fn from_gguf(gguf: &GgufFile, cfg: &CtPuncConfig, punc_size: usize) -> Result<Self> {
        let d = cfg.att_unit;
        let embed = bind_tensor(gguf, TENSOR_EMBED, &[cfg.vocab_size, cfg.embed_unit])?;

        let mut blocks = Vec::with_capacity(cfg.num_blocks);
        for i in 0..cfg.num_blocks {
            let p = block_prefix(i);
            blocks.push(CtPuncBlock {
                norm1_weight: bind_tensor(gguf, &format!("{p}.norm1.weight"), &[d])?,
                norm1_bias: bind_tensor(gguf, &format!("{p}.norm1.bias"), &[d])?,
                norm2_weight: bind_tensor(gguf, &format!("{p}.norm2.weight"), &[d])?,
                norm2_bias: bind_tensor(gguf, &format!("{p}.norm2.bias"), &[d])?,
                qkv_weight: bind_tensor(
                    gguf,
                    &format!("{p}.self_attn.linear_q_k_v.weight"),
                    &[3 * d, d],
                )?,
                qkv_bias: bind_tensor(gguf, &format!("{p}.self_attn.linear_q_k_v.bias"), &[3 * d])?,
                out_weight: bind_tensor(
                    gguf,
                    &format!("{p}.self_attn.linear_out.weight"),
                    &[d, d],
                )?,
                out_bias: bind_tensor(gguf, &format!("{p}.self_attn.linear_out.bias"), &[d])?,
                fsmn_weight: bind_tensor(
                    gguf,
                    &format!("{p}.self_attn.fsmn_block.weight"),
                    &[d, 1, cfg.kernel_size],
                )?,
                ffn_w1_weight: bind_tensor(
                    gguf,
                    &format!("{p}.feed_forward.w_1.weight"),
                    &[cfg.linear_units, d],
                )?,
                ffn_w1_bias: bind_tensor(
                    gguf,
                    &format!("{p}.feed_forward.w_1.bias"),
                    &[cfg.linear_units],
                )?,
                ffn_w2_weight: bind_tensor(
                    gguf,
                    &format!("{p}.feed_forward.w_2.weight"),
                    &[d, cfg.linear_units],
                )?,
                ffn_w2_bias: bind_tensor(gguf, &format!("{p}.feed_forward.w_2.bias"), &[d])?,
            });
        }

        Ok(Self {
            embed,
            blocks,
            after_norm_weight: bind_tensor(gguf, TENSOR_AFTER_NORM_WEIGHT, &[d])?,
            after_norm_bias: bind_tensor(gguf, TENSOR_AFTER_NORM_BIAS, &[d])?,
            decoder_weight: bind_tensor(gguf, TENSOR_DECODER_WEIGHT, &[punc_size, d])?,
            decoder_bias: bind_tensor(gguf, TENSOR_DECODER_BIAS, &[punc_size])?,
        })
    }
}

// ---------------------------------------------------------------------------
// Numeric kernels (private; unit-tested against hand-computed expectations)
// ---------------------------------------------------------------------------

/// Row-wise LayerNorm over `[rows, d]`, biased variance, matching
/// `torch.nn.LayerNorm`.
fn layer_norm(x: &[f32], rows: usize, d: usize, gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * d);
    let mut y = vec![0.0_f32; x.len()];
    let dn = d as f32;
    for r in 0..rows {
        let row = &x[r * d..(r + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / dn;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dn;
        let inv = 1.0 / vokra_math::sqrt(var + eps);
        for j in 0..d {
            y[r * d + j] = (row[j] - mean) * inv * gamma[j] + beta[j];
        }
    }
    y
}

/// Row-major `y[i, o] = sum_k x[i, k] * w[o, k] + b[o]`.
///
/// `x` is `[rows, d_in]`, `w` is `[d_out, d_in]` (PyTorch `nn.Linear`
/// layout), `b` is `[d_out]`. Mirrors the naive triple loop in
/// `vokra-bert`'s `matmul_bias_rm` — correctness first.
fn linear(x: &[f32], w: &[f32], b: &[f32], rows: usize, d_in: usize, d_out: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * d_in);
    debug_assert_eq!(w.len(), d_out * d_in);
    debug_assert_eq!(b.len(), d_out);
    let mut y = vec![0.0_f32; rows * d_out];
    for i in 0..rows {
        for o in 0..d_out {
            let mut acc = b[o];
            let wr = &w[o * d_in..(o + 1) * d_in];
            let xr = &x[i * d_in..(i + 1) * d_in];
            for (xv, wv) in xr.iter().zip(wr.iter()) {
                acc += xv * wv;
            }
            y[i * d_out + o] = acc;
        }
    }
    y
}

/// Numerically stable softmax over the last axis of `[rows, cols]`, in place.
fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), rows * cols);
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for v in row.iter_mut() {
            *v = vokra_math::exp(*v - max);
            sum += *v;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
}

/// `SinusoidalPositionEncoder` from `funasr/models/transformer/embedding.py`.
///
/// Returns `[t_len, depth]` row-major where
/// `enc[t][i]       = sin((t + 1) * inv[i])` for `i < depth / 2` and
/// `enc[t][depth/2 + i] = cos((t + 1) * inv[i])`,
/// with `inv[i] = exp(-i * ln(10000) / (depth / 2 - 1))`.
///
/// Two upstream details this reproduces exactly: positions start at **1**,
/// and the sine and cosine halves are **concatenated blocks**, not
/// interleaved pairs.
fn sinusoidal_position_encoding(t_len: usize, depth: usize) -> Result<Vec<f32>> {
    if depth < 4 || depth % 2 != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "ct_punc: sinusoidal position encoding needs an even depth >= 4 (upstream divides by \
             `depth / 2 - 1`), got {depth}"
        )));
    }
    let half = depth / 2;
    // log_timescale_increment = log(10000) / (depth / 2 - 1)
    let log_inc = vokra_math::log(10_000.0_f32) / ((half as f32) - 1.0);
    let mut inv = vec![0.0_f32; half];
    for (i, slot) in inv.iter_mut().enumerate() {
        *slot = vokra_math::exp((i as f32) * -log_inc);
    }
    let mut enc = vec![0.0_f32; t_len * depth];
    for t in 0..t_len {
        // Upstream: `positions = torch.arange(1, timesteps + 1)`.
        let pos = (t + 1) as f32;
        for (i, &iv) in inv.iter().enumerate() {
            let scaled = pos * iv;
            enc[t * depth + i] = vokra_math::sin(scaled);
            enc[t * depth + half + i] = vokra_math::cos(scaled);
        }
    }
    Ok(enc)
}

/// `MultiHeadedAttentionSANM.forward_fsmn` without the padding mask.
///
/// `v` is `[t_len, d]`; `w` is the depthwise kernel `[d, 1, k]` flattened.
/// Returns `depthwise_conv1d(pad(v)) + v`, i.e. the memory branch output
/// *including* its residual, exactly as upstream (`x += inputs`).
fn fsmn_memory(
    v: &[f32],
    w: &[f32],
    t_len: usize,
    d: usize,
    kernel: usize,
    left: usize,
    right: usize,
) -> Vec<f32> {
    debug_assert_eq!(v.len(), t_len * d);
    debug_assert_eq!(w.len(), d * kernel);
    debug_assert_eq!(left + right, kernel - 1);
    let mut out = vec![0.0_f32; t_len * d];
    for t in 0..t_len {
        for c in 0..d {
            let mut acc = 0.0_f32;
            for j in 0..kernel {
                // Padded index: position `t + j` in the padded signal maps
                // back to `t + j - left` in `v`; anything outside is the
                // ConstantPad1d zero.
                let src = (t + j) as isize - left as isize;
                if src >= 0 && (src as usize) < t_len {
                    acc += v[(src as usize) * d + c] * w[c * kernel + j];
                }
            }
            // Upstream `x += inputs` — the memory branch carries its own
            // residual before it is added to the attention output.
            out[t * d + c] = acc + v[t * d + c];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CtPunc
// ---------------------------------------------------------------------------

/// A bound CT-Punc model.
///
/// Load with [`from_gguf`](CtPunc::from_gguf) / [`open`](CtPunc::open), then
/// call [`predict_labels`](CtPunc::predict_labels) on token ids, or
/// [`restore`](CtPunc::restore) to get a punctuated string back.
#[derive(Debug, Clone)]
pub struct CtPunc {
    cfg: CtPuncConfig,
    labels: Vec<String>,
    weights: CtPuncWeights,
    license_class: LicenseClass,
}

impl CtPunc {
    /// Binds a CT-Punc model from an already-parsed GGUF.
    ///
    /// Verification order is deliberate: arch first (so a foreign artifact
    /// fails with a *specific* message rather than a downstream
    /// missing-tensor error), then the label inventory, then the topology,
    /// then the tensors.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] if `vokra.model.arch` is absent or is not
    /// [`ARCH`], if the label inventory is absent / empty / not a string
    /// array, if `sentence_end_id` is outside it, if any `vokra.ct_punc.*`
    /// axis is missing or zero, or if any required tensor is missing or
    /// wrongly shaped.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // --- 1. Strict arch verification -------------------------------
        match gguf.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "ct_punc: GGUF `{}` is `{other}`, expected `{ARCH}`. CT-Punc is a SANM \
                     encoder over TEXT tokens with a per-token punctuation head; it is NOT \
                     `bert_base` (post-norm, learned absolute positions, separate q/k/v \
                     projections, no FSMN memory branch), NOT `deberta_v2` / `deberta_v3` \
                     (disentangled relative-position attention), NOT `sensevoicesmall` (SAN-M \
                     too, but a speech encoder over fbank frames with four per-task heads) and \
                     NOT `fsmn-vad` (FSMN memory blocks but no self-attention, 2-class frame \
                     output). Loading a foreign checkpoint here would bind a wrong-topology \
                     forward (FR-EX-08). Primary source: {PRIMARY_SOURCE_HF}",
                    chunks::KEY_MODEL_ARCH
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "ct_punc: GGUF has no `{}` metadata — refusing to guess the architecture. \
                     Re-run `vokra-cli convert --model ct-punc`.",
                    chunks::KEY_MODEL_ARCH
                )));
            }
        }

        // --- 2. Label inventory (read, never hardcoded) ------------------
        let labels = read_punc_labels(gguf)?;

        // --- 3. Topology -------------------------------------------------
        let cfg = CtPuncConfig::from_gguf(gguf)?;
        if cfg.sentence_end_id >= labels.len() {
            return Err(VokraError::ModelLoad(format!(
                "ct_punc: `{KEY_SENTENCE_END_ID}` is {} but the label inventory has only {} \
                 entries ({labels:?}) — a sentence-end index outside the head's own columns \
                 cannot be honoured (FR-EX-08)",
                cfg.sentence_end_id,
                labels.len()
            )));
        }

        // --- 4. Tensors ---------------------------------------------------
        let weights = CtPuncWeights::from_gguf(gguf, &cfg, labels.len())?;

        let license_class = gguf
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            cfg,
            labels,
            weights,
            license_class,
        })
    }

    /// Opens and binds the model from a GGUF file on disk.
    ///
    /// # Errors
    ///
    /// Propagates the GGUF reader's error, or any
    /// [`from_gguf`](Self::from_gguf) failure.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// The bound topology.
    #[must_use]
    pub fn config(&self) -> &CtPuncConfig {
        &self.cfg
    }

    /// The punctuation label inventory **as read off the artifact**, in
    /// decoder-head order.
    ///
    /// For `funasr/ct-punc` this is
    /// `["<unk>", "_", "，", "。", "？", "、"]`, where `"_"` means "emit no
    /// punctuation after this token" — but that is a property of *that*
    /// checkpoint, not of this code, which is exactly why it is read rather
    /// than hardcoded.
    #[must_use]
    pub fn punc_labels(&self) -> &[String] {
        &self.labels
    }

    /// The weight-licence class stamped on the artifact
    /// ([`LicenseClass::Unknown`] when absent — fail-closed).
    #[must_use]
    pub fn license_class(&self) -> LicenseClass {
        self.license_class
    }

    /// The bound weights (exposed for parity harnesses).
    #[must_use]
    pub fn weights(&self) -> &CtPuncWeights {
        &self.weights
    }

    /// Runs the encoder and returns raw logits, `[token_ids.len(),
    /// punc_labels().len()]` row-major.
    ///
    /// Softmax is deliberately **not** applied — `argmax` is invariant under
    /// it, and a consumer that wants probabilities can do it itself (the
    /// same output contract the sibling classification binders use).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `token_ids` is empty or contains
    /// an id outside `0..vocab_size`. Both are loud: an out-of-range id
    /// silently clamped to row 0 would emit confident nonsense.
    pub fn logits(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let t_len = token_ids.len();
        if t_len == 0 {
            return Err(VokraError::InvalidArgument(
                "ct_punc: token_ids is empty — there is nothing to punctuate".to_owned(),
            ));
        }
        let d = self.cfg.att_unit;
        if self.cfg.embed_unit != d {
            return Err(VokraError::InvalidArgument(format!(
                "ct_punc: embed_unit ({}) != att_unit ({d}); the upstream SANM encoder feeds the \
                 embedding straight into the block stack with `input_layer = \"pe\"` (no \
                 projection layer), so a decoupled pair needs an explicit projection this \
                 checkpoint does not carry",
                self.cfg.embed_unit
            )));
        }

        // ---- 1. embedding lookup ------------------------------------------
        let mut x = vec![0.0_f32; t_len * d];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id >= self.cfg.vocab_size {
                return Err(VokraError::InvalidArgument(format!(
                    "ct_punc: token id {id} at position {t} is outside the {} entry vocabulary — \
                     refusing to clamp (FR-EX-08)",
                    self.cfg.vocab_size
                )));
            }
            x[t * d..(t + 1) * d].copy_from_slice(&self.weights.embed[id * d..(id + 1) * d]);
        }

        // ---- 2. scale by sqrt(output_size), then 3. add the PE -------------
        // Upstream `SANMEncoder.forward`:
        //     xs_pad = xs_pad * self.output_size() ** 0.5
        //     xs_pad = self.embed(xs_pad)      # input_layer == "pe"
        let scale = vokra_math::sqrt(d as f32);
        for v in x.iter_mut() {
            *v *= scale;
        }
        let pe = sinusoidal_position_encoding(t_len, d)?;
        for (v, p) in x.iter_mut().zip(pe.iter()) {
            *v += *p;
        }

        // ---- 4. encoder blocks ---------------------------------------------
        let (left, right) = self.cfg.fsmn_padding();
        for block in &self.weights.blocks {
            // x = x + self_attn(norm1(x))
            let h = layer_norm(
                &x,
                t_len,
                d,
                &block.norm1_weight,
                &block.norm1_bias,
                self.cfg.layer_norm_eps,
            );
            let attn = self.sanm_attention(&h, block, t_len, left, right);
            for (dst, src) in x.iter_mut().zip(attn.iter()) {
                *dst += *src;
            }

            // x = x + w_2(relu(w_1(norm2(x))))
            let h = layer_norm(
                &x,
                t_len,
                d,
                &block.norm2_weight,
                &block.norm2_bias,
                self.cfg.layer_norm_eps,
            );
            let mut hidden = linear(
                &h,
                &block.ffn_w1_weight,
                &block.ffn_w1_bias,
                t_len,
                d,
                self.cfg.linear_units,
            );
            for v in hidden.iter_mut() {
                // `PositionwiseFeedForward` defaults to `torch.nn.ReLU()`.
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            let ffn = linear(
                &hidden,
                &block.ffn_w2_weight,
                &block.ffn_w2_bias,
                t_len,
                self.cfg.linear_units,
                d,
            );
            for (dst, src) in x.iter_mut().zip(ffn.iter()) {
                *dst += *src;
            }
        }

        // ---- 5. after_norm --------------------------------------------------
        let x = layer_norm(
            &x,
            t_len,
            d,
            &self.weights.after_norm_weight,
            &self.weights.after_norm_bias,
            self.cfg.layer_norm_eps,
        );

        // ---- 6. punctuation head --------------------------------------------
        Ok(linear(
            &x,
            &self.weights.decoder_weight,
            &self.weights.decoder_bias,
            t_len,
            d,
            self.labels.len(),
        ))
    }

    /// `MultiHeadedAttentionSANM.forward` for one block, single unbatched
    /// sequence (every position valid, so the padding mask is a no-op).
    fn sanm_attention(
        &self,
        h: &[f32],
        block: &CtPuncBlock,
        t_len: usize,
        left: usize,
        right: usize,
    ) -> Vec<f32> {
        let d = self.cfg.att_unit;
        let heads = self.cfg.attention_heads;
        let d_k = self.cfg.d_k();

        // One fused projection to 3D, then split into q | k | v.
        let qkv = linear(h, &block.qkv_weight, &block.qkv_bias, t_len, d, 3 * d);
        let mut q = vec![0.0_f32; t_len * d];
        let mut k = vec![0.0_f32; t_len * d];
        let mut v = vec![0.0_f32; t_len * d];
        for t in 0..t_len {
            let row = &qkv[t * 3 * d..(t + 1) * 3 * d];
            q[t * d..(t + 1) * d].copy_from_slice(&row[0..d]);
            k[t * d..(t + 1) * d].copy_from_slice(&row[d..2 * d]);
            v[t * d..(t + 1) * d].copy_from_slice(&row[2 * d..3 * d]);
        }

        // The parallel FSMN memory branch runs off `v` BEFORE the head split.
        let fsmn = fsmn_memory(
            &v,
            &block.fsmn_weight,
            t_len,
            d,
            self.cfg.kernel_size,
            left,
            right,
        );

        // `q_h = q_h * self.d_k ** (-0.5)` — upstream scales q, not the
        // scores; algebraically identical, kept in the same place so a
        // future numeric-parity diff lines up term for term.
        let inv_sqrt_dk = 1.0 / vokra_math::sqrt(d_k as f32);
        for val in q.iter_mut() {
            *val *= inv_sqrt_dk;
        }

        let mut ctx = vec![0.0_f32; t_len * d];
        let mut scores = vec![0.0_f32; t_len * t_len];
        for hd in 0..heads {
            let base = hd * d_k;
            for t1 in 0..t_len {
                for t2 in 0..t_len {
                    let mut acc = 0.0_f32;
                    for j in 0..d_k {
                        acc += q[t1 * d + base + j] * k[t2 * d + base + j];
                    }
                    scores[t1 * t_len + t2] = acc;
                }
            }
            softmax_rows(&mut scores, t_len, t_len);
            for t1 in 0..t_len {
                for j in 0..d_k {
                    let mut acc = 0.0_f32;
                    for t2 in 0..t_len {
                        acc += scores[t1 * t_len + t2] * v[t2 * d + base + j];
                    }
                    ctx[t1 * d + base + j] = acc;
                }
            }
        }

        let mut out = linear(&ctx, &block.out_weight, &block.out_bias, t_len, d, d);
        // `return att_outs + fsmn_memory`
        for (dst, src) in out.iter_mut().zip(fsmn.iter()) {
            *dst += *src;
        }
        out
    }

    /// Runs the forward and returns one label index per token
    /// (`argmax` over the head, matching upstream's `y.topk(1, dim=1)`).
    ///
    /// Each returned index is a valid index into [`punc_labels`](Self::punc_labels).
    ///
    /// # Errors
    ///
    /// As [`logits`](Self::logits).
    pub fn predict_labels(&self, token_ids: &[u32]) -> Result<Vec<usize>> {
        let p = self.labels.len();
        let logits = self.logits(token_ids)?;
        Ok(logits
            .chunks_exact(p)
            .map(|row| {
                let mut best = 0usize;
                for (i, v) in row.iter().enumerate() {
                    if *v > row[best] {
                        best = i;
                    }
                }
                best
            })
            .collect())
    }

    /// Joins `tokens` with the punctuation selected by `labels`.
    ///
    /// This is the token-joining half of upstream
    /// `CTTransformer.inference`, transcribed for the single-window case:
    ///
    /// - a token whose first character is ASCII is capitalised when it
    ///   starts the utterance or follows a `。` / `？`;
    /// - a space is inserted before an ASCII-initial token when the previous
    ///   token is also ASCII-initial (and before the very first one);
    /// - the `"_"` label emits nothing; any other label emits its character,
    ///   ASCII-folded (`，`→`,`, `。`→`.`, `？`→`?`) when the token it
    ///   follows is ASCII-initial;
    /// - finally the utterance is forced to end in a sentence terminator: a
    ///   trailing `，`/`、` becomes `。`, a trailing `,` becomes `.`, and an
    ///   otherwise-unterminated utterance gets `。` or `.` depending on
    ///   whether its last character is ASCII.
    ///
    /// Upstream's cross-window cache policy (`split_size = 20`, re-cutting
    /// at the last `。`/`？`) is a caller concern — see the module docstring.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `tokens` and `labels` differ in
    /// length, or if a label is outside the inventory.
    pub fn restore_with_labels(&self, tokens: &[&str], labels: &[usize]) -> Result<String> {
        if tokens.len() != labels.len() {
            return Err(VokraError::InvalidArgument(format!(
                "ct_punc: {} tokens but {} labels — the punctuation head emits exactly one label \
                 per token",
                tokens.len(),
                labels.len()
            )));
        }
        if tokens.is_empty() {
            return Ok(String::new());
        }
        for (i, &l) in labels.iter().enumerate() {
            if l >= self.labels.len() {
                return Err(VokraError::InvalidArgument(format!(
                    "ct_punc: label {l} at position {i} is outside the {}-entry inventory",
                    self.labels.len()
                )));
            }
        }

        let mut out = String::new();
        for (i, tok) in tokens.iter().enumerate() {
            let prev_terminates = i > 0 && {
                let p = self.labels[labels[i - 1]].as_str();
                p == "。" || p == "？"
            };
            let mut word = (*tok).to_owned();
            if (i == 0 || prev_terminates) && ascii_initial(word.as_str()) {
                // Python `str.capitalize()`: upper first char, lower the rest.
                let capitalized = {
                    let mut ch = word.chars();
                    ch.next().map(|first| {
                        let rest: String = ch.flat_map(char::to_lowercase).collect();
                        first.to_uppercase().collect::<String>() + &rest
                    })
                };
                if let Some(c) = capitalized {
                    word = c;
                }
            }
            if ascii_initial(word.as_str()) && (i == 0 || ascii_initial(tokens[i - 1])) {
                out.push(' ');
            }
            out.push_str(&word);

            let label = self.labels[labels[i]].as_str();
            if label != "_" {
                let folded = if ascii_initial(word.as_str()) {
                    match label {
                        "，" => ",",
                        "。" => ".",
                        "？" => "?",
                        other => other,
                    }
                } else {
                    label
                };
                out.push_str(folded);
            }
        }

        // Force a sentence terminator at the end of the utterance.
        let last = out.chars().next_back();
        match last {
            Some('，') | Some('、') => {
                out.pop();
                out.push('。');
            }
            Some(',') => {
                out.pop();
                out.push('.');
            }
            Some(c) if c.len_utf8() != 1 && c != '。' && c != '？' => out.push('。'),
            Some(c) if c.len_utf8() == 1 && c != '.' && c != '?' => out.push('.'),
            _ => {}
        }
        Ok(out)
    }

    /// Convenience: runs the forward over `token_ids` and joins `tokens`
    /// with the predicted punctuation.
    ///
    /// `tokens` must be the *same* sequence `token_ids` encodes (the
    /// tokenizer is out of scope — see the module docstring), which is why
    /// the two are passed separately rather than derived from each other.
    ///
    /// # Errors
    ///
    /// As [`logits`](Self::logits) and
    /// [`restore_with_labels`](Self::restore_with_labels).
    pub fn restore(&self, tokens: &[&str], token_ids: &[u32]) -> Result<String> {
        let labels = self.predict_labels(token_ids)?;
        self.restore_with_labels(tokens, &labels)
    }
}

/// Whether `s` starts with a character that encodes to a single UTF-8 byte.
///
/// This is upstream's `len(token[0].encode()) == 1` test, which it uses to
/// tell "a Latin word that wants spaces, capitalisation and ASCII
/// punctuation" apart from "a CJK token that wants none of that".
fn ascii_initial(s: &str) -> bool {
    s.chars().next().is_some_and(|c| c.len_utf8() == 1)
}

/// Reads `vokra.ct_punc.punc_list` as an ordered `Vec<String>`.
fn read_punc_labels(gguf: &GgufFile) -> Result<Vec<String>> {
    let arr = gguf
        .get(KEY_PUNC_LIST)
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "ct_punc: GGUF is missing the `{KEY_PUNC_LIST}` Array<String> chunk. The \
                 punctuation label inventory is READ from the artifact, never hardcoded — a \
                 checkpoint with a different label set means different head columns, and \
                 guessing them would mis-punctuate every output (FR-EX-08). Re-run \
                 `vokra-cli convert --model ct-punc`."
            ))
        })?;
    if arr.values.is_empty() {
        return Err(VokraError::ModelLoad(format!(
            "ct_punc: `{KEY_PUNC_LIST}` is empty — a punctuation head with zero columns cannot \
             classify anything"
        )));
    }
    let mut labels = Vec::with_capacity(arr.values.len());
    for (i, v) in arr.values.iter().enumerate() {
        let s = v.as_str().ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "ct_punc: `{KEY_PUNC_LIST}` entry {i} is not a string — the inventory must be an \
                 Array<String> in decoder-head order"
            ))
        })?;
        labels.push(s.to_owned());
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType};

    // Small but structurally faithful topology: D must be divisible by the
    // head count, and >= 4 and even for the position encoder.
    const D: usize = 8;
    const HEADS: usize = 2;
    const FF: usize = 6;
    const VOCAB: usize = 5;
    const K: usize = 3;
    const BLOCKS: usize = 2;
    const EPS: f32 = 1e-12;
    const LABELS: [&str; 6] = ["<unk>", "_", "，", "。", "？", "、"];

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn string_array(values: &[&str]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: values
                .iter()
                .map(|s| GgufMetadataValue::String((*s).to_owned()))
                .collect(),
        })
    }

    /// Deterministic pseudo-random weights (SplitMix64 — the
    /// `LlmWeights::synthesized` pattern; no external rng dep, NFR-DS-02).
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Uniform in [-0.5, 0.5).
            ((z >> 40) as f32) / 16_777_216.0 - 0.5
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
    }

    /// How the block LayerNorms are filled — the lever the analytic test uses.
    #[derive(Clone, Copy, PartialEq)]
    enum BlockNorms {
        /// gamma = 0, beta = 0. `norm1(x)` and `norm2(x)` are then exactly
        /// zero, so with zero biases every block collapses to the identity
        /// and the whole encoder becomes computable by hand.
        Zeroed,
        /// gamma = 1, beta = 0 — the attention + FFN paths actually fire.
        Identity,
    }

    /// Builds a synthetic CT-Punc GGUF.
    ///
    /// With `BlockNorms::Zeroed` every block bias is also zeroed, which is
    /// what makes the encoder stack provably the identity.
    fn build_gguf(norms: BlockNorms, mutate: impl FnOnce(&mut GgufBuilder)) -> GgufFile {
        build_gguf_with(norms, D, mutate)
    }

    /// `decoder_dim` lets a test emit a deliberately wrong-width
    /// classification head. It has to be a build-time parameter rather than
    /// something the `mutate` hook overwrites, because `GgufBuilder::
    /// add_tensor` refuses a duplicate name with `GgufError::DuplicateTensor`
    /// (unlike the metadata setters, which replace in place).
    fn build_gguf_with(
        norms: BlockNorms,
        decoder_dim: usize,
        mutate: impl FnOnce(&mut GgufBuilder),
    ) -> GgufFile {
        let mut rng = Rng(0x5EED_1234_ABCD_0001);
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        );
        b.add_u32(KEY_VOCAB_SIZE, VOCAB as u32);
        b.add_u32(KEY_EMBED_UNIT, D as u32);
        b.add_u32(KEY_ATT_UNIT, D as u32);
        b.add_u32(KEY_ATTENTION_HEADS, HEADS as u32);
        b.add_u32(KEY_LINEAR_UNITS, FF as u32);
        b.add_u32(KEY_NUM_BLOCKS, BLOCKS as u32);
        b.add_u32(KEY_KERNEL_SIZE, K as u32);
        b.add_u32(KEY_SANM_SHIFT, 0);
        b.add_u32(KEY_SENTENCE_END_ID, 3);
        b.add_f32(KEY_LAYER_NORM_EPS, EPS);
        b.add_metadata(KEY_PUNC_LIST, string_array(&LABELS));

        let add = |b: &mut GgufBuilder, name: &str, dims: Vec<u64>, data: &[f32]| {
            b.add_tensor(name, GgmlType::F32, dims, f32_bytes(data))
                .expect("add tensor");
        };

        let embed = rng.vec(VOCAB * D);
        add(&mut b, TENSOR_EMBED, vec![VOCAB as u64, D as u64], &embed);

        for i in 0..BLOCKS {
            let p = block_prefix(i);
            let (nw, nb) = match norms {
                BlockNorms::Zeroed => (vec![0.0_f32; D], vec![0.0_f32; D]),
                BlockNorms::Identity => (vec![1.0_f32; D], vec![0.0_f32; D]),
            };
            add(&mut b, &format!("{p}.norm1.weight"), vec![D as u64], &nw);
            add(&mut b, &format!("{p}.norm1.bias"), vec![D as u64], &nb);
            add(&mut b, &format!("{p}.norm2.weight"), vec![D as u64], &nw);
            add(&mut b, &format!("{p}.norm2.bias"), vec![D as u64], &nb);

            // Under `Zeroed`, biases must be zero too: the identity proof
            // needs `linear(0) == 0`, and a non-zero bias would break it.
            let zero_bias = norms == BlockNorms::Zeroed;
            let qkv_b = if zero_bias {
                vec![0.0_f32; 3 * D]
            } else {
                rng.vec(3 * D)
            };
            let out_b = if zero_bias {
                vec![0.0_f32; D]
            } else {
                rng.vec(D)
            };
            let w1_b = if zero_bias {
                vec![0.0_f32; FF]
            } else {
                rng.vec(FF)
            };
            let w2_b = if zero_bias {
                vec![0.0_f32; D]
            } else {
                rng.vec(D)
            };

            let qkv_w = rng.vec(3 * D * D);
            let out_w = rng.vec(D * D);
            let fsmn_w = rng.vec(D * K);
            let w1_w = rng.vec(FF * D);
            let w2_w = rng.vec(D * FF);
            add(
                &mut b,
                &format!("{p}.self_attn.linear_q_k_v.weight"),
                vec![3 * D as u64, D as u64],
                &qkv_w,
            );
            add(
                &mut b,
                &format!("{p}.self_attn.linear_q_k_v.bias"),
                vec![3 * D as u64],
                &qkv_b,
            );
            add(
                &mut b,
                &format!("{p}.self_attn.linear_out.weight"),
                vec![D as u64, D as u64],
                &out_w,
            );
            add(
                &mut b,
                &format!("{p}.self_attn.linear_out.bias"),
                vec![D as u64],
                &out_b,
            );
            add(
                &mut b,
                &format!("{p}.self_attn.fsmn_block.weight"),
                vec![D as u64, 1, K as u64],
                &fsmn_w,
            );
            add(
                &mut b,
                &format!("{p}.feed_forward.w_1.weight"),
                vec![FF as u64, D as u64],
                &w1_w,
            );
            add(
                &mut b,
                &format!("{p}.feed_forward.w_1.bias"),
                vec![FF as u64],
                &w1_b,
            );
            add(
                &mut b,
                &format!("{p}.feed_forward.w_2.weight"),
                vec![D as u64, FF as u64],
                &w2_w,
            );
            add(
                &mut b,
                &format!("{p}.feed_forward.w_2.bias"),
                vec![D as u64],
                &w2_b,
            );
        }

        let an_w = vec![1.0_f32; D];
        let an_b = vec![0.0_f32; D];
        add(&mut b, TENSOR_AFTER_NORM_WEIGHT, vec![D as u64], &an_w);
        add(&mut b, TENSOR_AFTER_NORM_BIAS, vec![D as u64], &an_b);
        let dec_w = rng.vec(LABELS.len() * decoder_dim);
        let dec_b = rng.vec(LABELS.len());
        add(
            &mut b,
            TENSOR_DECODER_WEIGHT,
            vec![LABELS.len() as u64, decoder_dim as u64],
            &dec_w,
        );
        add(
            &mut b,
            TENSOR_DECODER_BIAS,
            vec![LABELS.len() as u64],
            &dec_b,
        );

        mutate(&mut b);
        GgufFile::parse(b.to_bytes().expect("serialize gguf")).expect("parse gguf")
    }

    fn model(norms: BlockNorms) -> CtPunc {
        CtPunc::from_gguf(&build_gguf(norms, |_| {})).expect("synthetic GGUF must bind")
    }

    // -----------------------------------------------------------------
    // Loading — arch gate
    // -----------------------------------------------------------------

    #[test]
    fn synthetic_gguf_binds_and_surfaces_config_labels_and_license() {
        let m = model(BlockNorms::Identity);
        let cfg = m.config();
        assert_eq!(cfg.vocab_size, VOCAB);
        assert_eq!(cfg.att_unit, D);
        assert_eq!(cfg.attention_heads, HEADS);
        assert_eq!(cfg.d_k(), D / HEADS);
        assert_eq!(cfg.num_blocks, BLOCKS);
        assert_eq!(cfg.kernel_size, K);
        assert_eq!(cfg.sanm_shift, 0);
        assert_eq!(cfg.sentence_end_id, 3);
        // kernel 3, shift 0 -> left = 1, right = 1.
        assert_eq!(cfg.fsmn_padding(), (1, 1));
        // Read off the artifact, in order.
        assert_eq!(m.punc_labels(), LABELS.as_slice());
        assert_eq!(m.license_class(), LicenseClass::Permissive);
        assert_eq!(m.weights().blocks.len(), BLOCKS);
    }

    #[test]
    fn missing_arch_is_a_loud_model_load_error() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        let g = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        // let-else rather than `unwrap_err()`: it works regardless of
        // whether the Ok type implements `Debug`, which is the standing
        // pattern for model structs in this crate.
        let Err(err) = CtPunc::from_gguf(&g) else {
            panic!("expected an error when vokra.model.arch is absent");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("no `vokra.model.arch`"),
            "must say the arch stamp is absent, got: {msg}"
        );
    }

    #[test]
    fn foreign_arch_names_both_expected_and_actual() {
        let g = build_gguf(BlockNorms::Identity, |b| {
            // Overwrite with a sibling that shares part of the machinery —
            // the most dangerous mis-route, since `fsmn-vad` also has FSMN
            // memory blocks.
            b.add_string(chunks::KEY_MODEL_ARCH, "fsmn-vad");
        });
        let Err(err) = CtPunc::from_gguf(&g) else {
            panic!("expected an error when the GGUF carries a foreign arch");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("fsmn-vad"),
            "must name the ACTUAL arch, got: {msg}"
        );
        assert!(
            msg.contains(ARCH),
            "must name the EXPECTED arch `{ARCH}`, got: {msg}"
        );
        assert!(
            msg.contains("bert_base") && msg.contains("sensevoicesmall"),
            "must enumerate the sibling families it is NOT, got: {msg}"
        );
    }

    // -----------------------------------------------------------------
    // Loading — metadata gates
    // -----------------------------------------------------------------

    #[test]
    fn missing_punc_list_is_refused_rather_than_defaulted() {
        // Rebuild without the label inventory: a builder cannot unset a
        // key, so construct a minimal GGUF that has the arch but no list.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        let g = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let Err(err) = CtPunc::from_gguf(&g) else {
            panic!("expected an error when the punctuation inventory is absent");
        };
        let msg = err.to_string();
        assert!(
            msg.contains(KEY_PUNC_LIST),
            "must name the missing inventory key, got: {msg}"
        );
        assert!(
            msg.contains("never hardcoded"),
            "must state that the inventory is read, not assumed, got: {msg}"
        );
    }

    #[test]
    fn zero_sentinel_axis_is_refused() {
        let g = build_gguf(BlockNorms::Identity, |b| {
            b.add_u32(KEY_ATT_UNIT, 0);
        });
        let Err(err) = CtPunc::from_gguf(&g) else {
            panic!("expected an error when att_unit is a 0 sentinel");
        };
        let msg = err.to_string();
        assert!(msg.contains(KEY_ATT_UNIT), "must name the axis, got: {msg}");
    }

    #[test]
    fn att_unit_not_divisible_by_heads_is_refused() {
        let g = build_gguf(BlockNorms::Identity, |b| {
            b.add_u32(KEY_ATTENTION_HEADS, 3); // D = 8 is not divisible by 3
        });
        let Err(err) = CtPunc::from_gguf(&g) else {
            panic!("expected an error when att_unit % attention_heads != 0");
        };
        assert!(
            err.to_string().contains("divisible"),
            "must name the divisibility constraint, got: {err}"
        );
    }

    #[test]
    fn sentence_end_id_outside_the_inventory_is_refused() {
        let g = build_gguf(BlockNorms::Identity, |b| {
            b.add_u32(KEY_SENTENCE_END_ID, LABELS.len() as u32);
        });
        let Err(err) = CtPunc::from_gguf(&g) else {
            panic!("expected an error when sentence_end_id is out of range");
        };
        assert!(
            err.to_string().contains(KEY_SENTENCE_END_ID),
            "must name sentence_end_id, got: {err}"
        );
    }

    // -----------------------------------------------------------------
    // Loading — tensor gates
    // -----------------------------------------------------------------

    #[test]
    fn missing_tensor_names_the_tensor() {
        // Build a GGUF with all metadata but no tensors at all: the first
        // required tensor must be named.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(KEY_VOCAB_SIZE, VOCAB as u32);
        b.add_u32(KEY_EMBED_UNIT, D as u32);
        b.add_u32(KEY_ATT_UNIT, D as u32);
        b.add_u32(KEY_ATTENTION_HEADS, HEADS as u32);
        b.add_u32(KEY_LINEAR_UNITS, FF as u32);
        b.add_u32(KEY_NUM_BLOCKS, BLOCKS as u32);
        b.add_u32(KEY_KERNEL_SIZE, K as u32);
        b.add_u32(KEY_SANM_SHIFT, 0);
        b.add_u32(KEY_SENTENCE_END_ID, 3);
        b.add_f32(KEY_LAYER_NORM_EPS, EPS);
        b.add_metadata(KEY_PUNC_LIST, string_array(&LABELS));
        let g = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");

        let Err(err) = CtPunc::from_gguf(&g) else {
            panic!("expected an error when every weight tensor is absent");
        };
        assert!(
            err.to_string().contains(TENSOR_EMBED),
            "must name the missing tensor, got: {err}"
        );
    }

    #[test]
    fn wrong_shaped_tensor_names_expected_and_actual() {
        // A decoder head one column too narrow for the stamped `att_unit`.
        let g = build_gguf_with(BlockNorms::Identity, D - 1, |_| {});
        let Err(err) = CtPunc::from_gguf(&g) else {
            panic!("expected an error when decoder.weight is wrongly shaped");
        };
        let msg = err.to_string();
        assert!(
            msg.contains(TENSOR_DECODER_WEIGHT),
            "must name the tensor, got: {msg}"
        );
        assert!(
            msg.contains(&(LABELS.len() * (D - 1)).to_string())
                && msg.contains(&(LABELS.len() * D).to_string()),
            "must name BOTH the actual and required element counts, got: {msg}"
        );
    }

    // -----------------------------------------------------------------
    // Numeric kernels — hand-computed expectations
    // -----------------------------------------------------------------

    /// Pins the three easy-to-get-wrong PE details: 1-based positions,
    /// `[sin | cos]` block layout, and the `log(10000) / (depth/2 - 1)`
    /// timescale.
    #[test]
    fn sinusoidal_position_encoding_matches_the_upstream_formula() {
        const DEPTH: usize = 4;
        let enc = sinusoidal_position_encoding(2, DEPTH).expect("depth 4 is valid");
        assert_eq!(enc.len(), 2 * DEPTH);

        // half = 2, so log_inc = ln(10000) / 1, inv = [1, 1e-4].
        let inv: [f32; 2] = [1.0, 1.0e-4];
        for (t, pos) in [(0usize, 1.0_f32), (1usize, 2.0_f32)] {
            for (i, &iv) in inv.iter().enumerate() {
                let scaled = pos * iv;
                let want_sin = vokra_math::sin(scaled);
                let want_cos = vokra_math::cos(scaled);
                assert!(
                    (enc[t * DEPTH + i] - want_sin).abs() < 1e-5,
                    "sin half at t={t} i={i}: got {} want {want_sin}",
                    enc[t * DEPTH + i]
                );
                assert!(
                    (enc[t * DEPTH + 2 + i] - want_cos).abs() < 1e-5,
                    "cos half at t={t} i={i}: got {} want {want_cos}",
                    enc[t * DEPTH + 2 + i]
                );
            }
        }
        // Position 0 must NOT be sin(0) = 0 — upstream is 1-based.
        assert!(
            enc[0].abs() > 0.5,
            "first token's first sine must be sin(1) ~= 0.8415, not sin(0) = 0; got {}",
            enc[0]
        );
    }

    #[test]
    fn sinusoidal_position_encoding_refuses_an_odd_or_tiny_depth() {
        assert!(sinusoidal_position_encoding(2, 5).is_err(), "odd depth");
        assert!(sinusoidal_position_encoding(2, 2).is_err(), "depth < 4");
    }

    /// Hand-computed depthwise FSMN convolution: one channel, kernel 3,
    /// left = right = 1, so `out[t] = w0*v[t-1] + w1*v[t] + w2*v[t+1] + v[t]`
    /// with zero padding at the edges.
    #[test]
    fn fsmn_memory_matches_a_hand_computed_depthwise_convolution() {
        let v = [1.0_f32, 2.0, 3.0];
        let kernel = [0.5_f32, -1.0, 2.0];
        let got = fsmn_memory(&v, &kernel, 3, 1, 3, 1, 1);
        // Each term is written as `kernel_tap * input_sample` so the reader can
        // check it against the convolution in the doc comment above. Folding
        // `-1.0 * x` to `-x` would save characters and lose that
        // correspondence, which is the entire point of a hand-computed oracle.
        #[allow(
            clippy::neg_multiply,
            reason = "each term mirrors kernel_tap * input_sample from the doc comment"
        )]
        let want = [
            0.5 * 0.0 + -1.0 * 1.0 + 2.0 * 2.0 + 1.0, // t=0
            0.5 * 1.0 + -1.0 * 2.0 + 2.0 * 3.0 + 2.0, // t=1
            0.5 * 2.0 + -1.0 * 3.0 + 2.0 * 0.0 + 3.0, // t=2
        ];
        for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - e).abs() < 1e-6, "t={i}: got {g} want {e}");
        }
    }

    /// The memory branch must be depthwise: channel 1's kernel may not
    /// touch channel 0's samples.
    #[test]
    fn fsmn_memory_does_not_mix_channels() {
        // Two channels, T = 2, kernel 1 (left = right = 0) so the answer is
        // simply `v[t][c] * w[c] + v[t][c]`.
        let v = [1.0_f32, 10.0, 2.0, 20.0]; // [[1, 10], [2, 20]]
        let w = [3.0_f32, 0.0]; // channel 0 scales by 3, channel 1 by 0
        let got = fsmn_memory(&v, &w, 2, 2, 1, 0, 0);
        assert!((got[0] - 4.0).abs() < 1e-6, "c0 t0: {}", got[0]);
        assert!((got[1] - 10.0).abs() < 1e-6, "c1 t0 untouched: {}", got[1]);
        assert!((got[2] - 8.0).abs() < 1e-6, "c0 t1: {}", got[2]);
        assert!((got[3] - 20.0).abs() < 1e-6, "c1 t1 untouched: {}", got[3]);
    }

    #[test]
    fn softmax_rows_normalises_each_row() {
        let mut x = [1.0_f32, 2.0, 3.0, 0.0, 0.0, 0.0];
        softmax_rows(&mut x, 2, 3);
        for r in 0..2 {
            let s: f32 = x[r * 3..(r + 1) * 3].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {r} sums to {s}");
        }
        // Uniform input -> uniform output.
        for v in &x[3..6] {
            assert!((v - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    // -----------------------------------------------------------------
    // Forward — analytic end-to-end check
    // -----------------------------------------------------------------

    /// With every block LayerNorm zeroed (gamma = beta = 0) and every block
    /// bias zeroed, `norm1(x) == 0` and `norm2(x) == 0`, so
    /// `self_attn(0) == 0` and `w_2(relu(w_1(0))) == 0`: the entire encoder
    /// stack provably collapses to the identity. The expected logits are
    /// then `decoder(after_norm(sqrt(D) * embed[id] + pe))`, which this test
    /// computes independently of [`CtPunc::logits`].
    ///
    /// That exercises, for real: the embedding lookup, the `sqrt(att_unit)`
    /// scaling, the positional encoding, the residual structure of both
    /// sub-layers, `after_norm`, and the classification head.
    #[test]
    fn encoder_stack_is_identity_when_block_norms_are_zeroed_and_matches_hand_computation() {
        let g = build_gguf(BlockNorms::Zeroed, |_| {});
        let m = CtPunc::from_gguf(&g).expect("bind");
        let ids: [u32; 3] = [4, 0, 2];

        let got = m.logits(&ids).expect("forward");
        assert_eq!(got.len(), ids.len() * LABELS.len());

        // Independent recomputation.
        let w = m.weights();
        let scale = vokra_math::sqrt(D as f32);
        let mut x = vec![0.0_f32; ids.len() * D];
        for (t, &id) in ids.iter().enumerate() {
            for j in 0..D {
                x[t * D + j] = w.embed[id as usize * D + j] * scale;
            }
        }
        let pe = sinusoidal_position_encoding(ids.len(), D).expect("pe");
        for (v, p) in x.iter_mut().zip(pe.iter()) {
            *v += *p;
        }
        let normed = layer_norm(
            &x,
            ids.len(),
            D,
            &w.after_norm_weight,
            &w.after_norm_bias,
            EPS,
        );
        let want = linear(
            &normed,
            &w.decoder_weight,
            &w.decoder_bias,
            ids.len(),
            D,
            LABELS.len(),
        );

        for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "logit {i}: forward gave {g}, hand computation gave {e}"
            );
        }
    }

    /// The counterpart: with live block LayerNorms the attention + FFN
    /// paths must actually change the answer. If they did not, the previous
    /// test would be passing for the wrong reason (a no-op encoder).
    #[test]
    fn live_block_norms_change_the_logits() {
        let ids: [u32; 3] = [4, 0, 2];
        let zeroed = model(BlockNorms::Zeroed).logits(&ids).expect("forward");
        let live = model(BlockNorms::Identity).logits(&ids).expect("forward");
        assert_eq!(zeroed.len(), live.len());
        let max_delta = zeroed
            .iter()
            .zip(live.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_delta > 1e-3,
            "the SANM attention + FFN path must contribute; max |delta| was {max_delta}"
        );
    }

    #[test]
    fn predict_labels_returns_one_valid_label_per_token_and_is_deterministic() {
        let m = model(BlockNorms::Identity);
        let ids: [u32; 4] = [0, 1, 2, 3];
        let a = m.predict_labels(&ids).expect("predict");
        let b = m.predict_labels(&ids).expect("predict again");
        assert_eq!(a.len(), ids.len());
        assert_eq!(a, b, "the forward must be deterministic");
        for (i, &l) in a.iter().enumerate() {
            assert!(
                l < LABELS.len(),
                "label {l} at {i} is outside the {}-entry inventory",
                LABELS.len()
            );
        }
    }

    #[test]
    fn empty_and_out_of_range_token_ids_are_refused_loudly() {
        let m = model(BlockNorms::Identity);
        let Err(err) = m.logits(&[]) else {
            panic!("expected an error when token_ids is empty");
        };
        assert!(err.to_string().contains("empty"), "got: {err}");

        let Err(err) = m.logits(&[VOCAB as u32]) else {
            panic!("expected an error when a token id is out of range");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("outside") && msg.contains("vocabulary"),
            "must refuse rather than clamp, got: {msg}"
        );
    }

    // -----------------------------------------------------------------
    // Joining
    // -----------------------------------------------------------------

    #[test]
    fn restore_with_labels_inserts_cjk_punctuation_and_forces_a_terminator() {
        let m = model(BlockNorms::Identity);
        // "_" = 1, "，" = 2, "。" = 3.
        let out = m
            .restore_with_labels(&["我们", "今天", "讨论", "议题"], &[1, 2, 1, 3])
            .expect("join");
        assert_eq!(out, "我们今天，讨论议题。");
    }

    #[test]
    fn restore_with_labels_folds_punctuation_to_ascii_after_ascii_tokens() {
        let m = model(BlockNorms::Identity);
        // Upstream folds ，/。/？ to ,/./? when the token they follow is
        // ASCII-initial, spaces ASCII tokens apart, and capitalises the
        // first token and any token after a sentence end.
        let out = m
            .restore_with_labels(&["hello", "world", "bye"], &[2, 3, 1])
            .expect("join");
        assert_eq!(out, " Hello, world. Bye.");
    }

    #[test]
    fn restore_with_labels_rewrites_a_trailing_comma_into_a_full_stop() {
        let m = model(BlockNorms::Identity);
        // Trailing "，" must become "。" (upstream sentence-end fixup).
        let out = m.restore_with_labels(&["一", "二"], &[1, 2]).expect("join");
        assert_eq!(out, "一二。");
        // Trailing "、" takes the same path.
        let out = m.restore_with_labels(&["一", "二"], &[1, 5]).expect("join");
        assert_eq!(out, "一二。");
    }

    #[test]
    fn restore_with_labels_rejects_mismatched_lengths_and_bad_labels() {
        let m = model(BlockNorms::Identity);
        let Err(err) = m.restore_with_labels(&["a", "b"], &[1]) else {
            panic!("expected an error when token and label counts differ");
        };
        assert!(
            err.to_string().contains("one label per token"),
            "got: {err}"
        );

        let Err(err) = m.restore_with_labels(&["a"], &[LABELS.len()]) else {
            panic!("expected an error when a label is outside the inventory");
        };
        assert!(err.to_string().contains("outside"), "got: {err}");
    }

    #[test]
    fn restore_runs_the_forward_and_joins() {
        let m = model(BlockNorms::Identity);
        let ids: [u32; 3] = [1, 2, 3];
        let out = m.restore(&["甲", "乙", "丙"], &ids).expect("restore");
        // Whatever the synthetic weights predict, the output must contain
        // every token in order and end in a sentence terminator.
        assert!(out.contains('甲') && out.contains('乙') && out.contains('丙'));
        let last = out.chars().next_back().expect("non-empty");
        assert!(
            matches!(last, '。' | '？' | '.' | '?'),
            "the utterance must end in a sentence terminator, got {last:?} in {out:?}"
        );
    }

    // -----------------------------------------------------------------
    // Contract pins
    // -----------------------------------------------------------------

    #[test]
    fn block_prefix_follows_the_upstream_encoders0_split() {
        assert_eq!(block_prefix(0), "encoder.encoders0.0");
        assert_eq!(block_prefix(1), "encoder.encoders.0");
        assert_eq!(block_prefix(2), "encoder.encoders.1");
    }

    #[test]
    fn arch_tag_is_distinct_from_every_sibling_it_could_be_confused_with() {
        assert_eq!(ARCH, "ct_punc");
        assert_eq!(NAME, "ct-punc");
        assert_eq!(CATEGORY, "punctuation");
        for sibling in [
            "bert_base",
            "deberta_v2",
            "deberta_v3",
            "sensevoicesmall",
            "fsmn-vad",
            "w2v_bert2",
            "whisper",
        ] {
            assert_ne!(
                ARCH, sibling,
                "`{sibling}` is a different topology — sharing an arch tag would misroute \
                 runtime dispatch (FR-EX-08)"
            );
        }
    }

    #[test]
    fn fsmn_padding_follows_the_upstream_rule() {
        let mut cfg = model(BlockNorms::Identity).config().clone();
        // kernel 11, shift 0 (the real funasr/ct-punc config) -> (5, 5).
        cfg.kernel_size = 11;
        cfg.sanm_shift = 0;
        assert_eq!(cfg.fsmn_padding(), (5, 5));
        // A positive shift moves the window left, per
        // `if sanm_shfit > 0: left_padding = left_padding + sanm_shfit`.
        cfg.sanm_shift = 2;
        assert_eq!(cfg.fsmn_padding(), (7, 3));
        // Even kernels keep `left + right == kernel - 1`.
        cfg.kernel_size = 4;
        cfg.sanm_shift = 0;
        let (l, r) = cfg.fsmn_padding();
        assert_eq!(l + r, 3);
    }
}
