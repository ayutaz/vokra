//! **w2v-BERT 2.0** (`facebook/w2v-bert-2.0`, **MIT**) — runtime binder for
//! the `w2v-bert-2` converter arch (Wave C2 2026-08-15).
//!
//! Closes a real read-side gap: `crates/vokra-convert/src/models/w2v_bert_2.rs`
//! stamps `vokra.model.arch = "w2v-bert-2"`, but a workspace-wide grep proved
//! that **nothing read that arch string back** — a converted checkpoint was
//! unloadable. This module is that consumer.
//!
//! # What w2v-BERT 2.0 is
//!
//! The Meta / SeamlessM4T-v2 speech encoder: a ~580M-parameter self-supervised
//! **Conformer** encoder (Chung et al. 2021, arXiv:2108.06209 — contrastive
//! wav2vec-2.0 branch + BERT-style masked-language-modeling branch over a
//! shared Conformer body), released as part of the Seamless stack (Barrault
//! et al. 2023, arXiv:2312.05187). It is a **feature extractor, not an
//! end-task model**: it maps audio features to a sequence of hidden states
//! that a downstream ASR / AST / speaker / VAD head consumes. The upstream
//! release ships **no task head** — `architectures: ["Wav2Vec2BertModel"]`,
//! not `...ForCTC` — so this binder deliberately exposes hidden states and
//! **never invents a classification head the checkpoint does not contain**.
//!
//! # Standalone vs internal-subgraph identity
//!
//! w2v-BERT 2.0 tensors also appear inside two **composite** converters:
//! `unity-2` (SeamlessM4T-v2 uses w2v-BERT as its speech encoder) and
//! `vieneu-tts` (VieNeu TTS uses it as its speaker encoder). Neither exposes
//! the encoder for standalone use. This binder is the standalone path: a
//! downstream training a per-language head on w2v-BERT features binds the
//! shared encoder from a single GGUF instead of stripping it out of a
//! composite.
//!
//! # Runtime layout (upstream `Wav2Vec2BertModel`)
//!
//! ```text
//! 80-band log-mel fbank @ 16 kHz, stride-2 frame stacking -> [T, 160]
//!      (upstream `SeamlessM4TFeatureExtractor`; the front-end is the
//!       CALLER's concern — this binder consumes already-stacked features)
//!   -> feature_projection: LayerNorm(160) -> Linear(160, 1024)   <- **REAL**
//!        ([`W2vBert2::project_features`] — implemented here, exercised
//!         by a hand-computed numeric unit test)
//!   -> 24 x Wav2Vec2BertEncoderLayer                             <- **loud-partial**
//!        residual = x
//!        residual += 0.5 * ffn1(ffn1_layer_norm(residual))
//!        residual +=       self_attn(self_attn_layer_norm(residual))
//!        residual +=       conv_module(residual)
//!        residual += 0.5 * ffn2(ffn2_layer_norm(residual))
//!        out       = final_layer_norm(residual)
//!   -> [T, 1024] hidden states (no task head upstream)
//! ```
//!
//! # Why the encoder stack is a loud-partial, in spite of `vokra_ops::conformer`
//!
//! The task brief asked to wire this onto the shared
//! [`vokra_ops::conformer::ConformerEncoder`] primitive rather than
//! re-implementing Conformer blocks, and that primitive was read first. It
//! covers the *macaron* layer skeleton exactly (half-scale FF1 / MHA / conv
//! module / half-scale FF2 / `norm_out`, Swish activation, GLU over the
//! channel axis) — but a block-by-block diff against the upstream
//! `modeling_wav2vec2_bert.py` source turns up **four** concrete divergences,
//! each pinned verbatim in [`CONFORMER_PRIMITIVE_GAPS`]:
//!
//! 1. **relative_key attention bias.** `facebook/w2v-bert-2.0` sets
//!    `position_embeddings_type = "relative_key"`, i.e. a Shaw-style
//!    per-layer `distance_embedding` (`nn.Embedding(left_max + right_max + 1,
//!    head_size)`) gathered into the attention score matrix.
//!    [`vokra_ops::conformer::PositionEncoding`] exposes only `None` and
//!    `Rope`, and its own module docstring records that the relative path is
//!    "omitted from the primitive".
//! 2. **Causal, left-only depthwise padding.** Upstream pads
//!    `(kernel_size - 1, 0)` and then convolves with `padding=0`;
//!    `vokra_ops::conformer::conformer_conv` applies *symmetric*
//!    same-padding (`padding = kernel_size / 2`). Same kernel, different
//!    receptive field — a silent swap would shift every output frame.
//! 3. **Bias-free convolution module.** Upstream builds `pointwise_conv1`,
//!    `depthwise_conv` and `pointwise_conv2` with `bias=False`;
//!    [`vokra_ops::conformer::ConformerConvWeights`] requires
//!    `pointwise1_b` / `depthwise_b` / `pointwise2_b`.
//! 4. **Pre-projection stem LayerNorm.** Upstream normalises the *stacked
//!    input* and then projects (`layer_norm` -> `projection`);
//!    `vokra_ops::conformer::ConvSubsampleKind::StackingNorm` projects and
//!    *then* normalises.
//!
//! Composing the stack anyway — by passing zero biases and accepting
//! symmetric padding and dropping the relative bias — would produce
//! shape-valid, numerically wrong hidden states: exactly the silent-misroute
//! failure FR-EX-08 exists to prevent. So [`W2vBert2::encode`] loud-partials
//! and names all four gaps, while everything around it lands real. That is
//! the CLAUDE.md 教訓 (a) posture: 「loud-partial は fake-complete より
//! honest」. Closing it is mechanical once `vokra-ops` grows a relative-key
//! position encoding and a causal-padding switch on the conv module — both
//! additive changes to a primitive shared with the parakeet / canary fleet,
//! which is why they are deliberately NOT made from inside this binder.
//!
//! # Real in this WP
//!
//! - Strict `vokra.model.arch == "w2v-bert-2"` verification, refusing a
//!   foreign GGUF loudly with **both** tags named and the SSL / composite
//!   neighbourhood enumerated ([`W2vBert2::from_gguf`]).
//! - **Topology derived from the tensors actually on disk**
//!   ([`W2vBert2Config::from_gguf`]). The converter is a BF16 pass-through
//!   skeleton — it stamps *no* `vokra.w2v_bert_2.*` chunk group at all — so
//!   there is no metadata to read. Rather than fabricate axes from
//!   primary-source constants, every axis is recovered from real tensor
//!   shapes, which are ground truth on disk. Notably `num_attention_heads`
//!   *is* recoverable: `distance_embedding.weight` is
//!   `[num_positions, head_size]`, and `head_size = hidden_size / heads`.
//! - Real tensor-name binding with per-layer presence checks over the
//!   verbatim upstream `state_dict` names; a missing tensor is a loud
//!   [`VokraError::ModelLoad`] **naming that tensor**.
//! - A real `feature_projection` forward ([`W2vBert2::project_features`]).
//! - Weight-license surfacing, fail-closed to [`LicenseClass::Unknown`].
//!
//! # Primary sources
//!
//! - HF release: <https://huggingface.co/facebook/w2v-bert-2.0>
//!   (`license: mit`; `config.json` and `preprocessor_config.json` both
//!   fetched 2026-08-15 for the axis pins in
//!   [`W2vBert2Config::w2v_bert_2_0_default`]).
//! - Reference implementation: `transformers`
//!   `src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py`
//!   (<https://github.com/huggingface/transformers>) — the source of every
//!   tensor-name path and of the four divergences above.
//! - Papers: Chung et al. 2021 (<https://arxiv.org/abs/2108.06209>) and
//!   Barrault et al. 2023 (<https://arxiv.org/abs/2312.05187>).
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] /
//! [`DEFAULT_LICENSE_SPDX`] mirror the converter's constants — the same rule
//! the sibling binders (`emotion2vec` / `wavlm` / `canary_1b_flash` / …) use
//! so `vokra-models` does not gain a dependency edge onto `vokra-convert`,
//! preserving the layered convention `vokra-ops → nothing GGUF-aware`,
//! `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
//! `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! w2v-BERT 2.0 ships as a single ~2.16 GB `model.safetensors`; this runtime
//! **never** touches ONNX or pickle (FR-LD-05 / NFR-DS-02). At that size the
//! conversion itself is a vast.ai handoff per memory
//! `[[feedback-large-models-on-vast-ai]]`, not an M1-iMac job.
//!
//! `docs/license-audit.md` §3.1 sign-off stays **blank** — owner-only per
//! `[[feedback-license-signoff-primary-source]]`.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/w2v_bert_2.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model w2v-bert-2`.
///
/// Deliberately distinct from every sibling SSL-encoder arch (`hubert`,
/// `wav2vec2_ctc`, `wavlm_sv`, `emotion2vec`) **and** from the two composite
/// arches that embed w2v-BERT as an internal subgraph (`unity-2`,
/// `vieneu-tts`). Silently aliasing any of them would misroute the runtime
/// dispatch onto a different encoder topology or onto a composite tensor
/// namespace (FR-EX-08).
pub const ARCH: &str = "w2v-bert-2";

/// Expected `vokra.model.name` value written by the converter.
pub const NAME: &str = "w2v-bert-2.0";

/// Expected `vokra.model.category` value written by the converter.
///
/// `asr` is the converter's own classification — w2v-BERT 2.0 is a
/// *foundational encoder* for downstream ASR / AST rather than an ASR model
/// in its own right (it ships no task head). Mirrored verbatim so the
/// model-card generator and the zoo manifest tier gate agree with the
/// artifact on the wire.
pub const CATEGORY: &str = "asr";

/// Upstream HuggingFace slug, echoed in loud diagnostics so a reader never
/// has to re-fetch a manifest to find the source.
pub const UPSTREAM_HF: &str = "facebook/w2v-bert-2.0";

/// SPDX id the converter stamps when the caller passes no `--license`
/// override. `facebook/w2v-bert-2.0` is MIT
/// ([`LicenseClass::Permissive`], T1 Commercial tier).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// --- Primary-source anchors (cited in every loud-partial message) ----------

/// Primary-source anchor: the HF release.
pub const PRIMARY_SOURCE_HF: &str = "huggingface.co/facebook/w2v-bert-2.0";
/// Primary-source anchor: the `transformers` reference implementation whose
/// module paths are the tensor names this binder walks.
pub const PRIMARY_SOURCE_CODE: &str = "github.com/huggingface/transformers -> src/transformers/models/wav2vec2_bert/\
     modeling_wav2vec2_bert.py";
/// Primary-source anchor: the w2v-BERT paper (Chung et al. 2021).
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2108.06209";
/// Primary-source anchor: the Seamless-M4T v2 paper (Barrault et al. 2023),
/// which is the release vehicle for this checkpoint.
pub const PRIMARY_SOURCE_SEAMLESS_PAPER: &str = "arxiv.org/abs/2312.05187";

// --- Front-end + numeric constants (primary-source verified) ---------------

/// LayerNorm epsilon — upstream `config.layer_norm_eps = 1e-05`
/// (`facebook/w2v-bert-2.0/config.json`, fetched 2026-08-15). Used by the
/// real [`W2vBert2::project_features`] forward.
pub const LAYER_NORM_EPS: f32 = 1e-5;

/// Mel-band count the upstream front-end produces — `num_mel_bins: 80`
/// (`facebook/w2v-bert-2.0/preprocessor_config.json`, fetched 2026-08-15).
pub const NUM_MEL_BINS: u32 = 80;

/// Frame-stacking stride the upstream front-end applies — `stride: 2`
/// (same `preprocessor_config.json`). `NUM_MEL_BINS * FEATURE_STRIDE = 160`
/// is exactly `config.feature_projection_input_dim`, which is why this
/// binder's input contract is 160-wide stacked features rather than raw PCM.
pub const FEATURE_STRIDE: u32 = 2;

/// Input sample rate the upstream front-end expects, in Hz
/// (`sampling_rate: 16000`, same `preprocessor_config.json`).
pub const SAMPLE_RATE: u32 = 16_000;

/// The four block-level divergences between upstream
/// `Wav2Vec2BertEncoderLayer` and the shared
/// [`vokra_ops::conformer::ConformerEncoder`] primitive that keep
/// [`W2vBert2::encode`] a loud-partial.
///
/// Pinned as data (not just prose) so the unit tests can assert that the
/// loud-partial message actually enumerates every one of them — a future
/// wave that closes a gap must delete its entry here, which forces the
/// message and the tests to move together.
///
/// See the module docstring for the full derivation of each entry against
/// `modeling_wav2vec2_bert.py`.
pub const CONFORMER_PRIMITIVE_GAPS: [&str; 4] = [
    "relative_key attention bias (Shaw-style per-layer `distance_embedding` gather, \
     `nn.Embedding(left_max + right_max + 1, head_size)`) — \
     `vokra_ops::conformer::PositionEncoding` exposes only `None` and `Rope`",
    "causal left-only depthwise padding (upstream pads `(kernel_size - 1, 0)` then \
     convolves with `padding=0`) — `vokra_ops::conformer::conformer_conv` applies \
     symmetric same-padding `kernel_size / 2`",
    "bias-free convolution module (upstream `pointwise_conv1` / `depthwise_conv` / \
     `pointwise_conv2` are all `bias=False`) — \
     `vokra_ops::conformer::ConformerConvWeights` requires `pointwise1_b` / \
     `depthwise_b` / `pointwise2_b`",
    "pre-projection stem LayerNorm (upstream normalises the stacked input, then \
     projects) — `vokra_ops::conformer::ConvSubsampleKind::StackingNorm` projects, \
     then normalises",
];

// --- Tensor-name contract (verbatim upstream `state_dict` paths) -----------

/// `feature_projection.layer_norm.weight` — stem LayerNorm gain,
/// `[feature_projection_input_dim]`.
pub const T_FP_LN_WEIGHT: &str = "feature_projection.layer_norm.weight";
/// `feature_projection.layer_norm.bias` — stem LayerNorm bias,
/// `[feature_projection_input_dim]`.
pub const T_FP_LN_BIAS: &str = "feature_projection.layer_norm.bias";
/// `feature_projection.projection.weight` — stem `nn.Linear` weight,
/// `[hidden_size, feature_projection_input_dim]`.
pub const T_FP_PROJ_WEIGHT: &str = "feature_projection.projection.weight";
/// `feature_projection.projection.bias` — stem `nn.Linear` bias,
/// `[hidden_size]`.
pub const T_FP_PROJ_BIAS: &str = "feature_projection.projection.bias";

/// Prefix every per-layer tensor carries — `encoder.layers.{i}.`.
pub const LAYER_PREFIX: &str = "encoder.layers.";

/// Per-layer tensor suffixes that MUST be present for every index in
/// `0..num_hidden_layers`, in upstream `Wav2Vec2BertEncoderLayer` order.
///
/// `conv_module.pointwise_conv1` / `depthwise_conv` / `pointwise_conv2`
/// appear **without** a `.bias` companion on purpose: upstream constructs all
/// three `nn.Conv1d`s with `bias=False`, so requiring biases would reject a
/// legitimate checkpoint.
///
/// `self_attn.distance_embedding.weight` is deliberately **absent** from this
/// list — it exists only when `position_embeddings_type == "relative_key"`,
/// so it is treated as optional-and-informative (see
/// [`W2vBert2Config::head_size`]).
pub const REQUIRED_LAYER_SUFFIXES: [&str; 26] = [
    "ffn1_layer_norm.weight",
    "ffn1_layer_norm.bias",
    "ffn1.intermediate_dense.weight",
    "ffn1.intermediate_dense.bias",
    "ffn1.output_dense.weight",
    "ffn1.output_dense.bias",
    "self_attn_layer_norm.weight",
    "self_attn_layer_norm.bias",
    "self_attn.linear_q.weight",
    "self_attn.linear_q.bias",
    "self_attn.linear_k.weight",
    "self_attn.linear_k.bias",
    "self_attn.linear_v.weight",
    "self_attn.linear_v.bias",
    "self_attn.linear_out.weight",
    "self_attn.linear_out.bias",
    "conv_module.layer_norm.weight",
    "conv_module.layer_norm.bias",
    "conv_module.pointwise_conv1.weight",
    "conv_module.depthwise_conv.weight",
    "conv_module.depthwise_layer_norm.weight",
    "conv_module.depthwise_layer_norm.bias",
    "conv_module.pointwise_conv2.weight",
    "ffn2_layer_norm.weight",
    "ffn2_layer_norm.bias",
    "final_layer_norm.weight",
];

/// Per-layer suffix of the optional relative-position embedding table
/// (`[num_positions, head_size]`), present only under
/// `position_embeddings_type == "relative_key"` — which is what
/// `facebook/w2v-bert-2.0` ships.
pub const SUFFIX_DISTANCE_EMBEDDING: &str = "self_attn.distance_embedding.weight";

// ---------------------------------------------------------------------------
// Small GGUF helpers — every failure names the tensor.
// ---------------------------------------------------------------------------

/// Looks up `name` and returns its dims as `usize`, or a loud
/// [`VokraError::ModelLoad`] naming the tensor.
fn tensor_dims(gguf: &GgufFile, name: &str) -> Result<Vec<usize>> {
    let info = gguf.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "w2v-bert-2: GGUF is missing required tensor `{name}` — the converter \
             (`vokra-cli convert --model w2v-bert-2`) copies every float tensor through \
             under its verbatim upstream `state_dict` name, so an absent name always \
             signals a mis-produced or truncated GGUF rather than a naming-convention \
             difference. Re-convert from an upstream `{UPSTREAM_HF}` \
             `model.safetensors` (FR-EX-08 — no silent partial bind)."
        ))
    })?;
    Ok(info.dimensions.iter().map(|&d| d as usize).collect())
}

/// Like [`tensor_dims`], but also asserts the rank, naming both the expected
/// rank and the dims actually found.
fn tensor_dims_ranked(gguf: &GgufFile, name: &str, rank: usize) -> Result<Vec<usize>> {
    let dims = tensor_dims(gguf, name)?;
    if dims.len() != rank {
        return Err(VokraError::ModelLoad(format!(
            "w2v-bert-2: tensor `{name}` has rank {} (dims {dims:?}), expected rank \
             {rank}. The converter is a verbatim pass-through of the upstream \
             safetensors shapes, so a rank mismatch means the artifact was reshaped \
             or produced by a different exporter (FR-EX-08).",
            dims.len(),
        )));
    }
    Ok(dims)
}

/// Reads `name` as owned `f32` (dequantizing K-quants / F16 / BF16 through
/// the single `vokra-core` choke point), naming the tensor on failure.
fn tensor_f32(gguf: &GgufFile, name: &str) -> Result<Vec<f32>> {
    gguf.tensor_f32(name).map_err(|e| {
        VokraError::ModelLoad(format!(
            "w2v-bert-2: failed to decode tensor `{name}`: {e}. Re-convert from an \
             upstream `{UPSTREAM_HF}` `model.safetensors`."
        ))
    })
}

/// Parses `i` out of an `encoder.layers.{i}.…` tensor name.
fn layer_index(name: &str) -> Option<usize> {
    let rest = name.strip_prefix(LAYER_PREFIX)?;
    let dot = rest.find('.')?;
    rest[..dot].parse::<usize>().ok()
}

// ---------------------------------------------------------------------------
// W2vBert2Config — derived from tensor shapes, not from metadata.
// ---------------------------------------------------------------------------

/// w2v-BERT 2.0 topology, **recovered from the tensor shapes on disk**.
///
/// # Why shapes and not a `vokra.w2v_bert_2.*` chunk group
///
/// Unlike the `wavlm_sv` binder — whose converter transcribes `config.json`
/// into a stamped chunk group — the `w2v-bert-2` converter is a BF16
/// pass-through skeleton that stamps **only** arch / name / category /
/// provenance. There is no topology metadata to read. The two honest options
/// are (a) fabricate the axes from primary-source constants, or (b) recover
/// them from the tensor shapes, which are ground truth in the artifact.
/// This binder takes (b): a checkpoint that was fine-tuned to a different
/// width still binds correctly, and a truncated artifact fails loud instead
/// of being silently reinterpreted under 1024-wide constants.
///
/// [`w2v_bert_2_0_default`](Self::w2v_bert_2_0_default) carries the released
/// `facebook/w2v-bert-2.0` axes for reference and for test pinning — the
/// loader never falls back to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct W2vBert2Config {
    /// Transformer / Conformer model width (`config.hidden_size`), read from
    /// `feature_projection.projection.weight` dim 0.
    pub hidden_size: u32,
    /// Number of `Wav2Vec2BertEncoderLayer`s (`config.num_hidden_layers`),
    /// read from the contiguous `encoder.layers.{i}.` index range.
    pub num_hidden_layers: u32,
    /// Feed-forward inner width (`config.intermediate_size`), read from
    /// `encoder.layers.0.ffn1.intermediate_dense.weight` dim 0.
    pub intermediate_size: u32,
    /// Width of the stacked features the stem consumes
    /// (`config.feature_projection_input_dim`), read from
    /// `feature_projection.projection.weight` dim 1. For the released
    /// checkpoint this is `NUM_MEL_BINS * FEATURE_STRIDE = 160`.
    pub feature_projection_input_dim: u32,
    /// Depthwise kernel width of the conv module
    /// (`config.conv_depthwise_kernel_size`), read from
    /// `encoder.layers.0.conv_module.depthwise_conv.weight` dim 2.
    pub conv_depthwise_kernel_size: u32,
    /// Per-head attention width, read from
    /// `encoder.layers.0.self_attn.distance_embedding.weight` dim 1.
    ///
    /// `None` when the checkpoint carries no `distance_embedding` — i.e. when
    /// `position_embeddings_type` is not `"relative_key"`. Head geometry is
    /// **not** otherwise recoverable from shapes (Q/K/V/out are all
    /// `[hidden, hidden]` regardless of head count), so this binder reports
    /// `None` rather than guessing (FR-EX-08).
    pub head_size: Option<u32>,
    /// Attention head count, derived as `hidden_size / head_size` when
    /// [`head_size`](Self::head_size) is known, `None` otherwise.
    pub num_attention_heads: Option<u32>,
    /// Relative-position table height (`left_max_position_embeddings +
    /// right_max_position_embeddings + 1`), read from
    /// `distance_embedding.weight` dim 0. `None` when absent.
    ///
    /// Only the *sum* is recoverable from the tensor; the left/right split is
    /// a `config.json` fact that this artifact does not carry, so it is
    /// deliberately not reconstructed.
    pub num_relative_positions: Option<u32>,
}

impl W2vBert2Config {
    /// The released `facebook/w2v-bert-2.0` axes, transcribed from
    /// `config.json` (fetched 2026-08-15 — `hidden_size: 1024`,
    /// `num_hidden_layers: 24`, `num_attention_heads: 16`,
    /// `intermediate_size: 4096`, `feature_projection_input_dim: 160`,
    /// `conv_depthwise_kernel_size: 31`, `position_embeddings_type:
    /// "relative_key"`, `left_max_position_embeddings: 64`,
    /// `right_max_position_embeddings: 8`).
    ///
    /// Reference + test-pin only: [`from_gguf`](Self::from_gguf) never falls
    /// back to these values.
    #[must_use]
    pub fn w2v_bert_2_0_default() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            intermediate_size: 4096,
            feature_projection_input_dim: 160,
            conv_depthwise_kernel_size: 31,
            head_size: Some(64),
            num_attention_heads: Some(16),
            // left_max (64) + right_max (8) + 1
            num_relative_positions: Some(73),
        }
    }

    /// Recovers the topology from the tensor shapes carried by `gguf`.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when a shape-bearing tensor is absent
    ///   (naming the tensor), when its rank is unexpected (naming expected
    ///   rank and actual dims), when the `encoder.layers.{i}.` index range is
    ///   empty or non-contiguous, or when a recovered `head_size` does not
    ///   divide `hidden_size`.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // --- stem shapes: [hidden_size, feature_projection_input_dim] ------
        let proj = tensor_dims_ranked(gguf, T_FP_PROJ_WEIGHT, 2)?;
        let hidden_size = proj[0];
        let feature_projection_input_dim = proj[1];
        if hidden_size == 0 || feature_projection_input_dim == 0 {
            return Err(VokraError::ModelLoad(format!(
                "w2v-bert-2: tensor `{T_FP_PROJ_WEIGHT}` has a zero-length axis \
                 (dims {proj:?}) — refusing to bind a degenerate stem (FR-EX-08)."
            )));
        }

        // --- layer count: contiguous `encoder.layers.{i}.` index range -----
        let mut max_idx: Option<usize> = None;
        for info in gguf.tensors() {
            if let Some(i) = layer_index(&info.name) {
                max_idx = Some(max_idx.map_or(i, |m: usize| m.max(i)));
            }
        }
        let Some(max_idx) = max_idx else {
            return Err(VokraError::ModelLoad(format!(
                "w2v-bert-2: GGUF carries no `{LAYER_PREFIX}{{i}}.…` tensors — a \
                 legitimate w2v-BERT 2.0 checkpoint carries 24 Conformer encoder \
                 layers (~580M parameters). Zero encoder layers always signals a \
                 mis-produced GGUF, or a composite artifact whose encoder lives under \
                 a different namespace (`unity-2` nests it beneath `speech_encoder.`). \
                 Re-convert from an upstream `{UPSTREAM_HF}` `model.safetensors` \
                 (FR-EX-08)."
            )));
        };
        let num_hidden_layers = max_idx + 1;
        // Contiguity: probe one always-present tensor per index. A gap means
        // a partially-written artifact, which must not be silently bound with
        // a shorter stack.
        for i in 0..num_hidden_layers {
            let probe = format!("{LAYER_PREFIX}{i}.final_layer_norm.weight");
            if gguf.tensor_info(&probe).is_none() {
                return Err(VokraError::ModelLoad(format!(
                    "w2v-bert-2: encoder layer index range is non-contiguous — the \
                     highest `{LAYER_PREFIX}` index present is {max_idx} (implying \
                     {num_hidden_layers} layers), but `{probe}` is absent. Binding a \
                     stack with a hole would silently drop a layer (FR-EX-08). \
                     Re-convert from an upstream `{UPSTREAM_HF}` `model.safetensors`."
                )));
            }
        }

        // --- FFN inner width: [intermediate_size, hidden_size] -------------
        let ffn1 = tensor_dims_ranked(
            gguf,
            &format!("{LAYER_PREFIX}0.ffn1.intermediate_dense.weight"),
            2,
        )?;
        let intermediate_size = ffn1[0];
        if ffn1[1] != hidden_size {
            return Err(VokraError::ModelLoad(format!(
                "w2v-bert-2: `{LAYER_PREFIX}0.ffn1.intermediate_dense.weight` dim 1 is \
                 {} but the stem projects to hidden_size={hidden_size} — the artifact \
                 mixes two widths and cannot be bound (FR-EX-08).",
                ffn1[1],
            )));
        }

        // --- depthwise kernel: [hidden_size, 1, kernel] --------------------
        let dw = tensor_dims_ranked(
            gguf,
            &format!("{LAYER_PREFIX}0.conv_module.depthwise_conv.weight"),
            3,
        )?;
        if dw[0] != hidden_size || dw[1] != 1 {
            return Err(VokraError::ModelLoad(format!(
                "w2v-bert-2: `{LAYER_PREFIX}0.conv_module.depthwise_conv.weight` has \
                 dims {dw:?}, expected `[hidden_size={hidden_size}, 1, kernel]` — \
                 upstream builds it as a fully depthwise `nn.Conv1d(groups=hidden_size)`, \
                 so dim 1 is always 1 (FR-EX-08)."
            )));
        }
        let conv_depthwise_kernel_size = dw[2];

        // --- head geometry from the optional relative-position table -------
        let de_name = format!("{LAYER_PREFIX}0.{SUFFIX_DISTANCE_EMBEDDING}");
        let (head_size, num_attention_heads, num_relative_positions) =
            if gguf.tensor_info(&de_name).is_some() {
                let de = tensor_dims_ranked(gguf, &de_name, 2)?;
                let (num_positions, head) = (de[0], de[1]);
                if head == 0 || hidden_size % head != 0 {
                    return Err(VokraError::ModelLoad(format!(
                        "w2v-bert-2: `{de_name}` implies head_size={head}, which does \
                         not divide hidden_size={hidden_size}. Upstream computes \
                         `head_size = hidden_size // num_attention_heads`, so an \
                         indivisible pair means the artifact is inconsistent — \
                         refusing to guess a head count (FR-EX-08)."
                    )));
                }
                (
                    Some(head as u32),
                    Some((hidden_size / head) as u32),
                    Some(num_positions as u32),
                )
            } else {
                (None, None, None)
            };

        Ok(Self {
            hidden_size: hidden_size as u32,
            num_hidden_layers: num_hidden_layers as u32,
            intermediate_size: intermediate_size as u32,
            feature_projection_input_dim: feature_projection_input_dim as u32,
            conv_depthwise_kernel_size: conv_depthwise_kernel_size as u32,
            head_size,
            num_attention_heads,
            num_relative_positions,
        })
    }

    /// `true` when the checkpoint carries the per-layer relative-position
    /// table, i.e. when `position_embeddings_type == "relative_key"` (what
    /// `facebook/w2v-bert-2.0` ships).
    #[inline]
    #[must_use]
    pub const fn is_relative_key(&self) -> bool {
        self.head_size.is_some()
    }
}

// ---------------------------------------------------------------------------
// W2vBert2Weights — real manifest + real (dequantized) stem weights.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a w2v-BERT 2.0 GGUF.
///
/// The **stem** (`feature_projection.*`) is dequantized eagerly because
/// [`W2vBert2::project_features`] runs it for real — for the released
/// checkpoint that is `160 + 160 + 1024x160 + 1024` floats (~660 KB). The
/// 24-layer encoder stack is *catalogued* (name + dims, with per-layer
/// presence enforced) but not dequantized, because its forward is the
/// loud-partial: decoding ~580M parameters to serve an error would be pure
/// waste. The follow-up wave sizes that dequant per its kernel needs.
#[derive(Debug)]
pub struct W2vBert2Weights {
    tensors: Vec<(String, Vec<usize>)>,
    fp_ln_weight: Vec<f32>,
    fp_ln_bias: Vec<f32>,
    fp_proj_weight: Vec<f32>,
    fp_proj_bias: Vec<f32>,
}

impl W2vBert2Weights {
    /// Catalogues every tensor, enforces the per-layer required set, and
    /// dequantizes the four stem tensors.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors, when
    ///   any required stem or per-layer tensor is absent (**naming that
    ///   tensor**), when a stem tensor's length disagrees with the derived
    ///   config, or when a tensor fails to decode.
    pub fn from_gguf(gguf: &GgufFile, cfg: &W2vBert2Config) -> Result<Self> {
        let tensors: Vec<(String, Vec<usize>)> = gguf
            .tensors()
            .iter()
            .map(|info| {
                (
                    info.name.clone(),
                    info.dimensions.iter().map(|&d| d as usize).collect(),
                )
            })
            .collect();

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "w2v-bert-2: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). A legitimate w2v-BERT 2.0 checkpoint carries a \
                 ~580M-parameter Conformer encoder (arch={ARCH}, name={NAME}); zero \
                 tensors always signals a mis-produced GGUF. Re-run \
                 `vokra-cli convert --model w2v-bert-2` against an upstream \
                 `{UPSTREAM_HF}` `model.safetensors`."
            )));
        }

        // Per-layer required-tensor sweep. Extra tensors are tolerated (a
        // fine-tune may carry a task head this binder does not model); a
        // MISSING one is loud and names itself.
        for i in 0..cfg.num_hidden_layers {
            for suffix in REQUIRED_LAYER_SUFFIXES {
                let name = format!("{LAYER_PREFIX}{i}.{suffix}");
                if gguf.tensor_info(&name).is_none() {
                    return Err(VokraError::ModelLoad(format!(
                        "w2v-bert-2: GGUF is missing required per-layer tensor \
                         `{name}` (encoder layer {i} of {}). Upstream \
                         `Wav2Vec2BertEncoderLayer` always carries it; note that the \
                         three `conv_module` convolutions are `bias=False` upstream, so \
                         their biases are correctly absent — this one is not. \
                         Re-convert from an upstream `{UPSTREAM_HF}` \
                         `model.safetensors` (FR-EX-08 — no silent partial bind).",
                        cfg.num_hidden_layers,
                    )));
                }
            }
        }

        // Stem — decoded for real (this is what `project_features` runs).
        let fp_in = cfg.feature_projection_input_dim as usize;
        let hidden = cfg.hidden_size as usize;
        let fp_ln_weight = tensor_f32(gguf, T_FP_LN_WEIGHT)?;
        let fp_ln_bias = tensor_f32(gguf, T_FP_LN_BIAS)?;
        let fp_proj_weight = tensor_f32(gguf, T_FP_PROJ_WEIGHT)?;
        let fp_proj_bias = tensor_f32(gguf, T_FP_PROJ_BIAS)?;

        for (name, got, want) in [
            (T_FP_LN_WEIGHT, fp_ln_weight.len(), fp_in),
            (T_FP_LN_BIAS, fp_ln_bias.len(), fp_in),
            (T_FP_PROJ_WEIGHT, fp_proj_weight.len(), hidden * fp_in),
            (T_FP_PROJ_BIAS, fp_proj_bias.len(), hidden),
        ] {
            if got != want {
                return Err(VokraError::ModelLoad(format!(
                    "w2v-bert-2: tensor `{name}` decoded to {got} elements, expected \
                     {want} for hidden_size={hidden} / \
                     feature_projection_input_dim={fp_in} (FR-EX-08)."
                )));
            }
        }

        Ok(Self {
            tensors,
            fp_ln_weight,
            fp_ln_bias,
            fp_proj_weight,
            fp_proj_bias,
        })
    }

    /// Number of tensors catalogued from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// W2vBert2 — the runtime binder handle.
// ---------------------------------------------------------------------------

/// w2v-BERT 2.0 (`facebook/w2v-bert-2.0`, MIT) speech-encoder binder.
///
/// Bind with [`from_gguf`](Self::from_gguf). This is a **feature extractor**,
/// not an end-task model: [`encode`](Self::encode) is defined to return the
/// `[T, hidden_size]` hidden-state sequence a downstream head consumes, and
/// no classification head is invented on top (upstream ships none — the
/// release is `Wav2Vec2BertModel`, not `...ForCTC`).
///
/// [`project_features`](Self::project_features) — the `feature_projection`
/// stem — is implemented for real today. [`encode`](Self::encode) is a
/// loud-partial pending four additive `vokra-ops` primitives; see the module
/// docstring and [`CONFORMER_PRIMITIVE_GAPS`].
#[derive(Debug)]
pub struct W2vBert2 {
    config: W2vBert2Config,
    weights: W2vBert2Weights,
    weight_license: LicenseClass,
}

impl W2vBert2 {
    /// Binds a w2v-BERT 2.0 GGUF: validates arch, recovers the topology from
    /// tensor shapes, enforces the required tensor set, decodes the stem, and
    /// surfaces the stamped weight-license class.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the
    /// missing or wrong key so a reader diagnosing a mis-produced GGUF has
    /// exactly one place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or is
    ///   not `"w2v-bert-2"`.
    /// - [`VokraError::ModelLoad`] from [`W2vBert2Config::from_gguf`] (shape
    ///   recovery) or [`W2vBert2Weights::from_gguf`] (tensor sweep + stem
    ///   decode).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check first, so a mis-typed model fails with a specific
        //    message instead of a downstream missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "w2v-bert-2: GGUF arch is `{other}`, expected `{ARCH}` (was this \
                     GGUF produced by `vokra-cli convert --model w2v-bert-2`?). Two \
                     distinct neighbourhoods are easy to confuse here. (a) Sibling SSL \
                     speech encoders — `hubert` (Meta HuBERT, masked prediction over \
                     k-means cluster targets, vanilla Transformer body), `wav2vec2_ctc` \
                     (Meta wav2vec 2.0 + CTC ASR head, vanilla Transformer body), \
                     `wavlm_sv` (Microsoft WavLM + XVector speaker head, gated relative \
                     position bias), `emotion2vec` (9-class emotion head) — all share \
                     the general feature-extractor + body + head shape, but w2v-BERT \
                     2.0 alone has a CONFORMER body (conv module per layer), so a \
                     shared loader would walk the wrong tensor names. (b) Composites \
                     that EMBED w2v-BERT as an internal subgraph — `unity-2` \
                     (SeamlessM4T-v2 speech encoder) and `vieneu-tts` (VieNeu TTS \
                     speaker encoder) — nest these tensors under a composite prefix, so \
                     their artifacts must be bound by their own loaders. Silently \
                     aliasing any of them would misroute the runtime dispatch \
                     (FR-EX-08 — no silent partial load)."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "w2v-bert-2: GGUF is missing `vokra.model.arch` — this is not a \
                     Vokra-native w2v-bert-2 GGUF (was it produced by `vokra-cli \
                     convert --model w2v-bert-2` against `{UPSTREAM_HF}`?)"
                )));
            }
        }

        // 2. Topology recovered from real tensor shapes (the converter stamps
        //    no `vokra.w2v_bert_2.*` chunk group — see W2vBert2Config docs).
        let config = W2vBert2Config::from_gguf(file)?;

        // 3. Tensor sweep + stem decode.
        let weights = W2vBert2Weights::from_gguf(file, &config)?;

        // 4. Provenance surfacing. The converter stamps `Permissive` (MIT);
        //    a GGUF missing the stamp reads back as `Unknown`, which is
        //    fail-closed at the M2-13 compliance gate.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            config,
            weights,
            weight_license,
        })
    }

    /// The topology recovered from the artifact's tensor shapes.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &W2vBert2Config {
        &self.config
    }

    /// The stamped weight-license class. The converter stamps `Permissive`
    /// (MIT); a GGUF missing the stamp reads back as
    /// [`LicenseClass::Unknown`] (fail-closed).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors catalogued from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Runs the **`feature_projection` stem** for real:
    /// `LayerNorm(feature_projection_input_dim)` followed by
    /// `Linear(feature_projection_input_dim, hidden_size)`.
    ///
    /// This is upstream `Wav2Vec2BertFeatureProjection.forward` verbatim
    /// (dropout is inference-inert), with `eps` = [`LAYER_NORM_EPS`].
    ///
    /// `features` is a flat row-major `[n_frames, feature_projection_input_dim]`
    /// buffer — for the released checkpoint that is 80-band log-mel fbank at
    /// 16 kHz with stride-2 frame stacking (160 wide), as produced by the
    /// upstream `SeamlessM4TFeatureExtractor`. The fbank front-end itself is
    /// the caller's concern; this binder does not resample or featurise, so a
    /// width mismatch is a loud error rather than a silent reinterpretation.
    ///
    /// Returns a flat row-major `[n_frames, hidden_size]` buffer.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `n_frames == 0` or when
    ///   `features.len() != n_frames * feature_projection_input_dim`.
    pub fn project_features(&self, features: &[f32], n_frames: usize) -> Result<Vec<f32>> {
        let fp_in = self.config.feature_projection_input_dim as usize;
        let hidden = self.config.hidden_size as usize;
        if n_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "w2v-bert-2 project_features: n_frames must be > 0".to_owned(),
            ));
        }
        let expected = n_frames * fp_in;
        if features.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "w2v-bert-2 project_features: features length {} does not match \
                 n_frames x feature_projection_input_dim = {n_frames} x {fp_in} = \
                 {expected}. The stem consumes ALREADY-STACKED features \
                 ({NUM_MEL_BINS} mel bands x stride {FEATURE_STRIDE} at \
                 {SAMPLE_RATE} Hz upstream), not raw PCM and not unstacked mel.",
                features.len(),
            )));
        }

        let ln_w = &self.weights.fp_ln_weight;
        let ln_b = &self.weights.fp_ln_bias;
        let proj_w = &self.weights.fp_proj_weight;
        let proj_b = &self.weights.fp_proj_bias;

        let mut out = vec![0.0f32; n_frames * hidden];
        let mut normed = vec![0.0f32; fp_in];
        let n = fp_in as f32;
        for (src, dst) in features
            .chunks_exact(fp_in)
            .zip(out.chunks_exact_mut(hidden))
        {
            // LayerNorm over the feature axis.
            let mean = src.iter().sum::<f32>() / n;
            let var = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inv = 1.0 / (var + LAYER_NORM_EPS).sqrt();
            for ((slot, &x), (&g, &b)) in normed
                .iter_mut()
                .zip(src.iter())
                .zip(ln_w.iter().zip(ln_b.iter()))
            {
                *slot = (x - mean) * inv * g + b;
            }
            // Linear(fp_in -> hidden); `proj_w` is row-major [hidden, fp_in].
            for (o, slot) in dst.iter_mut().enumerate() {
                let row = &proj_w[o * fp_in..(o + 1) * fp_in];
                *slot = proj_b[o]
                    + row
                        .iter()
                        .zip(normed.iter())
                        .map(|(&w, &x)| w * x)
                        .sum::<f32>();
            }
        }
        Ok(out)
    }

    /// Encodes stacked features into the `[n_frames, hidden_size]` hidden-state
    /// sequence a downstream ASR / AST / speaker head consumes.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Input shapes are validated first — so a shape mistake surfaces as
    /// [`VokraError::InvalidArgument`] rather than being masked — and then the
    /// 24-layer Conformer stack returns [`VokraError::UnsupportedOp`] naming
    /// all four [`CONFORMER_PRIMITIVE_GAPS`] between upstream
    /// `Wav2Vec2BertEncoderLayer` and the shared
    /// [`vokra_ops::conformer::ConformerEncoder`] primitive, plus every
    /// recovered config axis and all four primary sources.
    ///
    /// The stem this method would run first is available today and works for
    /// real: see [`project_features`](Self::project_features). **No fabricated
    /// hidden states are ever emitted** (FR-EX-08 — no silent partial output).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on a bad `n_frames` / `features`
    ///   shape (same contract as [`project_features`](Self::project_features)).
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred Conformer encoder stack.
    pub fn encode(&self, features: &[f32], n_frames: usize) -> Result<(Vec<f32>, usize)> {
        let fp_in = self.config.feature_projection_input_dim as usize;
        if n_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "w2v-bert-2 encode: n_frames must be > 0".to_owned(),
            ));
        }
        let expected = n_frames * fp_in;
        if features.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "w2v-bert-2 encode: features length {} does not match n_frames x \
                 feature_projection_input_dim = {n_frames} x {fp_in} = {expected}.",
                features.len(),
            )));
        }
        Err(encode_forward_loud_partial(&self.config))
    }
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`W2vBert2::encode`] until `vokra-ops` grows the four primitives listed in
/// [`CONFORMER_PRIMITIVE_GAPS`].
///
/// Names all four gaps verbatim, echoes every recovered config axis so a
/// reader can cross-check the topology the follow-up wave targets, and cites
/// four primary sources (HF release + `transformers` reference + both
/// papers). Mirror of the `wavlm` / `emotion2vec` / `canary_1b_flash`
/// loud-partial-message precedent (CLAUDE.md 教訓 (a)).
fn encode_forward_loud_partial(cfg: &W2vBert2Config) -> VokraError {
    let heads = cfg
        .num_attention_heads
        .map_or_else(|| "unknown".to_owned(), |v| v.to_string());
    let head_size = cfg
        .head_size
        .map_or_else(|| "unknown".to_owned(), |v| v.to_string());
    let rel_positions = cfg
        .num_relative_positions
        .map_or_else(|| "absent".to_owned(), |v| v.to_string());
    VokraError::UnsupportedOp(format!(
        "w2v-bert-2 encode (loud-partial): the feature_projection stem is implemented \
         and runs today (call `W2vBert2::project_features`), but the \
         {nl}-layer Conformer encoder stack is not implemented — it needs four \
         primitives that `vokra_ops::conformer` does not expose yet: \
         (1) {g0}; (2) {g1}; (3) {g2}; (4) {g3}. \
         The shared `vokra_ops::conformer::ConformerEncoder` DOES cover the macaron \
         layer skeleton (half-scale FF1 -> MHA -> conv module -> half-scale FF2 -> \
         norm_out, Swish, channel-axis GLU), so closing this gap is additive work on \
         that primitive rather than a new per-model Conformer; composing the stack \
         with the primitive AS-IS would emit shape-valid but numerically wrong hidden \
         states, which FR-EX-08 forbids. Recovered topology (read from tensor shapes, \
         not from metadata — the converter stamps no `vokra.w2v_bert_2.*` group): \
         hidden_size={hs}, num_hidden_layers={nl}, intermediate_size={is_}, \
         feature_projection_input_dim={fp}, conv_depthwise_kernel_size={k}, \
         num_attention_heads={heads}, head_size={head_size}, \
         num_relative_positions={rel_positions}. Front-end contract: {mel} mel bands x \
         stride {stride} at {sr} Hz. Primary sources: {hf}, {code}, {paper}, \
         {seamless}. No fabricated hidden states are ever emitted (FR-EX-08 — no \
         silent partial output).",
        g0 = CONFORMER_PRIMITIVE_GAPS[0],
        g1 = CONFORMER_PRIMITIVE_GAPS[1],
        g2 = CONFORMER_PRIMITIVE_GAPS[2],
        g3 = CONFORMER_PRIMITIVE_GAPS[3],
        hs = cfg.hidden_size,
        nl = cfg.num_hidden_layers,
        is_ = cfg.intermediate_size,
        fp = cfg.feature_projection_input_dim,
        k = cfg.conv_depthwise_kernel_size,
        mel = NUM_MEL_BINS,
        stride = FEATURE_STRIDE,
        sr = SAMPLE_RATE,
        hf = PRIMARY_SOURCE_HF,
        code = PRIMARY_SOURCE_CODE,
        paper = PRIMARY_SOURCE_PAPER,
        seamless = PRIMARY_SOURCE_SEAMLESS_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the w2v-BERT 2.0 runtime binder.
    //!
    //! Unlike a pure-scaffold binder, one test here checks **real arithmetic**:
    //! `project_features` is a genuine forward (LayerNorm + Linear), so
    //! `project_features_matches_hand_computed_layer_norm_then_linear` pins it
    //! against a value computed by hand rather than against the
    //! implementation. Everything the checkpoint does NOT let us compute
    //! (the 24-layer Conformer stack) is asserted to loud-partial instead,
    //! per CLAUDE.md 教訓 (a).
    //!
    //! Coverage:
    //! 1. Contract-constant pin (cross-crate consistency with the converter)
    //!    and arch-tag distinctness from the SSL siblings + the two composites
    //!    that embed w2v-BERT.
    //! 2. `config.json` axis pin for `w2v_bert_2_0_default`.
    //! 3. Shape-derived config round-trip, including the head-count recovery
    //!    from `distance_embedding`, and the `None` path when it is absent.
    //! 4. Loud-error negative space: missing arch / foreign arch / zero
    //!    tensors / missing per-layer tensor / non-contiguous layer range.
    //! 5. Real stem forward + its shape-guard.
    //! 6. `encode` loud-partial naming all four primitive gaps.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Tiny synthetic topology — mirrors the real tensor-name contract at
    /// toy widths so a full GGUF fits in a unit test.
    #[derive(Clone, Copy)]
    struct Shape {
        fp_in: u64,
        hidden: u64,
        inter: u64,
        kernel: u64,
        layers: u64,
        num_positions: u64,
        head_size: u64,
    }

    impl Default for Shape {
        fn default() -> Self {
            // hidden 8 / head_size 4 => 2 heads; kernel 3 is odd as upstream
            // requires for SAME padding.
            Self {
                fp_in: 6,
                hidden: 8,
                inter: 16,
                kernel: 3,
                layers: 2,
                num_positions: 5,
                head_size: 4,
            }
        }
    }

    /// Knobs for building deliberately-broken fixtures.
    #[derive(Default)]
    struct Opts {
        /// `None` => omit `vokra.model.arch` entirely.
        arch: Option<&'static str>,
        license: Option<LicenseClass>,
        /// Skip `encoder.layers.{i}.self_attn.distance_embedding.weight`.
        no_distance_embedding: bool,
        /// Skip this exact tensor name.
        omit: Option<String>,
        /// Emit no tensors at all.
        no_tensors: bool,
        /// Explicit stem weights instead of zeros.
        stem: Option<StemWeights>,
    }

    /// Explicit feature-projection stem weights for a synthetic fixture:
    /// `(layer_norm_weight, layer_norm_bias, projection_weight,
    /// projection_bias)`. Named rather than left as a bare 4-tuple so the
    /// positions cannot be transposed silently at a call site.
    type StemWeights = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    impl Opts {
        fn valid() -> Self {
            Self {
                arch: Some(ARCH),
                license: Some(LicenseClass::Permissive),
                ..Self::default()
            }
        }
    }

    /// Appends an all-zero F32 tensor of the given shape.
    fn zeros_tensor(b: &mut GgufBuilder, name: &str, dims: Vec<u64>) {
        let n: u64 = dims.iter().product();
        b.add_tensor(name, GgmlType::F32, dims, vec![0u8; (n * 4) as usize])
            .expect("add_tensor");
    }

    fn push_zeros(b: &mut GgufBuilder, opts: &Opts, name: &str, dims: Vec<u64>) {
        if opts.omit.as_deref() == Some(name) {
            return;
        }
        zeros_tensor(b, name, dims);
    }

    fn push_floats(b: &mut GgufBuilder, name: &str, dims: Vec<u64>, v: &[f32]) {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        b.add_tensor(name, GgmlType::F32, dims, bytes)
            .expect("add_tensor");
    }

    /// Builds a synthetic w2v-BERT 2.0 GGUF honouring the real tensor-name
    /// contract (`feature_projection.*` + the full
    /// [`REQUIRED_LAYER_SUFFIXES`] set per layer).
    fn build_gguf(sh: &Shape, opts: &Opts) -> GgufFile {
        let mut b = GgufBuilder::new();
        if let Some(arch) = opts.arch {
            b.add_string(chunks::KEY_MODEL_ARCH, arch);
        }
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        if let Some(cls) = opts.license {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        if opts.no_tensors {
            return GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        }

        let (h, fp, inter, k) = (sh.hidden, sh.fp_in, sh.inter, sh.kernel);

        // --- stem ---------------------------------------------------------
        if let Some((ln_w, ln_b, proj_w, proj_b)) = &opts.stem {
            push_floats(&mut b, T_FP_LN_WEIGHT, vec![fp], ln_w);
            push_floats(&mut b, T_FP_LN_BIAS, vec![fp], ln_b);
            push_floats(&mut b, T_FP_PROJ_WEIGHT, vec![h, fp], proj_w);
            push_floats(&mut b, T_FP_PROJ_BIAS, vec![h], proj_b);
        } else {
            push_zeros(&mut b, opts, T_FP_LN_WEIGHT, vec![fp]);
            push_zeros(&mut b, opts, T_FP_LN_BIAS, vec![fp]);
            push_zeros(&mut b, opts, T_FP_PROJ_WEIGHT, vec![h, fp]);
            push_zeros(&mut b, opts, T_FP_PROJ_BIAS, vec![h]);
        }

        // --- encoder layers ----------------------------------------------
        for i in 0..sh.layers {
            let p = format!("{LAYER_PREFIX}{i}");
            for ln in [
                "ffn1_layer_norm",
                "self_attn_layer_norm",
                "conv_module.layer_norm",
                "conv_module.depthwise_layer_norm",
                "ffn2_layer_norm",
                "final_layer_norm",
            ] {
                push_zeros(&mut b, opts, &format!("{p}.{ln}.weight"), vec![h]);
                push_zeros(&mut b, opts, &format!("{p}.{ln}.bias"), vec![h]);
            }
            for ffn in ["ffn1", "ffn2"] {
                push_zeros(
                    &mut b,
                    opts,
                    &format!("{p}.{ffn}.intermediate_dense.weight"),
                    vec![inter, h],
                );
                push_zeros(
                    &mut b,
                    opts,
                    &format!("{p}.{ffn}.intermediate_dense.bias"),
                    vec![inter],
                );
                push_zeros(
                    &mut b,
                    opts,
                    &format!("{p}.{ffn}.output_dense.weight"),
                    vec![h, inter],
                );
                push_zeros(
                    &mut b,
                    opts,
                    &format!("{p}.{ffn}.output_dense.bias"),
                    vec![h],
                );
            }
            for lin in ["linear_q", "linear_k", "linear_v", "linear_out"] {
                push_zeros(
                    &mut b,
                    opts,
                    &format!("{p}.self_attn.{lin}.weight"),
                    vec![h, h],
                );
                push_zeros(&mut b, opts, &format!("{p}.self_attn.{lin}.bias"), vec![h]);
            }
            if !opts.no_distance_embedding {
                push_zeros(
                    &mut b,
                    opts,
                    &format!("{p}.{SUFFIX_DISTANCE_EMBEDDING}"),
                    vec![sh.num_positions, sh.head_size],
                );
            }
            // conv module convolutions are `bias=False` upstream.
            push_zeros(
                &mut b,
                opts,
                &format!("{p}.conv_module.pointwise_conv1.weight"),
                vec![2 * h, h, 1],
            );
            push_zeros(
                &mut b,
                opts,
                &format!("{p}.conv_module.depthwise_conv.weight"),
                vec![h, 1, k],
            );
            push_zeros(
                &mut b,
                opts,
                &format!("{p}.conv_module.pointwise_conv2.weight"),
                vec![h, h, 1],
            );
        }

        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. Contract-constant + arch-distinctness pins
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        assert_eq!(ARCH, "w2v-bert-2", "arch tag pin");
        assert_eq!(NAME, "w2v-bert-2.0", "model name pin");
        assert_eq!(CATEGORY, "asr", "category pin");
        assert_eq!(UPSTREAM_HF, "facebook/w2v-bert-2.0", "upstream slug pin");
        assert_eq!(DEFAULT_LICENSE_SPDX, "mit", "default SPDX pin");
        // Front-end constants (preprocessor_config.json, fetched 2026-08-15).
        assert_eq!(NUM_MEL_BINS, 80);
        assert_eq!(FEATURE_STRIDE, 2);
        assert_eq!(SAMPLE_RATE, 16_000);
        // The stacked width the stem consumes is exactly mel x stride, which
        // is why `feature_projection_input_dim` is 160 upstream.
        assert_eq!(
            NUM_MEL_BINS * FEATURE_STRIDE,
            W2vBert2Config::w2v_bert_2_0_default().feature_projection_input_dim,
            "80 mel bands x stride 2 must equal feature_projection_input_dim"
        );
    }

    #[test]
    fn arch_tag_distinct_from_ssl_siblings_and_embedding_composites() {
        // Sibling SSL encoders: same general shape, different bodies/heads.
        for sibling in ["hubert", "wav2vec2_ctc", "wavlm_sv", "emotion2vec"] {
            assert_ne!(
                ARCH, sibling,
                "w2v-bert-2 has a Conformer body while `{sibling}` does not — sharing \
                 an arch tag would mis-route runtime dispatch (FR-EX-08)"
            );
        }
        // Composites that EMBED w2v-BERT as an internal subgraph.
        for composite in ["unity-2", "vieneu-tts"] {
            assert_ne!(
                ARCH, composite,
                "`{composite}` nests w2v-BERT tensors under a composite prefix and must \
                 be bound by its own loader (FR-EX-08)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 2. `config.json` axis pin
    // -----------------------------------------------------------------------

    #[test]
    fn released_config_axes_match_upstream_config_json() {
        // facebook/w2v-bert-2.0/config.json, fetched 2026-08-15.
        let c = W2vBert2Config::w2v_bert_2_0_default();
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.num_hidden_layers, 24);
        assert_eq!(c.intermediate_size, 4096);
        assert_eq!(c.feature_projection_input_dim, 160);
        assert_eq!(c.conv_depthwise_kernel_size, 31);
        assert_eq!(c.num_attention_heads, Some(16));
        assert_eq!(c.head_size, Some(64));
        // left_max_position_embeddings 64 + right_max 8 + 1
        assert_eq!(c.num_relative_positions, Some(73));
        assert!(c.is_relative_key());
        // Upstream computes head_size = hidden_size // num_attention_heads.
        assert_eq!(c.hidden_size / c.num_attention_heads.unwrap(), 64);
        // Upstream rejects an even depthwise kernel (SAME padding requires odd).
        assert_eq!(c.conv_depthwise_kernel_size % 2, 1);
    }

    // -----------------------------------------------------------------------
    // 3. Shape-derived config round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_recovers_topology_from_tensor_shapes() {
        let sh = Shape::default();
        let file = build_gguf(&sh, &Opts::valid());
        let m = W2vBert2::from_gguf(&file).expect("valid GGUF must bind");
        let c = m.config();
        assert_eq!(c.hidden_size, sh.hidden as u32);
        assert_eq!(c.num_hidden_layers, sh.layers as u32);
        assert_eq!(c.intermediate_size, sh.inter as u32);
        assert_eq!(c.feature_projection_input_dim, sh.fp_in as u32);
        assert_eq!(c.conv_depthwise_kernel_size, sh.kernel as u32);
        // Head geometry recovered from `distance_embedding` [num_pos, head_size]
        // — the one axis that is NOT visible in any Q/K/V/out shape.
        assert_eq!(c.head_size, Some(sh.head_size as u32));
        assert_eq!(
            c.num_attention_heads,
            Some((sh.hidden / sh.head_size) as u32)
        );
        assert_eq!(c.num_relative_positions, Some(sh.num_positions as u32));
        assert!(c.is_relative_key());
        assert_eq!(m.weight_license(), LicenseClass::Permissive);
        assert!(m.tensor_count() >= 1);
    }

    #[test]
    fn head_geometry_is_none_without_distance_embedding() {
        // A checkpoint whose `position_embeddings_type` is not `relative_key`
        // carries no `distance_embedding`, and head count is then genuinely
        // unrecoverable from shapes. The binder reports `None` instead of
        // guessing (FR-EX-08).
        let sh = Shape::default();
        let opts = Opts {
            no_distance_embedding: true,
            ..Opts::valid()
        };
        let file = build_gguf(&sh, &opts);
        let m = W2vBert2::from_gguf(&file).expect("checkpoint without rel-pos must still bind");
        assert_eq!(m.config().head_size, None);
        assert_eq!(m.config().num_attention_heads, None);
        assert_eq!(m.config().num_relative_positions, None);
        assert!(!m.config().is_relative_key());
        // The loud-partial must say "unknown" rather than inventing a count.
        let feats = vec![0.0f32; sh.fp_in as usize];
        let Err(err) = m.encode(&feats, 1) else {
            panic!("encode must loud-partial");
        };
        let VokraError::UnsupportedOp(msg) = err else {
            panic!("expected UnsupportedOp");
        };
        assert!(
            msg.contains("num_attention_heads=unknown"),
            "must not fabricate a head count: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // 4. Loud-error negative space
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let sh = Shape::default();
        let opts = Opts {
            arch: None,
            ..Opts::valid()
        };
        let file = build_gguf(&sh, &opts);
        let Err(err) = W2vBert2::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        let VokraError::ModelLoad(m) = err else {
            panic!("expected VokraError::ModelLoad");
        };
        assert!(
            m.contains("missing `vokra.model.arch`"),
            "message must name the missing key: {m}"
        );
        assert!(
            m.contains("not a Vokra-native w2v-bert-2 GGUF"),
            "message must name the surface: {m}"
        );
    }

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_both_tags() {
        // A `hubert` GGUF handed here by mistake: same SSL lineage, vanilla
        // Transformer body instead of a Conformer body.
        let sh = Shape::default();
        let opts = Opts {
            arch: Some("hubert"),
            ..Opts::valid()
        };
        let file = build_gguf(&sh, &opts);
        let Err(err) = W2vBert2::from_gguf(&file) else {
            panic!("expected ModelLoad on foreign arch");
        };
        let VokraError::ModelLoad(m) = err else {
            panic!("expected VokraError::ModelLoad");
        };
        assert!(
            m.contains("`hubert`") && m.contains("`w2v-bert-2`"),
            "message must name BOTH the actual and the expected arch: {m}"
        );
        // The whole confusable neighbourhood must be enumerated.
        for sibling in [
            "hubert",
            "wav2vec2_ctc",
            "wavlm_sv",
            "emotion2vec",
            "unity-2",
            "vieneu-tts",
        ] {
            assert!(
                m.contains(sibling),
                "expected `{sibling}` disambiguation in error: {m}"
            );
        }
        assert!(
            m.contains("CONFORMER body"),
            "message should call out the body-topology divergence: {m}"
        );
        assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
    }

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        let sh = Shape::default();
        let opts = Opts {
            no_tensors: true,
            ..Opts::valid()
        };
        let file = build_gguf(&sh, &opts);
        let Err(err) = W2vBert2::from_gguf(&file) else {
            panic!("expected ModelLoad on empty tensor manifest");
        };
        let VokraError::ModelLoad(m) = err else {
            panic!("expected VokraError::ModelLoad");
        };
        // The stem-shape probe is the first thing that touches tensors, so it
        // is what fires — and it names the tensor, which is the contract.
        assert!(
            m.contains(T_FP_PROJ_WEIGHT),
            "message must name the tensor it went looking for: {m}"
        );
        assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
    }

    #[test]
    fn from_gguf_rejects_missing_per_layer_tensor_naming_it() {
        // Drop one required per-layer tensor from the LAST layer, so config
        // derivation (which probes layer 0) still succeeds and the per-layer
        // sweep is what fires.
        let sh = Shape::default();
        let missing = format!("{LAYER_PREFIX}1.self_attn.linear_q.weight");
        let opts = Opts {
            omit: Some(missing.clone()),
            ..Opts::valid()
        };
        let file = build_gguf(&sh, &opts);
        let Err(err) = W2vBert2::from_gguf(&file) else {
            panic!("expected ModelLoad on missing per-layer tensor");
        };
        let VokraError::ModelLoad(m) = err else {
            panic!("expected VokraError::ModelLoad");
        };
        assert!(
            m.contains(&missing),
            "message must name the missing tensor `{missing}`: {m}"
        );
        assert!(
            m.contains("bias=False"),
            "message should pre-empt the obvious wrong guess (conv biases are \
             legitimately absent upstream): {m}"
        );
        assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
    }

    #[test]
    fn from_gguf_rejects_non_contiguous_layer_range() {
        // Layers 0 and 2 present, layer 1 entirely absent: binding a stack
        // with a hole would silently drop a layer.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        let (h, fp, inter, k) = (8u64, 6u64, 16u64, 3u64);
        zeros_tensor(&mut b, T_FP_LN_WEIGHT, vec![fp]);
        zeros_tensor(&mut b, T_FP_LN_BIAS, vec![fp]);
        zeros_tensor(&mut b, T_FP_PROJ_WEIGHT, vec![h, fp]);
        zeros_tensor(&mut b, T_FP_PROJ_BIAS, vec![h]);
        for i in [0u64, 2] {
            zeros_tensor(
                &mut b,
                &format!("{LAYER_PREFIX}{i}.ffn1.intermediate_dense.weight"),
                vec![inter, h],
            );
            zeros_tensor(
                &mut b,
                &format!("{LAYER_PREFIX}{i}.conv_module.depthwise_conv.weight"),
                vec![h, 1, k],
            );
            zeros_tensor(
                &mut b,
                &format!("{LAYER_PREFIX}{i}.final_layer_norm.weight"),
                vec![h],
            );
        }
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = W2vBert2::from_gguf(&file) else {
            panic!("expected ModelLoad on a non-contiguous layer range");
        };
        let VokraError::ModelLoad(m) = err else {
            panic!("expected VokraError::ModelLoad");
        };
        assert!(
            m.contains("non-contiguous"),
            "message must name the contiguity gap: {m}"
        );
        assert!(
            m.contains("encoder.layers.1.final_layer_norm.weight"),
            "message must name the probe tensor that was absent: {m}"
        );
    }

    // -----------------------------------------------------------------------
    // 5. REAL stem forward
    // -----------------------------------------------------------------------

    #[test]
    fn project_features_matches_hand_computed_layer_norm_then_linear() {
        // fp_in = 4, hidden = 2, one frame.
        //   input      = [1, 2, 3, 4]
        //   mean       = 2.5
        //   var        = ((-1.5)^2 + (-0.5)^2 + 0.5^2 + 1.5^2) / 4 = 1.25
        //   inv        = 1 / sqrt(1.25 + 1e-5)
        //   normed     = [-1.5, -0.5, 0.5, 1.5] * inv      (gamma=1, beta=0)
        //   proj row 0 = [1,0,0,0], row 1 = [0,0,0,1]; bias = [0.5, -0.5]
        //   out        = [normed[0] + 0.5, normed[3] - 0.5]
        let sh = Shape {
            fp_in: 4,
            hidden: 2,
            inter: 4,
            kernel: 3,
            layers: 1,
            num_positions: 5,
            head_size: 1,
        };
        let stem = (
            vec![1.0f32; 4],                                   // LayerNorm gamma
            vec![0.0f32; 4],                                   // LayerNorm beta
            vec![1.0, 0.0, 0.0, 0.0, /**/ 0.0, 0.0, 0.0, 1.0], // [hidden=2, fp_in=4]
            vec![0.5f32, -0.5],                                // projection bias
        );
        let opts = Opts {
            stem: Some(stem),
            ..Opts::valid()
        };
        let file = build_gguf(&sh, &opts);
        let m = W2vBert2::from_gguf(&file).expect("valid GGUF must bind");

        let out = m
            .project_features(&[1.0, 2.0, 3.0, 4.0], 1)
            .expect("stem must run for real");
        assert_eq!(out.len(), 2, "output must be [n_frames=1, hidden=2]");

        let inv = 1.0f32 / (1.25f32 + LAYER_NORM_EPS).sqrt();
        let expect0 = -1.5f32 * inv + 0.5;
        let expect1 = 1.5f32 * inv - 0.5;
        assert!(
            (out[0] - expect0).abs() < 1e-5,
            "out[0] = {} expected {expect0}",
            out[0]
        );
        assert!(
            (out[1] - expect1).abs() < 1e-5,
            "out[1] = {} expected {expect1}",
            out[1]
        );

        // Two frames must produce two independent rows (per-frame LayerNorm,
        // not a whole-buffer normalisation).
        let two = m
            .project_features(&[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], 2)
            .expect("two frames must run");
        assert_eq!(two.len(), 4);
        assert!((two[0] - out[0]).abs() < 1e-6 && (two[2] - out[0]).abs() < 1e-6);
    }

    #[test]
    fn project_features_rejects_bad_shapes() {
        let sh = Shape::default();
        let file = build_gguf(&sh, &Opts::valid());
        let m = W2vBert2::from_gguf(&file).unwrap();

        let Err(err) = m.project_features(&[], 0) else {
            panic!("expected InvalidArgument on n_frames == 0");
        };
        let VokraError::InvalidArgument(msg) = err else {
            panic!("expected VokraError::InvalidArgument");
        };
        assert!(msg.contains("n_frames must be > 0"), "{msg}");

        // 5 values against fp_in = 6.
        let Err(err) = m.project_features(&[0.0; 5], 1) else {
            panic!("expected InvalidArgument on a width mismatch");
        };
        let VokraError::InvalidArgument(msg) = err else {
            panic!("expected VokraError::InvalidArgument");
        };
        assert!(
            msg.contains("ALREADY-STACKED"),
            "message must explain the stacked-feature contract: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // 6. `encode` loud-partial
    // -----------------------------------------------------------------------

    #[test]
    fn encode_loud_partials_naming_every_missing_primitive() {
        let sh = Shape::default();
        let file = build_gguf(&sh, &Opts::valid());
        let m = W2vBert2::from_gguf(&file).unwrap();

        // Legitimately-shaped input, so the loud-partial fires for the real
        // reason rather than for a shape mistake.
        let feats = vec![0.0f32; (sh.fp_in * 4) as usize];
        let Err(err) = m.encode(&feats, 4) else {
            panic!("encode must loud-partial");
        };
        let VokraError::UnsupportedOp(msg) = err else {
            panic!("expected VokraError::UnsupportedOp");
        };

        assert!(msg.contains("w2v-bert-2 encode"), "surface: {msg}");
        assert!(msg.contains("loud-partial"), "posture label: {msg}");

        // Every one of the four gaps must be enumerated verbatim.
        for gap in CONFORMER_PRIMITIVE_GAPS {
            assert!(msg.contains(gap), "missing gap `{gap}` in message: {msg}");
        }
        // The primitive it needs must be named by path.
        assert!(
            msg.contains("vokra_ops::conformer"),
            "message must name the primitive that has to grow: {msg}"
        );
        // The working alternative must be pointed at.
        assert!(
            msg.contains("project_features"),
            "message must point at the stem that DOES work: {msg}"
        );
        // Recovered axes echoed so a reader can cross-check the topology.
        assert!(msg.contains("hidden_size=8"), "hidden_size axis: {msg}");
        assert!(
            msg.contains("num_hidden_layers=2"),
            "num_hidden_layers axis: {msg}"
        );
        assert!(
            msg.contains("intermediate_size=16"),
            "intermediate_size axis: {msg}"
        );
        assert!(
            msg.contains("feature_projection_input_dim=6"),
            "feature_projection_input_dim axis: {msg}"
        );
        assert!(
            msg.contains("conv_depthwise_kernel_size=3"),
            "conv_depthwise_kernel_size axis: {msg}"
        );
        assert!(
            msg.contains("num_attention_heads=2"),
            "recovered head count: {msg}"
        );
        // All four primary sources cited.
        for url in [
            PRIMARY_SOURCE_HF,
            PRIMARY_SOURCE_CODE,
            PRIMARY_SOURCE_PAPER,
            PRIMARY_SOURCE_SEAMLESS_PAPER,
        ] {
            assert!(msg.contains(url), "expected primary source `{url}`: {msg}");
        }
        assert!(msg.contains("FR-EX-08"), "FR-EX-08 rationale: {msg}");
    }

    #[test]
    fn encode_validates_shapes_before_loud_partialling() {
        // A shape mistake must surface AS a shape mistake — the loud-partial
        // must not mask it.
        let sh = Shape::default();
        let file = build_gguf(&sh, &Opts::valid());
        let m = W2vBert2::from_gguf(&file).unwrap();
        let Err(err) = m.encode(&[0.0; 5], 1) else {
            panic!("expected InvalidArgument on a width mismatch");
        };
        let VokraError::InvalidArgument(msg) = err else {
            panic!("expected VokraError::InvalidArgument, not the loud-partial");
        };
        assert!(msg.contains("w2v-bert-2 encode"), "{msg}");
    }

    // -----------------------------------------------------------------------
    // 7. Required-suffix table sanity
    // -----------------------------------------------------------------------

    #[test]
    fn required_layer_suffixes_exclude_bias_free_conv_weights() {
        // Upstream builds all three conv-module convolutions with bias=False,
        // so requiring their biases would reject a legitimate checkpoint.
        for banned in [
            "conv_module.pointwise_conv1.bias",
            "conv_module.depthwise_conv.bias",
            "conv_module.pointwise_conv2.bias",
        ] {
            assert!(
                !REQUIRED_LAYER_SUFFIXES.contains(&banned),
                "`{banned}` must NOT be required — upstream constructs that Conv1d \
                 with bias=False"
            );
        }
        // The relative-position table is optional (relative_key only).
        assert!(
            !REQUIRED_LAYER_SUFFIXES.contains(&SUFFIX_DISTANCE_EMBEDDING),
            "distance_embedding is present only under position_embeddings_type = \
             relative_key and must stay optional"
        );
        // The load-bearing weights ARE required.
        for needed in [
            "self_attn.linear_q.weight",
            "conv_module.depthwise_conv.weight",
            "final_layer_norm.weight",
        ] {
            assert!(REQUIRED_LAYER_SUFFIXES.contains(&needed), "{needed}");
        }
    }
}
