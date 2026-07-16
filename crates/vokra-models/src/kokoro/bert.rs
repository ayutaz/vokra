//! Kokoro-82M PL-BERT branch (M2-07-T13-beta, 2026-07-07).
//!
//! # Overview
//!
//! Kokoro-82M's checkpoint carries a **PL-BERT / ALBERT shared-weight
//! transformer** (`bert.module.*`, 24 weights; ONE shared block applied
//! `num_hidden_layers = 12` times per the upstream `config.json` `plbert`
//! section) plus a `bert_encoder.module.*` Linear that projects the 768-d
//! output to 512-d. In the StyleTTS 2 派生
//! runtime graph, the PL-BERT branch runs BEFORE the phoneme text encoder and
//! produces auxiliary per-token features the prosody predictor consumes
//! alongside the [`super::text_encoder::TextEncoder`] output. The M2-07 T13-alpha
//! scaffolds did NOT include this branch — this file is the T13-beta landing
//! that closes the missing top-level component identified by
//! `docs/adr/0007-kokoro-native.md` §"T02 upstream inspection findings".
//!
//! # Architecture — bound to the upstream manifest
//!
//! ```text
//! Embeddings:
//!   word_embeddings.weight [178, 128]           # phoneme vocab
//!   position_embeddings.weight [512, 128]       # max_pos = 512
//!   token_type_embeddings.weight [2, 128]       # single segment (id = 0)
//!   LayerNorm.{weight,bias} [128]
//!
//! Encoder body:
//!   embedding_hidden_mapping_in.{weight,bias}   # 128 → 768
//!   ALBERT-shared block (× 12, config.json plbert.num_hidden_layers):
//!     attention.{query,key,value,dense}.{weight,bias}  # 768 → 768 each
//!     attention.LayerNorm.{weight,bias}                # [768]
//!     ffn.{weight,bias}                                # 768 → 2048
//!     ffn_output.{weight,bias}                         # 2048 → 768
//!     full_layer_layer_norm.{weight,bias}              # [768]
//!
//! Pooler:
//!   pooler.{weight,bias}                        # 768 → 768
//!
//! Downstream projection (top-level `bert_encoder.module`):
//!   weight [512, 768] + bias [512]              # 768 → 512
//! ```
//!
//! # Design notes
//!
//! * **Weight names** are verbatim from
//!   `crates/vokra-models/src/kokoro/data/upstream_tensors_v1_0.tsv`. Every
//!   [`super::weights::TensorStore::tensor_shaped`] call names a real manifest
//!   entry; a missing name or a shape mismatch is a loud
//!   [`VokraError::InvalidArgument`] (FR-EX-08 — no silent architecture drift).
//! * **Constants** (`N_VOCAB = 178`, `EMBED_SIZE = 128`, `HIDDEN = 768`,
//!   `FFN_HIDDEN = 2048`, `MAX_POS = 512`, `N_TOKEN_TYPES = 2`,
//!   `OUT_DIM = 512`) are pinned by the manifest shapes; each is
//!   cross-checked against the loaded tensor's shape at load time via
//!   `tensor_shaped`. `N_LAYERS = 12` and `N_HEADS = 12` are pinned by the
//!   upstream `config.json` `plbert` section (`num_hidden_layers` /
//!   `num_attention_heads`) — the ALBERT weight-sharing pattern makes both
//!   invisible to the tensor manifest.
//! * **Weight pre-transpose**: every `nn.Linear` weight is stored as
//!   `[out, in]` row-major in the checkpoint; the [`Compute::gemm_f32`] seam
//!   expects the right operand as `[k, n]` = `[in, out]`. Each weight is
//!   therefore transposed **once** at load time and stored as `w_t` in a
//!   private [`BertLinear`], matching the whisper `Linear` convention.
//! * **Pooler application**: the pooler tensors are loaded strictly (manifest
//!   completeness) but **not applied**. Upstream `CustomAlbert.forward`
//!   (`kokoro==0.9.4` `modules.py:180-183`) returns
//!   `outputs.last_hidden_state`, discarding the pooler `AlbertModel`
//!   computes; `KModel.forward_with_tokens` (`model.py:102-103`) projects
//!   that last-hidden-state through `bert_encoder`. The pre-fix per-token
//!   `Linear + tanh` pooler insertion was part of the P1 `bert` divergence
//!   found by the 2026-07-16 real-weight eval.
//! * **LayerNorm eps** = `1e-12` (BERT / ALBERT convention, differs from
//!   Whisper's `1e-5`). Passed explicitly to
//!   [`Compute::layer_norm_f32`] on every call.
//! * **No RNG**: the forward path is deterministic. Two identical inputs
//!   produce identical outputs (asserted by the determinism test).
//!
//! # What this file is NOT
//!
//! * NOT a new first-class `vokra-ops` op — the ADR D2/D6/D7 red lines forbid
//!   composing a new op just for BERT. Every kernel call is either the shared
//!   [`Compute`] seam or [`super::nn`]'s existing helpers.
//! * NOT self-verifying numerically — this file's tests cover shape /
//!   determinism / load-time errors only. The byte-level parity vs the TRUE
//!   upstream `kokoro` package (`tools/parity/dump_kokoro_reference.py` →
//!   `crates/vokra-models/tests/parity_kokoro.rs`) is the numerical gate.

use vokra_core::{Result, VokraError};

use super::config::KokoroConfig;
use super::nn::gelu_new;
use super::weights::TensorStore;
use crate::compute::Compute;

// ---- Architectural constants pinned by the upstream manifest --------------
//
// Every constant here corresponds to a shape axis in
// `crates/vokra-models/src/kokoro/data/upstream_tensors_v1_0.tsv`; the load
// path cross-checks each one via `tensor_shaped`, so a mismatch fails loudly.

/// Phoneme vocabulary size — `bert.module.embeddings.word_embeddings.weight`
/// axis 0.
const N_VOCAB: usize = 178;

/// Word-embedding dimension — `bert.module.embeddings.word_embeddings.weight`
/// axis 1. Also the input dim of `embedding_hidden_mapping_in`.
const EMBED_SIZE: usize = 128;

/// Transformer hidden dim — output of `embedding_hidden_mapping_in`; Q/K/V/O
/// I/O; FFN input; pooler I/O.
const HIDDEN: usize = 768;

/// FFN intermediate dim — `bert.module.encoder.albert_layer_groups.0.
/// albert_layers.0.ffn.weight` axis 0.
const FFN_HIDDEN: usize = 2048;

/// Position-embedding table length — `bert.module.embeddings.
/// position_embeddings.weight` axis 0. Bounds the phoneme sequence length.
const MAX_POS: usize = 512;

/// Number of token-type ids — `bert.module.embeddings.token_type_embeddings.
/// weight` axis 0. Kokoro uses a single segment (id = 0).
const N_TOKEN_TYPES: usize = 2;

/// Number of ALBERT layers — the shared block is applied this many times.
///
/// Pinned by the upstream `hexgrad/Kokoro-82M` `config.json`
/// (`plbert.num_hidden_layers = 12`; the ALBERT weight-sharing pattern keeps
/// the checkpoint at ONE shared block regardless of the layer count, so this
/// value is invisible to the tensor manifest). The pre-fix value `4` was
/// inferred from shapes alone and was the dominant term of the P1 `bert`
/// divergence (Δ 5.84) found by the 2026-07-16 real-weight eval.
const N_LAYERS: usize = 12;

/// Downstream feature dim — `bert_encoder.module.weight` axis 0.
const OUT_DIM: usize = 512;

/// Attention head count — ALBERT-base convention (`HIDDEN / head_dim` with
/// `head_dim = 64`). Not pinned by a manifest axis (the manifest carries
/// `[HIDDEN, HIDDEN]` Q/K/V/O weights but not the head split); pinned here as
/// the standard ALBERT-base value. The T17 parity landing validates this
/// against the upstream `config.json`.
const N_HEADS: usize = 12;

/// LayerNorm epsilon — BERT / ALBERT convention (differs from Whisper's
/// `1e-5`). Sourced from the HuggingFace ALBERT config default
/// (`layer_norm_eps: 1e-12`).
const LAYER_NORM_EPS: f32 = 1e-12;

/// Bias-carrying row-major Linear layer with the weight pre-transposed to the
/// `[in, out]` layout [`Compute::gemm_f32`] expects as its right operand.
///
/// PyTorch stores `nn.Linear.weight` as `[out_features, in_features]`
/// row-major; the `gemm_f32` seam wants the right operand as `[k, n]`
/// = `[in, out]`. Transposing once at load time keeps every forward call to a
/// single kernel dispatch. Matches the whisper `Linear`/`w_t` convention
/// (see `crates/vokra-models/src/whisper/weights.rs`).
#[derive(Debug)]
struct BertLinear {
    /// `[in_features · out_features]` row-major transposed weight.
    w_t: Vec<f32>,
    /// `[out_features]` bias (BERT `nn.Linear` always has a bias term).
    bias: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl BertLinear {
    /// Owns the loaded weight (as stored in the checkpoint, `[out, in]`
    /// row-major) and transposes it to the `[in, out]` layout the GEMM seam
    /// expects.
    fn from_pytorch(w: Vec<f32>, bias: Vec<f32>, out_features: usize, in_features: usize) -> Self {
        debug_assert_eq!(w.len(), out_features * in_features, "BertLinear: w len");
        debug_assert_eq!(bias.len(), out_features, "BertLinear: bias len");
        let mut w_t = vec![0.0f32; in_features * out_features];
        for i in 0..out_features {
            for j in 0..in_features {
                // w[i, j] (out-major) → w_t[j, i] (in-major)
                w_t[j * out_features + i] = w[i * in_features + j];
            }
        }
        Self {
            w_t,
            bias,
            in_features,
            out_features,
        }
    }

    /// `out[t, o] = bias[o] + Σ_i x[t, i] · w_t[i, o]`.
    ///
    /// `x` is `[t, in_features]` row-major; `out` is sized `[t, out_features]`.
    fn linear_into(&self, compute: &Compute, x: &[f32], t: usize, out: &mut [f32]) -> Result<()> {
        compute.gemm_f32(
            t,
            self.out_features,
            self.in_features,
            x,
            &self.w_t,
            Some(&self.bias),
            out,
        )
    }
}

/// PyTorch `nn.LayerNorm` over the innermost axis, with the
/// `[hidden_dim]` γ / β pair the checkpoint carries.
#[derive(Debug)]
struct BertLayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    hidden: usize,
}

impl BertLayerNorm {
    fn new(gamma: Vec<f32>, beta: Vec<f32>, hidden: usize) -> Self {
        debug_assert_eq!(gamma.len(), hidden, "BertLayerNorm: gamma len");
        debug_assert_eq!(beta.len(), hidden, "BertLayerNorm: beta len");
        Self {
            gamma,
            beta,
            hidden,
        }
    }

    /// `out[t, c] = γ[c] · (x[t, c] − mean_t) / sqrt(var_t + eps) + β[c]`.
    fn forward_into(&self, compute: &Compute, x: &[f32], t: usize, out: &mut [f32]) -> Result<()> {
        compute.layer_norm_f32(
            x,
            out,
            t,
            self.hidden,
            &self.gamma,
            &self.beta,
            LAYER_NORM_EPS,
        )
    }
}

/// The one shared ALBERT block Kokoro's PL-BERT reuses `N_LAYERS = 12` times.
///
/// PyTorch names this `albert_layer_groups.0.albert_layers.0`; the surrounding
/// `groups`/`layers` iteration in the upstream reference is a for-loop over a
/// single element list (the ALBERT sharing pattern). The Rust representation
/// mirrors that: one struct, applied four times in the forward path.
#[derive(Debug)]
struct BertSharedLayer {
    query: BertLinear,
    key: BertLinear,
    value: BertLinear,
    attn_output: BertLinear,
    attn_ln: BertLayerNorm,
    ffn: BertLinear,
    ffn_output: BertLinear,
    full_ln: BertLayerNorm,
}

/// The Kokoro-82M PL-BERT branch — phoneme_ids → `[t · 512]` row-major
/// features consumed by the prosody predictor.
///
/// Every field is loaded via [`TensorStore::tensor_shaped`] under its exact
/// upstream name; the shared ALBERT block is stored ONCE and applied
/// [`N_LAYERS`] times in the forward path (the point of ALBERT weight sharing).
#[derive(Debug)]
#[allow(dead_code)] // fields consumed by [`Bert::forward`] + tests
pub(crate) struct Bert {
    // ---- Embeddings ----
    word_emb: Vec<f32>,       // [N_VOCAB, EMBED_SIZE]
    pos_emb: Vec<f32>,        // [MAX_POS, EMBED_SIZE]
    token_type_emb: Vec<f32>, // [N_TOKEN_TYPES, EMBED_SIZE]
    emb_ln: BertLayerNorm,    // [EMBED_SIZE]

    // ---- Encoder ----
    mapping_in: BertLinear, // 128 → 768
    shared_layer: BertSharedLayer,

    // ---- Pooler (loaded strictly for manifest completeness; NOT applied:
    // upstream `CustomAlbert` returns `last_hidden_state` and discards the
    // pooler — `modules.py:180-183`) ----
    #[allow(dead_code)]
    pooler: BertLinear, // 768 → 768

    // ---- Downstream projection (top-level bert_encoder.module) ----
    projection: BertLinear, // 768 → 512
}

impl Bert {
    /// Loads the PL-BERT branch from a Kokoro voice GGUF.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] naming the offending tensor if
    /// any weight in the `bert.module.*` or `bert_encoder.module.*` set is
    /// missing or has an unexpected shape (FR-EX-08). The `_config` argument is
    /// currently unused (the ALBERT-4 dims are pinned by the manifest, not the
    /// runtime `KokoroConfig`); it is threaded for API symmetry with the sibling
    /// component loaders and for the T17 parity landing which will cross-check
    /// against the `vokra.kokoro.*` metadata.
    pub(crate) fn new(store: &TensorStore, _config: &KokoroConfig) -> Result<Self> {
        // --- Embeddings ---
        let word_emb = store.tensor_shaped(
            "bert.module.embeddings.word_embeddings.weight",
            &[N_VOCAB, EMBED_SIZE],
        )?;
        let pos_emb = store.tensor_shaped(
            "bert.module.embeddings.position_embeddings.weight",
            &[MAX_POS, EMBED_SIZE],
        )?;
        let token_type_emb = store.tensor_shaped(
            "bert.module.embeddings.token_type_embeddings.weight",
            &[N_TOKEN_TYPES, EMBED_SIZE],
        )?;
        let emb_ln_g =
            store.tensor_shaped("bert.module.embeddings.LayerNorm.weight", &[EMBED_SIZE])?;
        let emb_ln_b =
            store.tensor_shaped("bert.module.embeddings.LayerNorm.bias", &[EMBED_SIZE])?;

        // --- Encoder: embedding_hidden_mapping_in (128 → 768) ---
        let mapping_w = store.tensor_shaped(
            "bert.module.encoder.embedding_hidden_mapping_in.weight",
            &[HIDDEN, EMBED_SIZE],
        )?;
        let mapping_b = store.tensor_shaped(
            "bert.module.encoder.embedding_hidden_mapping_in.bias",
            &[HIDDEN],
        )?;

        // --- Encoder: shared ALBERT block (loaded ONCE, applied N_LAYERS times) ---
        let prefix = "bert.module.encoder.albert_layer_groups.0.albert_layers.0";
        let q_w = store.tensor_shaped(
            &format!("{prefix}.attention.query.weight"),
            &[HIDDEN, HIDDEN],
        )?;
        let q_b = store.tensor_shaped(&format!("{prefix}.attention.query.bias"), &[HIDDEN])?;
        let k_w =
            store.tensor_shaped(&format!("{prefix}.attention.key.weight"), &[HIDDEN, HIDDEN])?;
        let k_b = store.tensor_shaped(&format!("{prefix}.attention.key.bias"), &[HIDDEN])?;
        let v_w = store.tensor_shaped(
            &format!("{prefix}.attention.value.weight"),
            &[HIDDEN, HIDDEN],
        )?;
        let v_b = store.tensor_shaped(&format!("{prefix}.attention.value.bias"), &[HIDDEN])?;
        let o_w = store.tensor_shaped(
            &format!("{prefix}.attention.dense.weight"),
            &[HIDDEN, HIDDEN],
        )?;
        let o_b = store.tensor_shaped(&format!("{prefix}.attention.dense.bias"), &[HIDDEN])?;
        let attn_ln_g =
            store.tensor_shaped(&format!("{prefix}.attention.LayerNorm.weight"), &[HIDDEN])?;
        let attn_ln_b =
            store.tensor_shaped(&format!("{prefix}.attention.LayerNorm.bias"), &[HIDDEN])?;
        let ffn_w = store.tensor_shaped(&format!("{prefix}.ffn.weight"), &[FFN_HIDDEN, HIDDEN])?;
        let ffn_b = store.tensor_shaped(&format!("{prefix}.ffn.bias"), &[FFN_HIDDEN])?;
        let ffn_out_w = store.tensor_shaped(
            &format!("{prefix}.ffn_output.weight"),
            &[HIDDEN, FFN_HIDDEN],
        )?;
        let ffn_out_b = store.tensor_shaped(&format!("{prefix}.ffn_output.bias"), &[HIDDEN])?;
        let full_ln_g =
            store.tensor_shaped(&format!("{prefix}.full_layer_layer_norm.weight"), &[HIDDEN])?;
        let full_ln_b =
            store.tensor_shaped(&format!("{prefix}.full_layer_layer_norm.bias"), &[HIDDEN])?;

        // --- Pooler (768 → 768) ---
        let pooler_w = store.tensor_shaped("bert.module.pooler.weight", &[HIDDEN, HIDDEN])?;
        let pooler_b = store.tensor_shaped("bert.module.pooler.bias", &[HIDDEN])?;

        // --- Downstream projection (768 → 512) ---
        let proj_w = store.tensor_shaped("bert_encoder.module.weight", &[OUT_DIM, HIDDEN])?;
        let proj_b = store.tensor_shaped("bert_encoder.module.bias", &[OUT_DIM])?;

        // Head-dim shape check: HIDDEN must be divisible by N_HEADS (64 = 768/12).
        // This is baked into the ALBERT-base convention; violate it and the
        // per-head slicing below is silently corrupt.
        if HIDDEN % N_HEADS != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro bert: HIDDEN ({HIDDEN}) not divisible by N_HEADS ({N_HEADS}) — \
                 head_dim would truncate"
            )));
        }

        Ok(Self {
            word_emb,
            pos_emb,
            token_type_emb,
            emb_ln: BertLayerNorm::new(emb_ln_g, emb_ln_b, EMBED_SIZE),
            mapping_in: BertLinear::from_pytorch(mapping_w, mapping_b, HIDDEN, EMBED_SIZE),
            shared_layer: BertSharedLayer {
                query: BertLinear::from_pytorch(q_w, q_b, HIDDEN, HIDDEN),
                key: BertLinear::from_pytorch(k_w, k_b, HIDDEN, HIDDEN),
                value: BertLinear::from_pytorch(v_w, v_b, HIDDEN, HIDDEN),
                attn_output: BertLinear::from_pytorch(o_w, o_b, HIDDEN, HIDDEN),
                attn_ln: BertLayerNorm::new(attn_ln_g, attn_ln_b, HIDDEN),
                ffn: BertLinear::from_pytorch(ffn_w, ffn_b, FFN_HIDDEN, HIDDEN),
                ffn_output: BertLinear::from_pytorch(ffn_out_w, ffn_out_b, HIDDEN, FFN_HIDDEN),
                full_ln: BertLayerNorm::new(full_ln_g, full_ln_b, HIDDEN),
            },
            pooler: BertLinear::from_pytorch(pooler_w, pooler_b, HIDDEN, HIDDEN),
            projection: BertLinear::from_pytorch(proj_w, proj_b, OUT_DIM, HIDDEN),
        })
    }

    /// Alias for [`Self::new`], mirroring the other kokoro component loaders
    /// so a future `KokoroTts::from_gguf_with_policy` wire-up (out of scope
    /// for this file) can call `Bert::load` consistently.
    #[allow(dead_code)] // consumed by the future KokoroTts wire-up
    pub(crate) fn load(store: &TensorStore, config: &KokoroConfig) -> Result<Self> {
        Self::new(store, config)
    }

    /// Runs the PL-BERT forward for one phoneme id sequence and returns a
    /// `[t · OUT_DIM]` row-major buffer (implicit shape `[t, 512]`).
    ///
    /// Pipeline:
    ///
    /// 1. Embedding sum: `word[id] + position[i] + token_type[0]` → `[t, 128]`.
    /// 2. LayerNorm over the innermost axis → `[t, 128]`.
    /// 3. `mapping_in` Linear → `[t, 768]`.
    /// 4. For each of `N_LAYERS = 12`, apply the shared ALBERT block:
    ///    MHA (12 heads, head_dim = 64) → residual + LN → FFN (768 → 2048 →
    ///    gelu_new → 768) → residual + LN.
    /// 5. `bert_encoder` projection on the last hidden state → `[t, 512]`
    ///    (upstream `CustomAlbert` discards the pooler — `modules.py:180-183`).
    ///
    /// # Errors
    ///
    /// * empty `phoneme_ids` — a 0-length sequence has no downstream use;
    /// * any id `< 0` or `≥ N_VOCAB` — silently clamping would hide a G2P bug
    ///   (FR-EX-08);
    /// * `phoneme_ids.len() > MAX_POS` — the position-embedding table has only
    ///   `MAX_POS = 512` rows.
    #[allow(dead_code)] // consumed by the future KokoroTts / prosody wire-up
    pub(crate) fn forward(&self, phoneme_ids: &[i64]) -> Result<Vec<f32>> {
        let t = phoneme_ids.len();
        if t == 0 {
            return Err(VokraError::InvalidArgument(
                "kokoro bert: empty phoneme id sequence".to_owned(),
            ));
        }
        if t > MAX_POS {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro bert: sequence length {t} exceeds MAX_POS {MAX_POS}"
            )));
        }

        let compute = Compute::cpu();

        // 1. Embedding sum → [t, EMBED_SIZE] row-major.
        //    Kokoro uses a single segment, so token_type_id = 0 for every position.
        let mut embeds = vec![0.0f32; t * EMBED_SIZE];
        for (i, &id) in phoneme_ids.iter().enumerate() {
            if id < 0 || (id as usize) >= N_VOCAB {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro bert: phoneme id {id} out of range 0..{N_VOCAB}"
                )));
            }
            let word_src = (id as usize) * EMBED_SIZE;
            let pos_src = i * EMBED_SIZE;
            let type_src = 0; // token_type_id = 0
            let dst = i * EMBED_SIZE;
            for c in 0..EMBED_SIZE {
                embeds[dst + c] = self.word_emb[word_src + c]
                    + self.pos_emb[pos_src + c]
                    + self.token_type_emb[type_src + c];
            }
        }

        // 2. LayerNorm → [t, EMBED_SIZE].
        let mut emb_ln_out = vec![0.0f32; t * EMBED_SIZE];
        self.emb_ln
            .forward_into(&compute, &embeds, t, &mut emb_ln_out)?;

        // 3. mapping_in Linear (128 → 768) → [t, HIDDEN].
        let mut hidden = vec![0.0f32; t * HIDDEN];
        self.mapping_in
            .linear_into(&compute, &emb_ln_out, t, &mut hidden)?;

        // 4. Shared ALBERT block × N_LAYERS (= 12, ALBERT weight sharing).
        for _layer_idx in 0..N_LAYERS {
            hidden = self.shared_layer_forward(&compute, &hidden, t)?;
        }

        // 5. bert_encoder projection (768 → 512) → [t, OUT_DIM], applied to
        //    the LAST HIDDEN STATE directly. Upstream `CustomAlbert.forward`
        //    (`modules.py:180-183`) returns `outputs.last_hidden_state` — the
        //    ALBERT pooler is computed by `AlbertModel` but DISCARDED, and
        //    `KModel.forward_with_tokens` (`model.py:102-103`) feeds that
        //    last-hidden-state to `self.bert_encoder`. The pre-fix per-token
        //    `pooler + tanh` insertion here was part of the P1 `bert`
        //    divergence (2026-07-16 real-weight eval). The pooler tensors are
        //    still loaded strictly (manifest completeness, FR-EX-08) but not
        //    applied — see the `pooler` field note.
        let mut out = vec![0.0f32; t * OUT_DIM];
        self.projection
            .linear_into(&compute, &hidden, t, &mut out)?;

        Ok(out)
    }

    /// One ALBERT block: multi-head self-attention (with residual + LN) followed
    /// by an FFN sub-block (with residual + LN).
    ///
    /// `hidden` is `[t · HIDDEN]` row-major; returns the block output as an
    /// owning `Vec` of the same length. The block is applied `N_LAYERS` times
    /// with the same weights (ALBERT sharing), so accepting/returning `Vec`s
    /// keeps the outer loop's ownership simple.
    fn shared_layer_forward(
        &self,
        compute: &Compute,
        hidden: &[f32],
        t: usize,
    ) -> Result<Vec<f32>> {
        let d = HIDDEN;
        let n_heads = N_HEADS;
        let head_dim = d / n_heads;
        let scale = (head_dim as f32).powf(-0.5);
        let layer = &self.shared_layer;

        // Q / K / V projections: hidden @ W_{q,k,v}^T + b_{q,k,v} → [t, d]
        let mut q = vec![0.0f32; t * d];
        let mut k = vec![0.0f32; t * d];
        let mut v = vec![0.0f32; t * d];
        layer.query.linear_into(compute, hidden, t, &mut q)?;
        layer.key.linear_into(compute, hidden, t, &mut k)?;
        layer.value.linear_into(compute, hidden, t, &mut v)?;

        // Scale Q by 1/sqrt(head_dim). Whisper applies the scale to Q rather
        // than to the scores; the two are mathematically identical and this
        // path mirrors the reference for numerical parity.
        for val in q.iter_mut() {
            *val *= scale;
        }

        // Per-head attention: gather qh / vh / kh_t → softmax(qh · kh_t) · vh
        // → scatter back into [t, d]. Reusing per-head scratch across heads
        // keeps allocation to a single pass per layer.
        let mut context = vec![0.0f32; t * d];
        let mut qh = vec![0.0f32; t * head_dim];
        let mut vh = vec![0.0f32; t * head_dim];
        let mut kh_t = vec![0.0f32; head_dim * t];
        let mut scores = vec![0.0f32; t * t];
        let mut probs = vec![0.0f32; t * t];
        let mut ctx_h = vec![0.0f32; t * head_dim];

        for h in 0..n_heads {
            let c0 = h * head_dim;
            // Gather qh [t, head_dim] and vh [t, head_dim]; transpose k → kh_t [head_dim, t].
            for i in 0..t {
                qh[i * head_dim..i * head_dim + head_dim]
                    .copy_from_slice(&q[i * d + c0..i * d + c0 + head_dim]);
            }
            for j in 0..t {
                vh[j * head_dim..j * head_dim + head_dim]
                    .copy_from_slice(&v[j * d + c0..j * d + c0 + head_dim]);
                for c in 0..head_dim {
                    kh_t[c * t + j] = k[j * d + c0 + c];
                }
            }
            // scores [t, t] = qh [t, head_dim] · kh_t [head_dim, t].
            compute.gemm_f32(t, t, head_dim, &qh, &kh_t, None, &mut scores)?;
            // BERT/ALBERT self-attention has no causal mask — all positions
            // attend to all positions. The prosody path uses the whole context
            // (no autoregressive step here).
            compute.softmax_f32(&scores, &mut probs, t, t)?;
            // ctx_h [t, head_dim] = probs [t, t] · vh [t, head_dim].
            compute.gemm_f32(t, head_dim, t, &probs, &vh, None, &mut ctx_h)?;
            // Scatter ctx_h back into the full-width context buffer.
            for i in 0..t {
                context[i * d + c0..i * d + c0 + head_dim]
                    .copy_from_slice(&ctx_h[i * head_dim..i * head_dim + head_dim]);
            }
        }

        // Attention output projection: context → attn_out (bias fused by GEMM).
        let mut attn_out = vec![0.0f32; t * d];
        layer
            .attn_output
            .linear_into(compute, &context, t, &mut attn_out)?;

        // Residual + LN (post-norm, matching BERT / ALBERT).
        for i in 0..(t * d) {
            attn_out[i] += hidden[i];
        }
        let mut attn_normed = vec![0.0f32; t * d];
        layer
            .attn_ln
            .forward_into(compute, &attn_out, t, &mut attn_normed)?;

        // FFN: attn_normed → ffn_h (GEMM + bias) → gelu_new → HIDDEN
        // (GEMM + bias). `gelu_new` (tanh approximation) is the AlbertConfig
        // default `hidden_act` and Kokoro's `config.json` does not override
        // it — the erf-exact `gelu` used pre-fix was part of the P1 `bert`
        // divergence (2026-07-16 real-weight eval).
        let mut ffn_h = vec![0.0f32; t * FFN_HIDDEN];
        layer
            .ffn
            .linear_into(compute, &attn_normed, t, &mut ffn_h)?;
        for val in ffn_h.iter_mut() {
            *val = gelu_new(*val);
        }
        let mut ffn_out = vec![0.0f32; t * d];
        layer
            .ffn_output
            .linear_into(compute, &ffn_h, t, &mut ffn_out)?;

        // Residual + LN (post-norm).
        for i in 0..(t * d) {
            ffn_out[i] += attn_normed[i];
        }
        let mut block_out = vec![0.0f32; t * d];
        layer
            .full_ln
            .forward_into(compute, &ffn_out, t, &mut block_out)?;

        Ok(block_out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::{
        KEY_HIDDEN_DIM, KEY_ISTFT_HOP, KEY_ISTFT_N_FFT, KEY_ISTFT_WIN_LENGTH, KEY_N_DECODER_LAYERS,
        KEY_N_TEXT_LAYERS, KEY_NUM_VOICES, KEY_PHONEME_SYMBOLS, KEY_SAMPLE_RATE, KEY_STYLE_DIM,
        KEY_VOICE_NAMES,
    };
    use super::*;
    use vokra_core::gguf::{
        GgmlType, GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType,
    };

    fn zeros_bytes(n: usize) -> Vec<u8> {
        vec![0u8; n * 4]
    }

    /// Deterministic ramp so a mis-index during load is instantly visible in
    /// the output. Payload = `seed + i · step` as GGUF-ready LE bytes.
    fn ramp_bytes(n: usize, seed: f32, step: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|i| (seed + i as f32 * step).to_le_bytes())
            .collect()
    }

    fn str_array(items: &[&str]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: items
                .iter()
                .map(|s| GgufMetadataValue::String((*s).to_owned()))
                .collect(),
        })
    }

    /// Minimal `KokoroConfig`-carrying builder (the [`Bert::new`] loader does
    /// not read any `vokra.kokoro.*` metadata, but `KokoroConfig::from_gguf`
    /// requires all keys before we can call the loader).
    fn add_kokoro_config(b: &mut GgufBuilder) {
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, 8);
        b.add_u32(KEY_NUM_VOICES, 2);
        b.add_u32(KEY_HIDDEN_DIM, 16);
        b.add_u32(KEY_N_TEXT_LAYERS, 2);
        b.add_u32(KEY_N_DECODER_LAYERS, 2);
        b.add_u32(KEY_ISTFT_N_FFT, 20);
        b.add_u32(KEY_ISTFT_HOP, 5);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 20);
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["a", "b"]));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af"]));
    }

    /// Adds every `bert.module.*` and `bert_encoder.module.*` tensor at its
    /// documented shape.
    ///
    /// `ramp_weights = true` fills the embeddings + the shared block with a
    /// non-zero deterministic ramp so a mis-index during forward propagates
    /// visibly; `false` uses zeros for the "loads and stays finite" smoke.
    fn add_bert_tensors(b: &mut GgufBuilder, ramp_weights: bool) {
        let payload = |n: usize, seed: f32| -> Vec<u8> {
            if ramp_weights {
                ramp_bytes(n, seed, 1e-4)
            } else {
                zeros_bytes(n)
            }
        };

        // --- Embeddings ---
        b.add_tensor(
            "bert.module.embeddings.word_embeddings.weight",
            GgmlType::F32,
            vec![N_VOCAB as u64, EMBED_SIZE as u64],
            payload(N_VOCAB * EMBED_SIZE, 0.01),
        )
        .expect("word_emb");
        b.add_tensor(
            "bert.module.embeddings.position_embeddings.weight",
            GgmlType::F32,
            vec![MAX_POS as u64, EMBED_SIZE as u64],
            payload(MAX_POS * EMBED_SIZE, 0.02),
        )
        .expect("pos_emb");
        b.add_tensor(
            "bert.module.embeddings.token_type_embeddings.weight",
            GgmlType::F32,
            vec![N_TOKEN_TYPES as u64, EMBED_SIZE as u64],
            payload(N_TOKEN_TYPES * EMBED_SIZE, 0.03),
        )
        .expect("type_emb");
        // LayerNorm γ = 1, β = 0 — non-trivial affine that keeps the numeric
        // path bounded (γ = 0 would silently skip the scale-path regression).
        b.add_tensor(
            "bert.module.embeddings.LayerNorm.weight",
            GgmlType::F32,
            vec![EMBED_SIZE as u64],
            (0..EMBED_SIZE).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
        )
        .expect("emb_ln_g");
        b.add_tensor(
            "bert.module.embeddings.LayerNorm.bias",
            GgmlType::F32,
            vec![EMBED_SIZE as u64],
            zeros_bytes(EMBED_SIZE),
        )
        .expect("emb_ln_b");

        // --- Encoder: mapping_in ---
        b.add_tensor(
            "bert.module.encoder.embedding_hidden_mapping_in.weight",
            GgmlType::F32,
            vec![HIDDEN as u64, EMBED_SIZE as u64],
            payload(HIDDEN * EMBED_SIZE, 0.04),
        )
        .expect("mapping_w");
        b.add_tensor(
            "bert.module.encoder.embedding_hidden_mapping_in.bias",
            GgmlType::F32,
            vec![HIDDEN as u64],
            zeros_bytes(HIDDEN),
        )
        .expect("mapping_b");

        // --- Encoder: shared block ---
        let prefix = "bert.module.encoder.albert_layer_groups.0.albert_layers.0";
        // Attention Q/K/V/O
        for (name, seed) in [
            ("attention.query.weight", 0.05f32),
            ("attention.key.weight", 0.06),
            ("attention.value.weight", 0.07),
            ("attention.dense.weight", 0.08),
        ] {
            b.add_tensor(
                &format!("{prefix}.{name}"),
                GgmlType::F32,
                vec![HIDDEN as u64, HIDDEN as u64],
                payload(HIDDEN * HIDDEN, seed),
            )
            .expect(name);
        }
        for name in [
            "attention.query.bias",
            "attention.key.bias",
            "attention.value.bias",
            "attention.dense.bias",
        ] {
            b.add_tensor(
                &format!("{prefix}.{name}"),
                GgmlType::F32,
                vec![HIDDEN as u64],
                zeros_bytes(HIDDEN),
            )
            .expect(name);
        }
        // attention.LayerNorm γ / β
        b.add_tensor(
            &format!("{prefix}.attention.LayerNorm.weight"),
            GgmlType::F32,
            vec![HIDDEN as u64],
            (0..HIDDEN).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
        )
        .expect("attn_ln_w");
        b.add_tensor(
            &format!("{prefix}.attention.LayerNorm.bias"),
            GgmlType::F32,
            vec![HIDDEN as u64],
            zeros_bytes(HIDDEN),
        )
        .expect("attn_ln_b");
        // FFN
        b.add_tensor(
            &format!("{prefix}.ffn.weight"),
            GgmlType::F32,
            vec![FFN_HIDDEN as u64, HIDDEN as u64],
            payload(FFN_HIDDEN * HIDDEN, 0.09),
        )
        .expect("ffn_w");
        b.add_tensor(
            &format!("{prefix}.ffn.bias"),
            GgmlType::F32,
            vec![FFN_HIDDEN as u64],
            zeros_bytes(FFN_HIDDEN),
        )
        .expect("ffn_b");
        b.add_tensor(
            &format!("{prefix}.ffn_output.weight"),
            GgmlType::F32,
            vec![HIDDEN as u64, FFN_HIDDEN as u64],
            payload(HIDDEN * FFN_HIDDEN, 0.10),
        )
        .expect("ffn_out_w");
        b.add_tensor(
            &format!("{prefix}.ffn_output.bias"),
            GgmlType::F32,
            vec![HIDDEN as u64],
            zeros_bytes(HIDDEN),
        )
        .expect("ffn_out_b");
        // full_layer_layer_norm γ / β
        b.add_tensor(
            &format!("{prefix}.full_layer_layer_norm.weight"),
            GgmlType::F32,
            vec![HIDDEN as u64],
            (0..HIDDEN).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
        )
        .expect("full_ln_w");
        b.add_tensor(
            &format!("{prefix}.full_layer_layer_norm.bias"),
            GgmlType::F32,
            vec![HIDDEN as u64],
            zeros_bytes(HIDDEN),
        )
        .expect("full_ln_b");

        // --- Pooler ---
        b.add_tensor(
            "bert.module.pooler.weight",
            GgmlType::F32,
            vec![HIDDEN as u64, HIDDEN as u64],
            payload(HIDDEN * HIDDEN, 0.11),
        )
        .expect("pooler_w");
        b.add_tensor(
            "bert.module.pooler.bias",
            GgmlType::F32,
            vec![HIDDEN as u64],
            zeros_bytes(HIDDEN),
        )
        .expect("pooler_b");

        // --- Downstream projection ---
        b.add_tensor(
            "bert_encoder.module.weight",
            GgmlType::F32,
            vec![OUT_DIM as u64, HIDDEN as u64],
            payload(OUT_DIM * HIDDEN, 0.12),
        )
        .expect("proj_w");
        b.add_tensor(
            "bert_encoder.module.bias",
            GgmlType::F32,
            vec![OUT_DIM as u64],
            zeros_bytes(OUT_DIM),
        )
        .expect("proj_b");
    }

    fn build_bert(ramp_weights: bool) -> Bert {
        let mut b = GgufBuilder::new();
        add_kokoro_config(&mut b);
        add_bert_tensors(&mut b, ramp_weights);
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("config");
        let store = TensorStore::new(file);
        Bert::new(&store, &config).expect("bert loads")
    }

    /// The T13-beta loader must bind every `bert.module.*` +
    /// `bert_encoder.module.*` tensor at its documented shape; a synthetic GGUF
    /// carrying all 27 tensors loads successfully.
    #[test]
    fn loads_all_tensors_from_synthetic_gguf() {
        let _bert = build_bert(/*ramp_weights=*/ false);
    }

    /// Forward output shape must be `[t, OUT_DIM]` (implicit — the returned
    /// `Vec` is `[t * OUT_DIM]`). Every element must be finite.
    #[test]
    fn forward_returns_expected_shape() {
        let bert = build_bert(/*ramp_weights=*/ false);
        let out = bert.forward(&[1, 2, 3]).expect("forward ok");
        assert_eq!(
            out.len(),
            3 * OUT_DIM,
            "output len must equal t * OUT_DIM (t=3)"
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "forward must produce only finite values"
        );
    }

    /// The forward path is deterministic: identical inputs → bit-identical
    /// outputs. Uses ramp weights so the output is non-trivial.
    #[test]
    fn forward_is_deterministic_across_two_calls() {
        let bert = build_bert(/*ramp_weights=*/ true);
        let a = bert.forward(&[1, 2, 3]).expect("first");
        let b = bert.forward(&[1, 2, 3]).expect("second");
        assert_eq!(
            a, b,
            "bert forward must be bit-exact deterministic for identical inputs"
        );
    }

    /// Empty input is rejected — a 0-length sequence has no downstream use.
    #[test]
    fn forward_rejects_empty_input() {
        let bert = build_bert(false);
        let err = bert.forward(&[]).expect_err("empty input must error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    /// An id `< 0` or `≥ N_VOCAB` fails loudly rather than silently clamping
    /// (FR-EX-08 — a G2P bug producing a stray id must surface, not degrade).
    #[test]
    fn forward_rejects_out_of_range_id() {
        let bert = build_bert(false);
        let err = bert
            .forward(&[1, N_VOCAB as i64, 3])
            .expect_err("out-of-range id must error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn forward_rejects_negative_id() {
        let bert = build_bert(false);
        let err = bert
            .forward(&[-1, 1, 2])
            .expect_err("negative id must error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    /// A phoneme sequence longer than `MAX_POS` overflows the position table
    /// and must be rejected up front rather than reading past the end.
    #[test]
    fn forward_rejects_sequence_beyond_max_pos() {
        let bert = build_bert(false);
        let too_long: Vec<i64> = (0..(MAX_POS + 1) as i64).map(|i| i % 10).collect();
        let err = bert
            .forward(&too_long)
            .expect_err("sequence beyond MAX_POS must error");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("MAX_POS"), "error should name MAX_POS: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// A missing tensor fails at the first `tensor_shaped` call with a message
    /// naming the offending tensor — the FR-EX-08 red line for architecture
    /// drift. The test omits `word_embeddings.weight` from an otherwise
    /// complete builder.
    #[test]
    fn new_reports_missing_word_embedding_tensor() {
        let mut b = GgufBuilder::new();
        add_kokoro_config(&mut b);
        // Deliberately skip word_embeddings; add the rest via
        // `add_bert_tensors` and then rebuild to strip it — since
        // `GgufBuilder` has no delete API, we build the whole set into `b2`
        // minus the word embedding by hand.
        let mut b2 = GgufBuilder::new();
        add_kokoro_config(&mut b2);
        // Position + token_type + LN (skip word_embeddings).
        b2.add_tensor(
            "bert.module.embeddings.position_embeddings.weight",
            GgmlType::F32,
            vec![MAX_POS as u64, EMBED_SIZE as u64],
            zeros_bytes(MAX_POS * EMBED_SIZE),
        )
        .expect("pos_emb");
        b2.add_tensor(
            "bert.module.embeddings.token_type_embeddings.weight",
            GgmlType::F32,
            vec![N_TOKEN_TYPES as u64, EMBED_SIZE as u64],
            zeros_bytes(N_TOKEN_TYPES * EMBED_SIZE),
        )
        .expect("type_emb");
        b2.add_tensor(
            "bert.module.embeddings.LayerNorm.weight",
            GgmlType::F32,
            vec![EMBED_SIZE as u64],
            zeros_bytes(EMBED_SIZE),
        )
        .expect("emb_ln_g");
        b2.add_tensor(
            "bert.module.embeddings.LayerNorm.bias",
            GgmlType::F32,
            vec![EMBED_SIZE as u64],
            zeros_bytes(EMBED_SIZE),
        )
        .expect("emb_ln_b");
        // `b` unused after the strip — silence dead_code.
        let _ = &mut b;

        let file = GgufFile::parse(b2.to_bytes().expect("serialize")).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("config");
        let store = TensorStore::new(file);
        let err = Bert::new(&store, &config).expect_err("missing word_emb must fail");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("bert.module.embeddings.word_embeddings.weight"),
                    "error should name the missing tensor; got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// A wrong-shape tensor fails with a message naming the offending tensor
    /// AND the expected shape — the same FR-EX-08 red line as the missing-name
    /// case. Uses a `position_embeddings` table with the wrong first axis.
    #[test]
    fn new_reports_wrong_shape_tensor() {
        let mut b = GgufBuilder::new();
        add_kokoro_config(&mut b);
        // Word embedding: correct shape.
        b.add_tensor(
            "bert.module.embeddings.word_embeddings.weight",
            GgmlType::F32,
            vec![N_VOCAB as u64, EMBED_SIZE as u64],
            zeros_bytes(N_VOCAB * EMBED_SIZE),
        )
        .expect("word_emb");
        // Position embedding: WRONG shape — first axis 256 vs expected 512.
        // The loader must reject this rather than silently truncating.
        b.add_tensor(
            "bert.module.embeddings.position_embeddings.weight",
            GgmlType::F32,
            vec![256_u64, EMBED_SIZE as u64],
            zeros_bytes(256 * EMBED_SIZE),
        )
        .expect("wrong-shape pos_emb");
        // Remaining tensors do not need to be present — the loader will
        // fail on the wrong-shape check before reaching them.

        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("config");
        let store = TensorStore::new(file);
        let err = Bert::new(&store, &config).expect_err("wrong shape must fail");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("bert.module.embeddings.position_embeddings.weight"),
                    "error should name the offending tensor; got: {msg}"
                );
                assert!(
                    msg.contains(&MAX_POS.to_string()),
                    "error should mention the expected MAX_POS ({MAX_POS}); got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }
}
