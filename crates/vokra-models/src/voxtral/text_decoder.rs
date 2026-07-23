//! Voxtral text decoder — Mistral LLaMA-style transformer.
//!
//! # Structural summary (from the upstream Mistral release)
//!
//! - **Pre-norm** blocks: input → RMSNorm → attention → residual → RMSNorm
//!   → SwiGLU FFN → residual;
//! - **GQA** attention: `n_head_q` query heads, `n_head_kv` key/value heads
//!   (`n_head_q % n_head_kv == 0`, key/value are broadcast `n_head_q /
//!   n_head_kv` times). `head_dim` is **decoupled** from `hidden_dim /
//!   n_head_q`: the shipping `Voxtral-Mini-3B-2507` has `hidden_dim = 3072`
//!   but 32 query heads × `head_dim = 128`, i.e. a `[4096, 3072]` Q
//!   projection and a `[3072, 4096]` O projection (2026-07-16 real-weight
//!   eval — the pre-fix loader assumed a square `[d, d]` Q/O and rejected
//!   the real checkpoint);
//! - **RoPE** applied to query & key before the score matmul;
//! - **SwiGLU** FFN: `w2(silu(w1(x)) * w3(x))` (equivalently
//!   `down(silu(gate(x)) * up(x))`);
//! - **RMSNorm** with the checkpoint's ε (Mistral ships `1e-5`);
//! - **Logits head**: the shipping mini release carries an **untied**
//!   `lm_head.weight` (byte-compared ≠ the token embedding); the loader
//!   binds it when present and ties the token embedding only when the
//!   tensor is genuinely absent (upstream `tie_word_embeddings` semantics).
//!
//! # Scope (M3-10-T09 / T10 + follow-up autoregressive forward)
//!
//! This file:
//! - reads the Mistral text decoder weights out of the GGUF (recognising
//!   both the packaged Voxtral prefix `language_model.model.*` and the plain
//!   Mistral prefix `model.*`);
//! - exposes small, unit-testable Rust primitives ([`rms_norm`],
//!   [`silu_inplace`], [`rope_apply`]) that the block forward composes;
//! - implements the full autoregressive block forward — pre-norm, GQA
//!   self-attention with RoPE + causal mask + per-block K/V cache append,
//!   pre-norm SwiGLU FFN, final RMSNorm and tied-logits head — through the
//!   Compute seam so a GPU backend runs the same GEMM path (see
//!   [`forward_step`]);
//! - the KV cache lives on the caller side (`TextDecoderSession`) — one
//!   `KvCache` with width `n_head_kv * head_dim` per layer.
//!
//! The primitives (RMSNorm / SwiGLU / RoPE) are covered by internal oracle
//! tests; the block forward is covered by shape / determinism smoke tests on
//! synthesized weights. Real-checkpoint parity is deferred to a follow-up
//! ticket (T19+) that requires an upstream Voxtral safetensors dump.

use std::sync::{Arc, Mutex, MutexGuard};

use vokra_core::gguf::{GgufFile, GgufTensorInfo};
use vokra_core::{KvCache, Result, VokraError};

use super::VoxtralConfig;
use crate::compute::Compute;
use crate::mapped_weights::{MappedModel, lock_scratch, mapped_info, transpose_widen, widen_into};

/// Error identity for the Voxtral mapped store: the resident alternative is
/// [`TextDecoder::load`], which goes through `GgufFile::tensor_f32` and so
/// dequantizes block-wise payloads the mapped path cannot address per element.
const VOXTRAL_MAPPED: MappedModel = MappedModel {
    name: "voxtral",
    resident_entry: "TextDecoder::load",
};

/// A `nn.Linear` decoded for direct row-major GEMM (`w_t` is `[in, out]`).
///
/// Mistral decoder projections are always **bias-less** (`bias = None`).
///
/// `in_features` / `out_features` are load-time invariants — kept alongside
/// `w_t` for external validation (e.g. audio adapter follow-up sanity
/// checks); silenced from `dead_code` because the block forward reads the
/// shapes off the config, not this struct.
#[allow(dead_code)]
pub(crate) struct Linear {
    pub(crate) w_t: Vec<f32>,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
}

/// A block's four attention projections (decoded to `[in, out]` for
/// row-major GEMM). GQA: `q` is `[d, n_head_q*head_dim]`, `k` / `v` are
/// `[d, n_head_kv*head_dim]`, `o` is `[n_head_q*head_dim, d]`.
/// `n_head_q*head_dim` equals `d` only when the checkpoint ties `head_dim`
/// to `d / n_head_q` — NOT true of the shipping Voxtral mini (4096 vs 3072).
pub(crate) struct GqaAttention {
    pub(crate) q: Linear,
    pub(crate) k: Linear,
    pub(crate) v: Linear,
    pub(crate) o: Linear,
}

/// SwiGLU FFN weights: `w2(silu(w1(x)) * w3(x))`.
pub(crate) struct SwiGluFfn {
    pub(crate) gate: Linear, // = w1
    pub(crate) up: Linear,   // = w3
    pub(crate) down: Linear, // = w2
}

/// One Mistral decoder block.
pub(crate) struct DecoderBlock {
    /// RMSNorm γ vector (no bias — RMSNorm is scale-only, `[hidden_dim]`).
    pub(crate) attn_norm_gamma: Vec<f32>,
    pub(crate) attn: GqaAttention,
    pub(crate) ffn_norm_gamma: Vec<f32>,
    pub(crate) ffn: SwiGluFfn,
}

/// Where one mapped layer's nine tensors live inside the GGUF.
///
/// Voxtral stores every projection separately (no fused QKV or gate/up pack,
/// unlike Moshi), so a layer is nine independent descriptors rather than six.
struct MappedTextLayerLocs {
    attn_norm: GgufTensorInfo,
    q: GgufTensorInfo,
    k: GgufTensorInfo,
    v: GgufTensorInfo,
    o: GgufTensorInfo,
    ffn_norm: GgufTensorInfo,
    gate: GgufTensorInfo,
    up: GgufTensorInfo,
    down: GgufTensorInfo,
}

/// Decoder blocks left **in the GGUF mapping** and widened to f32 one layer at
/// a time during the forward pass — the Voxtral counterpart of
/// `moshi::backbone::MappedTemporalBlocks`.
///
/// Motivation, measured on the real `mistralai/Voxtral-Mini-3B` GGUF: the
/// `language_model` tensors are 7.48 GiB stored (BF16) but **14.95 GiB once
/// widened to owned f32**, so [`TextDecoder::load`] cannot bind the decoder on
/// a 16 GiB machine at all. Mapping the blocks moves that bulk off the heap
/// and leaves only the head tensors resident.
///
/// - **Bind time** ([`Self::bind`]): all nine tensors of every layer are
///   resolved and shape/dtype-validated up front, so a malformed GGUF fails at
///   load rather than mid-stream (FR-EX-08).
/// - **Step time** ([`Self::materialize_into`]): the requested layer is
///   widened + transposed straight out of the mapped bytes into the reused
///   scratch block, producing **bit-identical** f32 values to the resident
///   binding — same widening formula, same `[out, in]` -> `[in, out]` index
///   math (pinned by `mapped_blocks_match_resident_bitwise`).
///
/// # What this does *not* bound
///
/// Only the per-layer blocks are mapped. `token_emb` and (when untied)
/// `lm_head` stay resident: at the shipping mini they are `[131072, 3072]`
/// each, i.e. ~1.5 GiB apiece. The honest ceiling is therefore
/// "~3 GiB of heads + one layer of scratch", not "bounded". Streaming the
/// `lm_head` GEMV out of the mapping is a separate change — it would have to
/// replace [`TextDecoder::output_head`]'s `&[f32]` contract.
///
/// The scratch sits behind a `Mutex` so the owning decoder stays `Send + Sync`
/// (`VoxtralAsr` is served concurrently from the server registry). Concurrent
/// sessions on one decoder therefore **serialize** their block forwards on
/// this lock — the bounded-memory trade; per-session scratch would multiply
/// the buffer per session.
pub struct MappedTextBlocks {
    file: Arc<GgufFile>,
    locs: Vec<MappedTextLayerLocs>,
    scratch: Mutex<DecoderBlock>,
    d: usize,
    q_hidden: usize,
    kv_hidden: usize,
    ffn_dim: usize,
}

impl std::fmt::Debug for MappedTextBlocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedTextBlocks")
            .field("n_layer", &self.locs.len())
            .field("d", &self.d)
            .field("q_hidden", &self.q_hidden)
            .field("kv_hidden", &self.kv_hidden)
            .field("ffn_dim", &self.ffn_dim)
            .finish()
    }
}

impl MappedTextBlocks {
    /// Resolves and validates every layer's nine tensors against `cfg`.
    ///
    /// `prefix` must be the same one [`pick_prefix`] chose for the resident
    /// path, so both bindings name identical tensors.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming any missing tensor, any element count
    /// that disagrees with the config shapes, or any payload dtype outside
    /// `{F32, BF16}` (a quantized GGUF must use the resident loader).
    pub fn bind(file: Arc<GgufFile>, cfg: &VoxtralConfig, prefix: &str) -> Result<Self> {
        let d = cfg.text.hidden_dim;
        let head_dim = cfg.text.head_dim();
        let q_hidden = cfg.text.n_head_q * head_dim;
        let kv_hidden = cfg.text.n_head_kv * head_dim;
        let ffn_dim = cfg.text.ffn_dim;
        if d == 0 || head_dim == 0 || q_hidden == 0 || kv_hidden == 0 || ffn_dim == 0 {
            return Err(bad(format!(
                "mapped bind needs non-zero shapes (d={d}, head_dim={head_dim}, \
                 q_hidden={q_hidden}, kv_hidden={kv_hidden}, ffn_dim={ffn_dim}) — \
                 re-convert with a converter that writes the full \
                 vokra.voxtral.text_decoder.* group"
            )));
        }
        let m = VOXTRAL_MAPPED;
        let mut locs = Vec::with_capacity(cfg.text.n_layer);
        for i in 0..cfg.text.n_layer {
            let p = format!("{prefix}layers.{i}");
            locs.push(MappedTextLayerLocs {
                attn_norm: mapped_info(&file, &format!("{p}.input_layernorm.weight"), d, m)?,
                q: mapped_info(
                    &file,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_hidden * d,
                    m,
                )?,
                k: mapped_info(
                    &file,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_hidden * d,
                    m,
                )?,
                v: mapped_info(
                    &file,
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_hidden * d,
                    m,
                )?,
                o: mapped_info(
                    &file,
                    &format!("{p}.self_attn.o_proj.weight"),
                    d * q_hidden,
                    m,
                )?,
                ffn_norm: mapped_info(
                    &file,
                    &format!("{p}.post_attention_layernorm.weight"),
                    d,
                    m,
                )?,
                gate: mapped_info(&file, &format!("{p}.mlp.gate_proj.weight"), ffn_dim * d, m)?,
                up: mapped_info(&file, &format!("{p}.mlp.up_proj.weight"), ffn_dim * d, m)?,
                down: mapped_info(&file, &format!("{p}.mlp.down_proj.weight"), d * ffn_dim, m)?,
            });
        }
        Ok(Self {
            file,
            locs,
            scratch: Mutex::new(empty_block(d, q_hidden, kv_hidden, ffn_dim)),
            d,
            q_hidden,
            kv_hidden,
            ffn_dim,
        })
    }

    /// Number of mapped layers (must equal `cfg.text.n_layer`).
    pub fn n_layer(&self) -> usize {
        self.locs.len()
    }

    /// Locks the shared materialization scratch (whole-forward hold).
    fn lock_scratch(&self) -> Result<MutexGuard<'_, DecoderBlock>> {
        lock_scratch(&self.scratch, VOXTRAL_MAPPED)
    }

    /// Widens + transposes `layer` out of the mapping into `scratch`.
    ///
    /// Each projection is stored `[out, in]` and consumed as `w_t = [in, out]`,
    /// exactly as the resident [`linear`] builds it — so the transpose
    /// arguments are `(rows = out, cols = in)`.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] if a payload's byte length disagrees with the
    /// bound shape (bind-time validation should have made this unreachable).
    fn materialize_into<'a>(
        &self,
        scratch: &'a mut DecoderBlock,
        layer: usize,
    ) -> Result<&'a DecoderBlock> {
        let locs = &self.locs[layer];
        let (d, q_hidden, kv_hidden, ffn_dim) =
            (self.d, self.q_hidden, self.kv_hidden, self.ffn_dim);
        let m = VOXTRAL_MAPPED;
        let f = &self.file;

        widen_into(
            f.tensor_bytes(&locs.attn_norm),
            locs.attn_norm.dtype,
            &mut scratch.attn_norm_gamma,
            m,
        )?;
        let t = |info: &GgufTensorInfo, rows, cols, dst: &mut Vec<f32>| {
            transpose_widen(f.tensor_bytes(info), info.dtype, rows, cols, dst, m)
        };
        t(&locs.q, q_hidden, d, &mut scratch.attn.q.w_t)?;
        t(&locs.k, kv_hidden, d, &mut scratch.attn.k.w_t)?;
        t(&locs.v, kv_hidden, d, &mut scratch.attn.v.w_t)?;
        t(&locs.o, d, q_hidden, &mut scratch.attn.o.w_t)?;
        widen_into(
            f.tensor_bytes(&locs.ffn_norm),
            locs.ffn_norm.dtype,
            &mut scratch.ffn_norm_gamma,
            m,
        )?;
        t(&locs.gate, ffn_dim, d, &mut scratch.ffn.gate.w_t)?;
        t(&locs.up, ffn_dim, d, &mut scratch.ffn.up.w_t)?;
        t(&locs.down, d, ffn_dim, &mut scratch.ffn.down.w_t)?;
        Ok(scratch)
    }
}

/// An all-empty block with the right shape metadata — the mapped scratch
/// before its first materialization. The `w_t` vectors size themselves on the
/// first `transpose_widen` (which `resize`s), so this allocates nothing.
fn empty_block(d: usize, q_hidden: usize, kv_hidden: usize, ffn_dim: usize) -> DecoderBlock {
    let lin = |in_features, out_features| Linear {
        w_t: Vec::new(),
        in_features,
        out_features,
    };
    DecoderBlock {
        attn_norm_gamma: Vec::new(),
        attn: GqaAttention {
            q: lin(d, q_hidden),
            k: lin(d, kv_hidden),
            v: lin(d, kv_hidden),
            o: lin(q_hidden, d),
        },
        ffn_norm_gamma: Vec::new(),
        ffn: SwiGluFfn {
            gate: lin(d, ffn_dim),
            up: lin(d, ffn_dim),
            down: lin(ffn_dim, d),
        },
    }
}

/// Rows of the vocab-sized head matrices, served straight out of the GGUF
/// mapping instead of as owned f32.
///
/// At the shipping Voxtral-Mini shape `token_emb` and the untied `lm_head` are
/// `[131072, 3072]` each — **~1.5 GiB apiece widened**, i.e. ~3 GiB of the
/// decoder's resident cost, more than the mapped blocks leave behind. Neither
/// is used in a way that needs the whole matrix live:
///
/// - `token_emb` is read **one row per token** (`embed_row_into`).
/// - `lm_head` is consumed by a per-row GEMV, so it can be walked in **row
///   chunks** (`logits_into`): `out[v] = Σ_c head[v, c] * x[c]` involves no
///   cross-row term, and each row's accumulation order is untouched, so the
///   chunked result is **bit-identical** to the resident one.
///
/// Scratches sit behind mutexes so the owning decoder stays `Send + Sync`.
pub struct MappedHeads {
    file: Arc<GgufFile>,
    token_emb: GgufTensorInfo,
    /// `None` = the checkpoint ties the head to `token_emb` (upstream
    /// `tie_word_embeddings`), exactly as the resident `lm_head: Option<_>`.
    lm_head: Option<GgufTensorInfo>,
    vocab: usize,
    d: usize,
    /// One widened head chunk (`HEAD_CHUNK_ROWS x d`).
    chunk: Mutex<Vec<f32>>,
}

/// Rows of the head matrix widened per GEMV chunk.
///
/// 512 rows x 3072 f32 = 6 MiB — small enough to stay in L2/L3 across the
/// chunk's GEMV, and large enough that the per-chunk call overhead is noise
/// against 512 dot products of length `d`.
const HEAD_CHUNK_ROWS: usize = 512;

impl std::fmt::Debug for MappedHeads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedHeads")
            .field("vocab", &self.vocab)
            .field("d", &self.d)
            .field("untied_lm_head", &self.lm_head.is_some())
            .finish()
    }
}

impl MappedHeads {
    /// Resolves and validates the head tensors without widening them.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] on a missing tensor, an element count that
    /// disagrees with `[vocab, d]`, or a dtype the mapped path cannot widen.
    fn bind(file: Arc<GgufFile>, prefix: &str, vocab: usize, d: usize) -> Result<Self> {
        let m = VOXTRAL_MAPPED;
        let token_emb = mapped_info(&file, &format!("{prefix}embed_tokens.weight"), vocab * d, m)?;
        let lm_head_name = format!("{}lm_head.weight", lm_head_base(prefix));
        let lm_head = if file.tensor_info(&lm_head_name).is_some() {
            Some(mapped_info(&file, &lm_head_name, vocab * d, m)?)
        } else {
            None
        };
        Ok(Self {
            file,
            token_emb,
            lm_head,
            vocab,
            d,
            chunk: Mutex::new(Vec::new()),
        })
    }

    /// The tensor the logits GEMV reads: the untied head when present, else
    /// the tied token embedding — the mapped mirror of
    /// [`TextDecoder::output_head`].
    fn head_info(&self) -> &GgufTensorInfo {
        self.lm_head.as_ref().unwrap_or(&self.token_emb)
    }

    /// Widens row `row` of `info` into `dst` (exactly `d` values).
    fn widen_rows(
        &self,
        info: &GgufTensorInfo,
        first_row: usize,
        n_rows: usize,
        dst: &mut Vec<f32>,
    ) -> Result<()> {
        let esz = info.dtype.type_size();
        let bytes = self.file.tensor_bytes(info);
        let start = first_row * self.d * esz;
        let end = start + n_rows * self.d * esz;
        widen_into(&bytes[start..end], info.dtype, dst, VOXTRAL_MAPPED)
    }
}

/// Per-forward block dispatch: resident slice or mapped materialization (the
/// guard is held for the whole forward so consecutive layers reuse one scratch
/// allocation). Mirrors `moshi::backbone::BlockSource`.
enum BlockSource<'a> {
    Resident(&'a [DecoderBlock]),
    Mapped(&'a MappedTextBlocks, MutexGuard<'a, DecoderBlock>),
}

impl BlockSource<'_> {
    /// The block for `layer` (materializing it on the mapped path).
    fn block(&mut self, layer: usize) -> Result<&DecoderBlock> {
        match self {
            Self::Resident(blocks) => Ok(&blocks[layer]),
            Self::Mapped(m, guard) => m.materialize_into(guard, layer),
        }
    }
}

/// All text-decoder weights. The logits head is the untied `lm_head` when
/// the checkpoint ships one, else the tied token embedding — read it through
/// [`TextDecoder::output_head`].
pub struct TextDecoder {
    /// Token embedding `[vocab_size, hidden_dim]` — also the tied LM head
    /// when `lm_head` is `None`.
    pub(crate) token_emb: Vec<f32>,
    /// Untied LM head `[vocab_size, hidden_dim]`, bound when the checkpoint
    /// carries a separate `lm_head.weight` (the shipping Voxtral mini does —
    /// byte-compared ≠ `embed_tokens` in the 2026-07-16 eval). `None` = tied
    /// (genuinely absent tensor, upstream `tie_word_embeddings` semantics).
    pub(crate) lm_head: Option<Vec<f32>>,
    /// Per-block weights — populated by [`TextDecoder::load`] (resident) and
    /// left **empty** by [`TextDecoder::load_mapped`], where `mapped` serves
    /// the blocks instead. Exactly one of the two is ever non-empty.
    pub(crate) blocks: Vec<DecoderBlock>,
    /// Bounded-memory block store. `Some` only on the mapped path; the
    /// forward dispatches on this (see [`BlockSource`]).
    pub(crate) mapped: Option<MappedTextBlocks>,
    /// Bounded-memory head store. `Some` only on the mapped path, where
    /// `token_emb` / `lm_head` are left empty and served from the mapping.
    pub(crate) mapped_heads: Option<MappedHeads>,
    /// Final RMSNorm γ (post-block, pre-head).
    pub(crate) final_norm_gamma: Vec<f32>,
    /// Which safetensors prefix the tensors were found under.
    pub(crate) prefix: &'static str,
}

impl TextDecoder {
    /// Binds every text-decoder tensor.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] with the offending tensor named on any
    /// missing / mis-shaped tensor. A **present but mis-shaped** `lm_head`
    /// is an error too — it must never silently fall back to the tied
    /// embedding (FR-EX-08).
    pub fn load(file: &GgufFile, cfg: &VoxtralConfig) -> Result<Self> {
        // The shape-only converter path leaves `n_layer == 0` — surface an
        // empty decoder to the caller (forward will still refuse to run).
        if cfg.text.n_layer == 0 || cfg.text.hidden_dim == 0 {
            return Ok(Self {
                token_emb: Vec::new(),
                lm_head: None,
                blocks: Vec::new(),
                mapped: None,
                mapped_heads: None,
                final_norm_gamma: Vec::new(),
                prefix: "",
            });
        }
        // Try both possible prefixes: modern Voxtral packaging vs. plain
        // Mistral release.
        let prefix = pick_prefix(file);
        let d = cfg.text.hidden_dim;
        let vocab = cfg.text.vocab_size;
        if vocab == 0 {
            return Err(bad("text_decoder.vocab_size must be non-zero".to_owned()));
        }

        let token_emb = tensor(file, &format!("{prefix}embed_tokens.weight"), &[vocab, d])?;

        // GQA head widths. `head_dim` comes off the config (explicit
        // metadata, or the `hidden_dim / n_head_q` legacy derivation) — NOT
        // recomputed here: the real mini has q_hidden = 32 x 128 = 4096 while
        // d = 3072, so the pre-fix `d / n_head_q` (= 96) mis-shaped every
        // attention projection.
        let n_head_q = cfg.text.n_head_q;
        let n_head_kv = cfg.text.n_head_kv;
        if n_head_q == 0 || n_head_kv == 0 {
            return Err(bad(
                "text_decoder.n_head_q and n_head_kv must be non-zero (GQA head split)".to_owned(),
            ));
        }
        let head_dim = cfg.text.head_dim();
        if head_dim == 0 {
            return Err(bad(
                "text_decoder.head_dim resolves to 0 — re-convert with a converter that writes \
                 vokra.voxtral.text_decoder.head_dim"
                    .to_owned(),
            ));
        }
        let q_hidden = n_head_q * head_dim;
        let kv_hidden = n_head_kv * head_dim;

        let mut blocks = Vec::with_capacity(cfg.text.n_layer);
        for i in 0..cfg.text.n_layer {
            let p = format!("{prefix}layers.{i}");
            let attn_norm_gamma = tensor(file, &format!("{p}.input_layernorm.weight"), &[d])?;
            let attn = GqaAttention {
                q: linear(file, &format!("{p}.self_attn.q_proj"), d, q_hidden)?,
                k: linear(file, &format!("{p}.self_attn.k_proj"), d, kv_hidden)?,
                v: linear(file, &format!("{p}.self_attn.v_proj"), d, kv_hidden)?,
                o: linear(file, &format!("{p}.self_attn.o_proj"), q_hidden, d)?,
            };
            let ffn_norm_gamma =
                tensor(file, &format!("{p}.post_attention_layernorm.weight"), &[d])?;
            let ffn = SwiGluFfn {
                gate: linear(file, &format!("{p}.mlp.gate_proj"), d, cfg.text.ffn_dim)?,
                up: linear(file, &format!("{p}.mlp.up_proj"), d, cfg.text.ffn_dim)?,
                down: linear(file, &format!("{p}.mlp.down_proj"), cfg.text.ffn_dim, d)?,
            };
            blocks.push(DecoderBlock {
                attn_norm_gamma,
                attn,
                ffn_norm_gamma,
                ffn,
            });
        }
        let final_norm_gamma = tensor(file, &format!("{prefix}norm.weight"), &[d])?;

        // Untied LM head. The tensor lives OUTSIDE the decoder submodule —
        // `language_model.lm_head.weight` for the Voxtral packaging,
        // `lm_head.weight` for a plain Mistral release. Present → bind (a
        // wrong shape is a hard error inside `tensor`, never a silent tie);
        // genuinely absent → tied token embedding.
        let lm_head_name = format!("{}lm_head.weight", lm_head_base(prefix));
        let lm_head = if file.tensor_info(&lm_head_name).is_some() {
            Some(tensor(file, &lm_head_name, &[vocab, d])?)
        } else {
            None
        };

        Ok(Self {
            token_emb,
            lm_head,
            blocks,
            mapped: None,
            mapped_heads: None,
            final_norm_gamma,
            prefix: prefix_label(prefix),
        })
    }

    /// Binds the decoder in **bounded memory**: head tensors resident, the
    /// per-layer blocks left in the GGUF mapping and widened one at a time
    /// during the forward ([`MappedTextBlocks`]).
    ///
    /// Use this when the checkpoint's `language_model` group is too large to
    /// materialise as owned f32 — on the shipping `Voxtral-Mini-3B` that group
    /// is 7.48 GiB stored but **14.95 GiB widened**, which [`Self::load`]
    /// cannot bind on a 16 GiB host. Values are bit-identical to
    /// [`Self::load`]; only the residency differs.
    ///
    /// Requires an `Arc<GgufFile>` because the mapping must outlive the
    /// decoder — pass one from `vokra_mmap::open_gguf`. Handing it a buffered
    /// [`GgufFile::open`] file works but defeats the purpose (the whole file
    /// is already resident).
    ///
    /// # Errors
    ///
    /// As [`Self::load`] for the head tensors, plus [`MappedTextBlocks::bind`]
    /// for the per-layer validation (a quantized GGUF is refused here and must
    /// use [`Self::load`]).
    pub fn load_mapped(file: Arc<GgufFile>, cfg: &VoxtralConfig) -> Result<Self> {
        if cfg.text.n_layer == 0 || cfg.text.hidden_dim == 0 {
            // Same shape-only sentinel as `load` — nothing to map.
            return Self::load(&file, cfg);
        }
        let prefix = pick_prefix(&file);
        let d = cfg.text.hidden_dim;
        let vocab = cfg.text.vocab_size;
        if vocab == 0 {
            return Err(bad("text_decoder.vocab_size must be non-zero".to_owned()));
        }
        // Heads stay in the mapping too: at the real shape `token_emb` and the
        // untied `lm_head` are ~1.5 GiB each once widened — more than the
        // mapped blocks leave behind — and neither is consumed in a way that
        // needs the whole matrix live (see `MappedHeads`).
        let final_norm_gamma = tensor(&file, &format!("{prefix}norm.weight"), &[d])?;
        let mapped_heads = MappedHeads::bind(Arc::clone(&file), prefix, vocab, d)?;
        let mapped = MappedTextBlocks::bind(Arc::clone(&file), cfg, prefix)?;
        if mapped.n_layer() != cfg.text.n_layer {
            return Err(bad(format!(
                "mapped store bound {} layers, config says {}",
                mapped.n_layer(),
                cfg.text.n_layer
            )));
        }
        Ok(Self {
            // Empty on purpose: `mapped` / `mapped_heads` are the sources. A
            // populated resident field alongside them would be a second source
            // of truth for the same weights.
            token_emb: Vec::new(),
            lm_head: None,
            blocks: Vec::new(),
            mapped: Some(mapped),
            mapped_heads: Some(mapped_heads),
            final_norm_gamma,
            prefix: prefix_label(prefix),
        })
    }

    /// The prefix the tensors were found under. Useful for diagnostics /
    /// validation from external test harnesses.
    #[must_use]
    pub fn source_prefix(&self) -> &'static str {
        self.prefix
    }

    /// Number of decoder layers, whichever residency backs them (the
    /// resident `blocks` store or the mapped one — exactly one is populated).
    #[must_use]
    pub fn n_layer(&self) -> usize {
        match &self.mapped {
            Some(m) => m.n_layer(),
            None => self.blocks.len(),
        }
    }

    /// Whether the checkpoint shipped a separate (untied) `lm_head.weight`.
    #[must_use]
    pub fn has_untied_lm_head(&self) -> bool {
        self.lm_head.is_some()
    }

    /// The logits projection `[vocab_size, hidden_dim]` — the untied
    /// `lm_head` when present, else the tied token embedding.
    pub(crate) fn output_head(&self) -> &[f32] {
        self.lm_head.as_deref().unwrap_or(&self.token_emb)
    }

    /// Writes token `tok`'s embedding row (`d` values) into `dst`.
    ///
    /// Residency-agnostic: a resident decoder copies the row out of the owned
    /// matrix, a mapped one widens it straight from the mapping. Identical
    /// values either way.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] if the row is out of range for the bound
    /// vocabulary.
    pub(crate) fn embed_row_into(&self, tok: usize, d: usize, dst: &mut Vec<f32>) -> Result<()> {
        match &self.mapped_heads {
            None => {
                let end = tok * d + d;
                if end > self.token_emb.len() {
                    return Err(bad(format!(
                        "token id {tok} out of range for token_emb of {} rows",
                        self.token_emb.len() / d.max(1)
                    )));
                }
                dst.clear();
                dst.extend_from_slice(&self.token_emb[tok * d..end]);
                Ok(())
            }
            Some(h) => {
                if tok >= h.vocab {
                    return Err(bad(format!(
                        "token id {tok} out of range for token_emb of {} rows",
                        h.vocab
                    )));
                }
                let info = h.token_emb.clone();
                h.widen_rows(&info, tok, 1, dst)
            }
        }
    }

    /// Computes one logits row: `out[v] = Σ_c head[v, c] * x[c]`.
    ///
    /// The mapped path walks the head in [`HEAD_CHUNK_ROWS`] chunks, widening
    /// each chunk just before its GEMV. Rows carry no cross-row term and each
    /// row's accumulation order is unchanged, so the result is **bit-identical**
    /// to the resident single-call GEMV (pinned by
    /// `mapped_heads_match_resident_bitwise`).
    ///
    /// # Errors
    ///
    /// Propagates the compute seam's errors, plus widening errors on the
    /// mapped path.
    pub(crate) fn logits_into(
        &self,
        compute: &Compute,
        vocab: usize,
        d: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let Some(h) = &self.mapped_heads else {
            return compute.gemv_f32(vocab, d, self.output_head(), x, None, out);
        };
        let info = h.head_info().clone();
        let mut chunk = lock_scratch(&h.chunk, VOXTRAL_MAPPED)?;
        let mut row0 = 0usize;
        while row0 < vocab {
            let rows = HEAD_CHUNK_ROWS.min(vocab - row0);
            h.widen_rows(&info, row0, rows, &mut chunk)?;
            compute.gemv_f32(rows, d, &chunk, x, None, &mut out[row0..row0 + rows])?;
            row0 += rows;
        }
        Ok(())
    }

    /// Number of vocabulary rows the head stores, whichever residency backs it.
    pub(crate) fn head_rows(&self, d: usize) -> usize {
        match &self.mapped_heads {
            Some(h) => h.vocab,
            None => self.output_head().len() / d.max(1),
        }
    }

    /// Number of token-embedding rows, whichever residency backs it.
    pub(crate) fn token_emb_rows(&self, d: usize) -> usize {
        match &self.mapped_heads {
            Some(h) => h.vocab,
            None => self.token_emb.len() / d.max(1),
        }
    }
}

/// The name prefix the (untied) LM head lives under. It sits **outside** the
/// decoder submodule: stripping a trailing `model.` **segment** maps
/// `language_model.model.` → `language_model.` and `model.` → `` (root);
/// `language_model.` (no submodule) is kept as-is — a naive
/// `strip_suffix("model.")` would corrupt it to `language_`.
fn lm_head_base(prefix: &str) -> &str {
    prefix
        .strip_suffix("model.")
        .filter(|base| base.is_empty() || base.ends_with('.'))
        .unwrap_or(prefix)
}

/// A single-step decoder state placeholder — the shape a future full
/// autoregressive forward will attach KV caches to (M3-03 paged KV cache).
///
/// Foundation-only: currently just carries the current sequence length.
pub struct TextDecoderStep {
    /// Number of tokens processed so far.
    pub seq_len: usize,
}

impl TextDecoderStep {
    /// Fresh state (nothing decoded).
    #[must_use]
    pub fn new() -> Self {
        Self { seq_len: 0 }
    }

    /// Advance one token (increment `seq_len`).
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }
}

impl Default for TextDecoderStep {
    fn default() -> Self {
        Self::new()
    }
}

// ---------- primitives ------------------------------------------------------

/// RMSNorm applied row-wise: `out[i, c] = x[i, c] * gamma[c] / sqrt(mean(x^2) + eps)`.
pub fn rms_norm(x: &[f32], gamma: &[f32], eps: f32, rows: usize, out: &mut [f32]) -> Result<()> {
    let d = gamma.len();
    if x.len() != rows * d || out.len() != rows * d {
        return Err(VokraError::InvalidArgument(format!(
            "rms_norm: x/out len must be rows*d ({}*{}={}), got x={}, out={}",
            rows,
            d,
            rows * d,
            x.len(),
            out.len(),
        )));
    }
    for i in 0..rows {
        let row = &x[i * d..(i + 1) * d];
        let sum_sq: f32 = row.iter().map(|&v| v * v).sum();
        let inv = 1.0 / (sum_sq / d as f32 + eps).sqrt();
        let dst = &mut out[i * d..(i + 1) * d];
        for c in 0..d {
            dst[c] = row[c] * inv * gamma[c];
        }
    }
    Ok(())
}

/// In-place SiLU: `x <- x * sigmoid(x)`.
pub fn silu_inplace(x: &mut [f32]) {
    for v in x {
        let s = 1.0 / (1.0 + (-*v).exp());
        *v *= s;
    }
}

/// Element-wise multiply: `a[i] <- a[i] * b[i]`. Length mismatch is a
/// programming error, so we surface it as an error rather than truncating.
pub fn hadamard_inplace(a: &mut [f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(VokraError::InvalidArgument(format!(
            "hadamard_inplace: length mismatch {} != {}",
            a.len(),
            b.len()
        )));
    }
    for (dst, &src) in a.iter_mut().zip(b) {
        *dst *= src;
    }
    Ok(())
}

/// Applies RoPE to one head's `q` / `k` slice in place.
///
/// `x` is `[seq_len, head_dim]` row-major; `head_dim` MUST be even. The
/// rotation frequencies are computed from `rope_base` per the standard
/// formula: `theta_j = rope_base ^ (-2j / head_dim)` for `j = 0..head_dim/2`.
///
/// `position_offset` supports incremental decoding: pass the absolute
/// starting position of `x[0]`. RoPE at row `i` uses frequency `theta_j`
/// scaled by `position_offset + i`.
pub fn rope_apply(
    x: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    rope_base: f32,
    position_offset: usize,
) -> Result<()> {
    if head_dim % 2 != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "rope_apply: head_dim ({head_dim}) must be even"
        )));
    }
    if x.len() != seq_len * head_dim {
        return Err(VokraError::InvalidArgument(format!(
            "rope_apply: x len {} != seq_len*head_dim {}",
            x.len(),
            seq_len * head_dim
        )));
    }
    let half = head_dim / 2;
    for i in 0..seq_len {
        let m = (position_offset + i) as f32;
        let row = &mut x[i * head_dim..(i + 1) * head_dim];
        for j in 0..half {
            let theta = rope_base.powf(-2.0 * (j as f32) / (head_dim as f32));
            let angle = m * theta;
            let (s, c) = angle.sin_cos();
            let a = row[j];
            let b = row[j + half];
            row[j] = a * c - b * s;
            row[j + half] = a * s + b * c;
        }
    }
    Ok(())
}

// ---------- autoregressive block forward -----------------------------------

/// Scratch buffers for [`forward_step`] — reused across steps by the caller
/// (see [`crate::voxtral::TextDecoderSession`]). Sized to `[max_t_q * d]` at
/// construction; steps up to `max_t_q` reuse without reallocating.
pub(crate) struct StepScratch {
    /// Residual hidden state `[t, d]`.
    pub(crate) h: Vec<f32>,
    /// Pre-norm buffer `[t, d]`.
    pub(crate) norm: Vec<f32>,
    /// Query projection `[t, q_hidden]` (`q_hidden = n_head_q × head_dim`,
    /// equal to `d` only on head_dim-tied checkpoints).
    pub(crate) q_proj: Vec<f32>,
    /// Key projection `[t, kv_hidden]` (n_head_kv × head_dim).
    pub(crate) k_proj: Vec<f32>,
    /// Value projection `[t, kv_hidden]`.
    pub(crate) v_proj: Vec<f32>,
    /// One-head Q slice buffer for RoPE `[t, head_dim]`.
    pub(crate) rope_scratch: Vec<f32>,
    /// Attention scores per head `[t, t_kv]`.
    pub(crate) scores: Vec<f32>,
    /// Softmax'd probs per head `[t, t_kv]`.
    pub(crate) probs: Vec<f32>,
    /// Attention output per head `[t, head_dim]`.
    pub(crate) head_out: Vec<f32>,
    /// Concatenated multi-head attention output `[t, q_hidden]`.
    pub(crate) attn_out: Vec<f32>,
    /// Post-`o_proj` output `[t, d]`.
    pub(crate) attn_o: Vec<f32>,
    /// FFN gate `[t, ffn_dim]`.
    pub(crate) ffn_gate: Vec<f32>,
    /// FFN up `[t, ffn_dim]`.
    pub(crate) ffn_up: Vec<f32>,
    /// FFN down `[t, d]`.
    pub(crate) ffn_down: Vec<f32>,
    /// Logits per step `[t, vocab_size]`.
    pub(crate) logits: Vec<f32>,
    /// One token-embedding row, reused across steps (the mapped head path
    /// widens into it rather than borrowing an owned matrix).
    pub(crate) embed_row: Vec<f32>,
}

impl StepScratch {
    // The reserve dims mirror the forward's buffer set (residual d, GQA
    // q/kv widths, RoPE head width, FFN inner, vocab, KV window) — an
    // intrinsic parameter bundle, same posture as `forward_step`'s allow.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_reserve(
        max_t_q: usize,
        d: usize,
        q_hidden: usize,
        kv_hidden: usize,
        head_dim: usize,
        ffn_dim: usize,
        vocab_size: usize,
        max_t_kv: usize,
    ) -> Self {
        Self {
            h: Vec::with_capacity(max_t_q * d),
            norm: Vec::with_capacity(max_t_q * d),
            q_proj: Vec::with_capacity(max_t_q * q_hidden),
            k_proj: Vec::with_capacity(max_t_q * kv_hidden),
            v_proj: Vec::with_capacity(max_t_q * kv_hidden),
            rope_scratch: Vec::with_capacity(max_t_q * head_dim),
            scores: Vec::with_capacity(max_t_q * max_t_kv),
            probs: Vec::with_capacity(max_t_q * max_t_kv),
            head_out: Vec::with_capacity(max_t_q * head_dim),
            attn_out: Vec::with_capacity(max_t_q * q_hidden),
            attn_o: Vec::with_capacity(max_t_q * d),
            ffn_gate: Vec::with_capacity(max_t_q * ffn_dim),
            ffn_up: Vec::with_capacity(max_t_q * ffn_dim),
            ffn_down: Vec::with_capacity(max_t_q * d),
            logits: Vec::with_capacity(max_t_q * vocab_size),
            embed_row: Vec::with_capacity(d),
        }
    }
}

fn resize_zero(v: &mut Vec<f32>, len: usize) {
    v.clear();
    v.resize(len, 0.0);
}

/// Runs one decoder step: forwards `tokens` through every block with the
/// caller-owned `kv_cache`, appending each block's K/V rows and leaving the
/// `[t, vocab_size]` logits in `scratch.logits`.
///
/// `position_offset` is the absolute position of `tokens[0]` in the full
/// decode (0 on the first call, then `cache.positions()` before each
/// subsequent call). RoPE uses this offset; the causal mask uses
/// `t_kv = position_offset + t` (past cache rows count) so a step past the
/// first sees prior positions correctly.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] on token out-of-range, decoder not
///   initialised, or `position_offset + t > config.text.n_ctx`.
pub(crate) fn forward_step(
    compute: &Compute,
    cfg: &VoxtralConfig,
    decoder: &TextDecoder,
    scratch: &mut StepScratch,
    kv_cache: &mut KvCache,
    tokens: &[u32],
    position_offset: usize,
) -> Result<()> {
    let d = cfg.text.hidden_dim;
    let vocab = cfg.text.vocab_size;
    let n_head_q = cfg.text.n_head_q;
    let n_head_kv = cfg.text.n_head_kv;
    let n_layer = cfg.text.n_layer;
    let t = tokens.len();

    if t == 0 {
        return Ok(());
    }
    if d == 0 || vocab == 0 || n_head_q == 0 || n_head_kv == 0 || n_layer == 0 {
        return Err(VokraError::ModelLoad(
            "voxtral text_decoder.forward_step: config carries 0-sentinel — re-convert with a \
             full VoxtralConfig (FR-EX-08 — no silent default)."
                .into(),
        ));
    }
    if position_offset + t > cfg.text.n_ctx {
        return Err(VokraError::InvalidArgument(format!(
            "voxtral text_decoder.forward_step: position {} > n_ctx {}",
            position_offset + t,
            cfg.text.n_ctx
        )));
    }
    // Residency-agnostic: `n_layer()` reads whichever store is populated
    // (resident `blocks` or the mapped one), so a mapped decoder is not
    // mistaken for an unloaded one.
    if decoder.n_layer() != n_layer {
        return Err(VokraError::ModelLoad(format!(
            "voxtral text_decoder.forward_step: loaded blocks {} != config n_layer {n_layer}",
            decoder.n_layer()
        )));
    }

    // Token embedding lookup into scratch.h.
    resize_zero(&mut scratch.h, t * d);
    for (i, &tok) in tokens.iter().enumerate() {
        let tok = tok as usize;
        if tok >= vocab {
            return Err(VokraError::InvalidArgument(format!(
                "voxtral text_decoder.forward_step: token id {tok} >= vocab {vocab}"
            )));
        }
        // Residency-agnostic row fetch: resident copies out of the owned
        // matrix, mapped widens the single row straight from the mapping.
        decoder.embed_row_into(tok, d, &mut scratch.embed_row)?;
        let dst = &mut scratch.h[i * d..i * d + d];
        dst.copy_from_slice(&scratch.embed_row);
    }

    forward_step_body(compute, cfg, decoder, scratch, kv_cache, t, position_offset)
}

/// Runs one decoder step where the hidden state is a caller-supplied raw
/// **embedding** rather than a token id sequence. Used by the audio-conditioned
/// ASR path (M3-10 Wave 8): the audio adapter's output is a `[t_prefix, d]`
/// soft-prefix embedding that must go straight into the decoder residual
/// stream, bypassing the token-embedding table lookup.
///
/// `prefix_embed` must have length `t_prefix * cfg.text.hidden_dim`;
/// `position_offset` is the absolute position of `prefix_embed[0]` (typically
/// `0` on the first call). The block forward is identical to
/// [`forward_step`] — RoPE, GQA self-attention with causal mask, KV cache
/// append, SwiGLU FFN, final RMSNorm and tied-logits head — the only
/// difference is where the initial hidden state comes from.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] on shape mismatch or
///   `position_offset + t_prefix > cfg.text.n_ctx`.
/// - [`VokraError::ModelLoad`] on a `0`-sentinel config.
// The 8 args mirror `forward_step` (compute + cfg + decoder + scratch +
// kv_cache) plus the caller-owned embedding buffer + its length + the
// starting position — an intrinsic parameter set for a forward step, same
// bundle the token-id sibling carries.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_step_with_embed_prefix(
    compute: &Compute,
    cfg: &VoxtralConfig,
    decoder: &TextDecoder,
    scratch: &mut StepScratch,
    kv_cache: &mut KvCache,
    prefix_embed: &[f32],
    t_prefix: usize,
    position_offset: usize,
) -> Result<()> {
    let d = cfg.text.hidden_dim;
    let vocab = cfg.text.vocab_size;
    let n_head_q = cfg.text.n_head_q;
    let n_head_kv = cfg.text.n_head_kv;
    let n_layer = cfg.text.n_layer;

    if t_prefix == 0 {
        return Ok(());
    }
    if d == 0 || vocab == 0 || n_head_q == 0 || n_head_kv == 0 || n_layer == 0 {
        return Err(VokraError::ModelLoad(
            "voxtral text_decoder.forward_step_with_embed_prefix: config carries 0-sentinel — \
             re-convert with a full VoxtralConfig (FR-EX-08 — no silent default)."
                .into(),
        ));
    }
    if prefix_embed.len() != t_prefix * d {
        return Err(VokraError::InvalidArgument(format!(
            "voxtral text_decoder.forward_step_with_embed_prefix: prefix_embed len {} != \
             t_prefix*hidden_dim ({}*{}={})",
            prefix_embed.len(),
            t_prefix,
            d,
            t_prefix * d
        )));
    }
    if position_offset + t_prefix > cfg.text.n_ctx {
        return Err(VokraError::InvalidArgument(format!(
            "voxtral text_decoder.forward_step_with_embed_prefix: position {} > n_ctx {}",
            position_offset + t_prefix,
            cfg.text.n_ctx
        )));
    }
    // Residency-agnostic: `n_layer()` reads whichever store is populated
    // (resident `blocks` or the mapped one), so a mapped decoder is not
    // mistaken for an unloaded one.
    if decoder.n_layer() != n_layer {
        return Err(VokraError::ModelLoad(format!(
            "voxtral text_decoder.forward_step_with_embed_prefix: loaded blocks {} != config n_layer {n_layer}",
            decoder.n_layer()
        )));
    }
    // Prime the hidden state from the caller-supplied embeddings.
    resize_zero(&mut scratch.h, t_prefix * d);
    scratch.h.copy_from_slice(prefix_embed);

    forward_step_body(
        compute,
        cfg,
        decoder,
        scratch,
        kv_cache,
        t_prefix,
        position_offset,
    )
}

/// Runs the per-block loop, final RMSNorm and tied-logits head assuming
/// `scratch.h[..t*d]` already holds the initial residual state. This is the
/// shared body of [`forward_step`] (token-id entry) and
/// [`forward_step_with_embed_prefix`] (soft-prefix entry).
fn forward_step_body(
    compute: &Compute,
    cfg: &VoxtralConfig,
    decoder: &TextDecoder,
    scratch: &mut StepScratch,
    kv_cache: &mut KvCache,
    t: usize,
    position_offset: usize,
) -> Result<()> {
    let d = cfg.text.hidden_dim;
    let ffn_dim = cfg.text.ffn_dim;
    let vocab = cfg.text.vocab_size;
    let n_head_q = cfg.text.n_head_q;
    let n_head_kv = cfg.text.n_head_kv;
    let rope_base = cfg.text.rope_base;
    let eps = cfg.text.rms_norm_eps;
    // Explicit-or-derived per-head width (see `TextDecoderConfig::head_dim`).
    // NOT `d / n_head_q`: the real mini decouples the two (q_hidden 4096 vs
    // d 3072), which is exactly what the pre-fix derivation broke on.
    let head_dim = cfg.text.head_dim();
    if head_dim == 0 {
        return Err(VokraError::ModelLoad(
            "voxtral text_decoder.forward_step: head_dim resolves to 0 — re-convert with a \
             converter that writes vokra.voxtral.text_decoder.head_dim (FR-EX-08 — no silent \
             default)."
                .into(),
        ));
    }
    let q_hidden = n_head_q * head_dim;
    let kv_hidden = n_head_kv * head_dim;
    let n_kv_groups = n_head_q / n_head_kv;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    if n_head_q % n_head_kv != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "voxtral text_decoder.forward_step: n_head_q ({n_head_q}) must be divisible by n_head_kv ({n_head_kv}) — GQA"
        )));
    }
    // Row counts, not slice lengths: a mapped decoder keeps the resident
    // vectors empty on purpose, so measuring `.len()` would reject it.
    if decoder.token_emb_rows(d) != vocab {
        return Err(VokraError::ModelLoad(format!(
            "voxtral text_decoder.forward_step: token_emb has {} rows != vocab {vocab}",
            decoder.token_emb_rows(d)
        )));
    }
    if decoder.head_rows(d) != vocab {
        return Err(VokraError::ModelLoad(format!(
            "voxtral text_decoder.forward_step: lm_head has {} rows != vocab {vocab}",
            decoder.head_rows(d)
        )));
    }

    // Pre-size mutable scratch (avoid per-block reallocation).
    resize_zero(&mut scratch.norm, t * d);
    resize_zero(&mut scratch.q_proj, t * q_hidden);
    resize_zero(&mut scratch.k_proj, t * kv_hidden);
    resize_zero(&mut scratch.v_proj, t * kv_hidden);
    resize_zero(&mut scratch.rope_scratch, t * head_dim);
    resize_zero(&mut scratch.attn_out, t * q_hidden);
    resize_zero(&mut scratch.attn_o, t * d);
    resize_zero(&mut scratch.ffn_gate, t * ffn_dim);
    resize_zero(&mut scratch.ffn_up, t * ffn_dim);
    resize_zero(&mut scratch.ffn_down, t * d);

    // Resident blocks iterate the owned store; mapped blocks widen one layer
    // at a time into the decoder's shared scratch (guard held for the whole
    // forward — `BlockSource` docs). Values are bit-identical between the two
    // sources (`MappedTextBlocks` docs).
    let mut blocks = match &decoder.mapped {
        Some(m) => BlockSource::Mapped(m, m.lock_scratch()?),
        None => BlockSource::Resident(&decoder.blocks),
    };
    for layer_idx in 0..decoder.n_layer() {
        let block = blocks.block(layer_idx)?;
        // ---------- Pre-norm self-attention ----------
        rms_norm(
            &scratch.h,
            &block.attn_norm_gamma,
            eps,
            t,
            &mut scratch.norm,
        )?;

        // Q = norm @ q.w_t: [t, d] × [d, q_hidden] → [t, q_hidden]
        compute.gemm_f32(
            t,
            q_hidden,
            d,
            &scratch.norm,
            &block.attn.q.w_t,
            None,
            &mut scratch.q_proj,
        )?;
        // K = norm @ k.w_t: [t, d] × [d, kv_hidden] → [t, kv_hidden]
        compute.gemm_f32(
            t,
            kv_hidden,
            d,
            &scratch.norm,
            &block.attn.k.w_t,
            None,
            &mut scratch.k_proj,
        )?;
        // V = norm @ v.w_t: [t, d] × [d, kv_hidden] → [t, kv_hidden]
        compute.gemm_f32(
            t,
            kv_hidden,
            d,
            &scratch.norm,
            &block.attn.v.w_t,
            None,
            &mut scratch.v_proj,
        )?;

        // Apply RoPE per-head to Q and K.
        for h in 0..n_head_q {
            // Extract head h's Q slice into rope_scratch, apply RoPE, write back.
            for i in 0..t {
                let src =
                    &scratch.q_proj[i * q_hidden + h * head_dim..i * q_hidden + (h + 1) * head_dim];
                scratch.rope_scratch[i * head_dim..(i + 1) * head_dim].copy_from_slice(src);
            }
            rope_apply(
                &mut scratch.rope_scratch[..t * head_dim],
                t,
                head_dim,
                rope_base,
                position_offset,
            )?;
            for i in 0..t {
                let dst = &mut scratch.q_proj
                    [i * q_hidden + h * head_dim..i * q_hidden + (h + 1) * head_dim];
                dst.copy_from_slice(&scratch.rope_scratch[i * head_dim..(i + 1) * head_dim]);
            }
        }
        for h in 0..n_head_kv {
            for i in 0..t {
                let src = &scratch.k_proj
                    [i * kv_hidden + h * head_dim..i * kv_hidden + (h + 1) * head_dim];
                scratch.rope_scratch[i * head_dim..(i + 1) * head_dim].copy_from_slice(src);
            }
            rope_apply(
                &mut scratch.rope_scratch[..t * head_dim],
                t,
                head_dim,
                rope_base,
                position_offset,
            )?;
            for i in 0..t {
                let dst = &mut scratch.k_proj
                    [i * kv_hidden + h * head_dim..i * kv_hidden + (h + 1) * head_dim];
                dst.copy_from_slice(&scratch.rope_scratch[i * head_dim..(i + 1) * head_dim]);
            }
        }

        // Append K/V to cache.
        kv_cache.append(
            layer_idx,
            &scratch.k_proj[..t * kv_hidden],
            &scratch.v_proj[..t * kv_hidden],
        );
        let t_kv = position_offset + t;
        let k_cache = kv_cache.k(layer_idx);
        let v_cache = kv_cache.v(layer_idx);
        // K/V cache rows for layer_idx: [t_kv, kv_hidden].

        // Attention: for each Q head h_q, use K/V head h_kv = h_q / n_kv_groups.
        // scores[t, t_kv] = Q_h @ K_h.T * scale
        // apply causal mask (row i can attend up to position_offset + i)
        // probs = softmax(scores)
        // attn_head[t, head_dim] = probs @ V_h
        // scatter head output into scratch.attn_out [t, d].
        resize_zero(&mut scratch.scores, t * t_kv);
        resize_zero(&mut scratch.probs, t * t_kv);
        resize_zero(&mut scratch.head_out, t * head_dim);
        for h_q in 0..n_head_q {
            let h_kv = h_q / n_kv_groups;
            // scores[i, j] = Σ_c Q[i, h_q*head_dim + c] * K[j, h_kv*head_dim + c] * scale
            for i in 0..t {
                let q_row = &scratch.q_proj
                    [i * q_hidden + h_q * head_dim..i * q_hidden + (h_q + 1) * head_dim];
                let row_start = i * t_kv;
                for j in 0..t_kv {
                    let k_row = &k_cache
                        [j * kv_hidden + h_kv * head_dim..j * kv_hidden + (h_kv + 1) * head_dim];
                    let mut s = 0.0f32;
                    for c in 0..head_dim {
                        s += q_row[c] * k_row[c];
                    }
                    scratch.scores[row_start + j] = s * scale;
                }
                // Causal mask: row i's absolute position is position_offset + i,
                // so keys at j > position_offset + i are masked out.
                let cur_pos = position_offset + i;
                for j in (cur_pos + 1)..t_kv {
                    scratch.scores[row_start + j] = f32::NEG_INFINITY;
                }
            }
            // Row-wise softmax.
            compute.softmax_f32(&scratch.scores, &mut scratch.probs, t, t_kv)?;
            // head_out[i, c] = Σ_j probs[i, j] * V[j, h_kv*head_dim + c]
            for i in 0..t {
                let dst = &mut scratch.head_out[i * head_dim..(i + 1) * head_dim];
                for c in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..t_kv {
                        let v_row = &v_cache[j * kv_hidden + h_kv * head_dim
                            ..j * kv_hidden + (h_kv + 1) * head_dim];
                        sum += scratch.probs[i * t_kv + j] * v_row[c];
                    }
                    dst[c] = sum;
                }
                // Scatter into scratch.attn_out at the h_q head slot.
                let out_dst = &mut scratch.attn_out
                    [i * q_hidden + h_q * head_dim..i * q_hidden + (h_q + 1) * head_dim];
                out_dst.copy_from_slice(dst);
            }
        }

        // O projection: attn_out @ o.w_t: [t, q_hidden] × [q_hidden, d] → [t, d]
        compute.gemm_f32(
            t,
            d,
            q_hidden,
            &scratch.attn_out,
            &block.attn.o.w_t,
            None,
            &mut scratch.attn_o,
        )?;

        // Residual add.
        for i in 0..t * d {
            scratch.h[i] += scratch.attn_o[i];
        }

        // ---------- Pre-norm SwiGLU FFN ----------
        rms_norm(&scratch.h, &block.ffn_norm_gamma, eps, t, &mut scratch.norm)?;
        // gate = norm @ gate.w_t → [t, ffn_dim]
        compute.gemm_f32(
            t,
            ffn_dim,
            d,
            &scratch.norm,
            &block.ffn.gate.w_t,
            None,
            &mut scratch.ffn_gate,
        )?;
        // up = norm @ up.w_t → [t, ffn_dim]
        compute.gemm_f32(
            t,
            ffn_dim,
            d,
            &scratch.norm,
            &block.ffn.up.w_t,
            None,
            &mut scratch.ffn_up,
        )?;
        // silu(gate) * up
        silu_inplace(&mut scratch.ffn_gate);
        hadamard_inplace(&mut scratch.ffn_gate, &scratch.ffn_up)?;
        // down = (silu(gate) * up) @ down.w_t → [t, d]
        compute.gemm_f32(
            t,
            d,
            ffn_dim,
            &scratch.ffn_gate,
            &block.ffn.down.w_t,
            None,
            &mut scratch.ffn_down,
        )?;
        // Residual add.
        for i in 0..t * d {
            scratch.h[i] += scratch.ffn_down[i];
        }
    }
    // Advance the position clock once (after all layer appends).
    kv_cache.advance(t);

    // Final RMSNorm.
    rms_norm(
        &scratch.h,
        &decoder.final_norm_gamma,
        eps,
        t,
        &mut scratch.norm,
    )?;

    // Logits head: logits[t, vocab] = norm[t, d] × head.T[d, vocab], where
    // `head` is the untied lm_head when the checkpoint ships one, else the
    // tied token embedding (see `TextDecoder::output_head`). The matrix is
    // stored as [vocab, d] (row-major); gemm_f32 has no transpose flag, so
    // we compute row-by-row via gemv:
    //   logits[i, v] = Σ_c norm[i, c] * head[v, c]
    //   ⇒ logits_row_i = gemv(m=vocab, k=d, a=head, x=norm_row_i)
    resize_zero(&mut scratch.logits, t * vocab);
    for i in 0..t {
        let x = &scratch.norm[i * d..(i + 1) * d];
        let out = &mut scratch.logits[i * vocab..(i + 1) * vocab];
        // Residency-agnostic: resident issues one GEMV over the owned matrix,
        // mapped walks it in row chunks. Rows are independent, so the values
        // are identical either way.
        decoder.logits_into(compute, vocab, d, x, out)?;
    }
    Ok(())
}

// ---------- internals -------------------------------------------------------

fn pick_prefix(file: &GgufFile) -> &'static str {
    // Modern Voxtral packages the Mistral backbone as a submodule.
    if file
        .tensor_info("language_model.model.embed_tokens.weight")
        .is_some()
    {
        "language_model.model."
    } else if file
        .tensor_info("language_model.embed_tokens.weight")
        .is_some()
    {
        "language_model."
    } else {
        "model."
    }
}

fn prefix_label(p: &str) -> &'static str {
    match p {
        "language_model.model." => "language_model.model.",
        "language_model." => "language_model.",
        "model." => "model.",
        _ => "",
    }
}

fn bad(msg: String) -> VokraError {
    VokraError::ModelLoad(format!("voxtral text_decoder: {msg}"))
}

fn tensor(file: &GgufFile, name: &str, want: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| bad(format!("`{name}` missing from GGUF")))?;
    let got: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    if got != want {
        return Err(bad(format!("`{name}` shape {got:?} != expected {want:?}")));
    }
    file.tensor_f32(name)
        .map_err(|e| bad(format!("`{name}`: {e}")))
}

fn linear(
    file: &GgufFile,
    prefix: &str,
    in_features: usize,
    out_features: usize,
) -> Result<Linear> {
    // Mistral projections are bias-less. The stored shape is `[out, in]`
    // (safetensors convention); we transpose once so row-major GEMM reads
    // `[in, out]`.
    let w = tensor(
        file,
        &format!("{prefix}.weight"),
        &[out_features, in_features],
    )?;
    let mut w_t = vec![0.0f32; in_features * out_features];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        for (i, &v) in row.iter().enumerate() {
            w_t[i * out_features + o] = v;
        }
    }
    Ok(Linear {
        w_t,
        in_features,
        out_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a GGUF carrying a full text decoder (heads + `n_layer` blocks)
    /// with deterministic non-trivial weights, in the requested payload dtype.
    ///
    /// `dtype` drives the mapped store's two supported widen paths: BF16 is
    /// the real-checkpoint hot path (761 of 762 Voxtral tensors), F32 the
    /// other. Values are generated as BF16-representable patterns so the two
    /// dtypes describe the *same* numbers — the resident/mapped comparison is
    /// then about residency, not about dtype rounding.
    fn decoder_gguf(cfg: &VoxtralConfig, dtype: vokra_core::gguf::GgmlType) -> GgufFile {
        use vokra_core::gguf::{GgmlType, GgufBuilder};
        let d = cfg.text.hidden_dim;
        let vocab = cfg.text.vocab_size;
        let ffn = cfg.text.ffn_dim;
        let q_hidden = cfg.text.n_head_q * cfg.text.head_dim();
        let kv_hidden = cfg.text.n_head_kv * cfg.text.head_dim();

        // A deterministic value whose low 16 mantissa bits are zero, so the
        // f32 and BF16 encodings denote exactly the same number.
        let val = |seed: usize, i: usize| -> f32 {
            let k = (seed.wrapping_mul(2_654_435_761).wrapping_add(i * 40_503)) as u32;
            let bits = (k >> 16) << 16;
            let v = f32::from_bits(bits);
            if v.is_finite() {
                v.clamp(-2.0, 2.0)
            } else {
                0.25
            }
        };
        let bytes = |vals: &[f32]| -> Vec<u8> {
            match dtype {
                GgmlType::F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
                GgmlType::BF16 => vals
                    .iter()
                    .flat_map(|v| (((v.to_bits()) >> 16) as u16).to_le_bytes())
                    .collect(),
                other => panic!("fixture covers F32/BF16 only, got {other:?}"),
            }
        };
        let mut b = GgufBuilder::new();
        let put = |b: &mut GgufBuilder, name: &str, dims: Vec<u64>, seed: usize| {
            let n: usize = dims.iter().map(|&x| x as usize).product();
            let vals: Vec<f32> = (0..n).map(|i| val(seed, i)).collect();
            b.add_tensor(name, dtype, dims, bytes(&vals)).unwrap();
        };
        put(
            &mut b,
            "model.embed_tokens.weight",
            vec![vocab as u64, d as u64],
            1,
        );
        put(&mut b, "model.norm.weight", vec![d as u64], 2);
        for i in 0..cfg.text.n_layer {
            let p = format!("model.layers.{i}");
            put(
                &mut b,
                &format!("{p}.input_layernorm.weight"),
                vec![d as u64],
                10 + i,
            );
            put(
                &mut b,
                &format!("{p}.self_attn.q_proj.weight"),
                vec![q_hidden as u64, d as u64],
                20 + i,
            );
            put(
                &mut b,
                &format!("{p}.self_attn.k_proj.weight"),
                vec![kv_hidden as u64, d as u64],
                30 + i,
            );
            put(
                &mut b,
                &format!("{p}.self_attn.v_proj.weight"),
                vec![kv_hidden as u64, d as u64],
                40 + i,
            );
            put(
                &mut b,
                &format!("{p}.self_attn.o_proj.weight"),
                vec![d as u64, q_hidden as u64],
                50 + i,
            );
            put(
                &mut b,
                &format!("{p}.post_attention_layernorm.weight"),
                vec![d as u64],
                60 + i,
            );
            put(
                &mut b,
                &format!("{p}.mlp.gate_proj.weight"),
                vec![ffn as u64, d as u64],
                70 + i,
            );
            put(
                &mut b,
                &format!("{p}.mlp.up_proj.weight"),
                vec![ffn as u64, d as u64],
                80 + i,
            );
            put(
                &mut b,
                &format!("{p}.mlp.down_proj.weight"),
                vec![d as u64, ffn as u64],
                90 + i,
            );
        }
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    /// Runs a few decode steps and returns the raw f32 hidden state, so two
    /// residencies can be compared bit-for-bit rather than by tolerance.
    fn run_steps(cfg: &VoxtralConfig, dec: &TextDecoder) -> Vec<u32> {
        let compute = Compute::for_backend(vokra_core::BackendKind::Cpu, &[]).unwrap();
        let kv_hidden = cfg.text.n_head_kv * cfg.text.head_dim();
        let mut scratch = StepScratch::with_reserve(
            4,
            cfg.text.hidden_dim,
            cfg.text.n_head_q * cfg.text.head_dim(),
            kv_hidden,
            cfg.text.head_dim(),
            cfg.text.ffn_dim,
            cfg.text.vocab_size,
            8,
        );
        let mut kv = KvCache::with_reserve(cfg.text.n_layer, kv_hidden, 8);
        let mut out = Vec::new();
        for (step, tok) in [1u32, 2, 3].into_iter().enumerate() {
            forward_step(&compute, cfg, dec, &mut scratch, &mut kv, &[tok], step).unwrap();
            out.extend(scratch.logits.iter().map(|v| v.to_bits()));
        }
        out
    }

    /// **The bounded-memory contract's numerical half.** Materializing a layer
    /// at a time out of the mapping must reproduce the resident f32 forward
    /// BIT-identically — same widen formula, same `[out, in]` -> `[in, out]`
    /// index math. Anything less would make the mapped path a different model,
    /// not a cheaper loader.
    ///
    /// Both payload dtypes are covered because the mapped store widens them
    /// through different arms.
    #[test]
    fn mapped_blocks_match_resident_bitwise() {
        use vokra_core::gguf::GgmlType;
        for dtype in [GgmlType::BF16, GgmlType::F32] {
            let cfg = crate::voxtral::test_support::gqa_config();
            let file = decoder_gguf(&cfg, dtype);
            let resident = TextDecoder::load(&file, &cfg).unwrap();
            let mapped =
                TextDecoder::load_mapped(Arc::new(decoder_gguf(&cfg, dtype)), &cfg).unwrap();

            assert_eq!(
                resident.n_layer(),
                cfg.text.n_layer,
                "resident bound all layers"
            );
            assert_eq!(
                mapped.n_layer(),
                cfg.text.n_layer,
                "mapped bound all layers"
            );
            assert!(
                mapped.blocks.is_empty(),
                "mapped must not also own resident blocks"
            );
            assert!(
                resident.mapped.is_none(),
                "resident must not own a mapped store"
            );

            assert_eq!(
                run_steps(&cfg, &resident),
                run_steps(&cfg, &mapped),
                "{dtype:?}: mapped forward must be bit-identical to resident"
            );
        }
    }

    /// The head store is the other half of the bounded-memory contract: the
    /// chunked GEMV over mapped rows must reproduce the resident single-call
    /// GEMV **bit-for-bit**, and a mapped decoder must serve embedding rows
    /// identically. If either drifted, the memory saving would be bought with
    /// a different model.
    ///
    /// `gqa_config`'s vocab is deliberately larger than [`HEAD_CHUNK_ROWS`] is
    /// not — so this also pins the single-partial-chunk case; the
    /// multi-chunk case is covered by raising vocab past the chunk edge.
    #[test]
    fn mapped_heads_match_resident_bitwise() {
        use vokra_core::gguf::GgmlType;
        for dtype in [GgmlType::BF16, GgmlType::F32] {
            for vocab in [
                crate::voxtral::test_support::gqa_config().text.vocab_size,
                HEAD_CHUNK_ROWS + 7,
            ] {
                let mut cfg = crate::voxtral::test_support::gqa_config();
                cfg.text.vocab_size = vocab;
                let resident = TextDecoder::load(&decoder_gguf(&cfg, dtype), &cfg).unwrap();
                let mapped =
                    TextDecoder::load_mapped(Arc::new(decoder_gguf(&cfg, dtype)), &cfg).unwrap();

                assert!(
                    mapped.token_emb.is_empty() && mapped.lm_head.is_none(),
                    "a mapped decoder must not also own resident head matrices"
                );
                assert_eq!(mapped.head_rows(cfg.text.hidden_dim), vocab);
                assert_eq!(mapped.token_emb_rows(cfg.text.hidden_dim), vocab);

                // Embedding rows must match exactly, including the last row.
                let d = cfg.text.hidden_dim;
                for tok in [0usize, 1, vocab - 1] {
                    let (mut a, mut b) = (Vec::new(), Vec::new());
                    resident.embed_row_into(tok, d, &mut a).unwrap();
                    mapped.embed_row_into(tok, d, &mut b).unwrap();
                    assert_eq!(
                        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{dtype:?} vocab={vocab}: embed row {tok} must be bit-identical"
                    );
                }

                // And the full decode, which exercises the chunked head GEMV.
                assert_eq!(
                    run_steps(&cfg, &resident),
                    run_steps(&cfg, &mapped),
                    "{dtype:?} vocab={vocab}: mapped heads must be bit-identical"
                );
            }
        }
    }

    /// An out-of-range token is a loud error on both residencies — the mapped
    /// path must not silently read a neighbouring row out of the mapping.
    #[test]
    fn an_out_of_range_token_is_rejected_on_both_residencies() {
        use vokra_core::gguf::GgmlType;
        let cfg = crate::voxtral::test_support::gqa_config();
        let d = cfg.text.hidden_dim;
        let vocab = cfg.text.vocab_size;
        let resident = TextDecoder::load(&decoder_gguf(&cfg, GgmlType::BF16), &cfg).unwrap();
        let mapped =
            TextDecoder::load_mapped(Arc::new(decoder_gguf(&cfg, GgmlType::BF16)), &cfg).unwrap();
        let mut out = Vec::new();
        for (label, dec) in [("resident", &resident), ("mapped", &mapped)] {
            let err = dec.embed_row_into(vocab, d, &mut out).unwrap_err();
            assert!(
                format!("{err}").contains("out of range"),
                "{label}: out-of-range token must be explicit, got: {err}"
            );
        }
    }

    /// A payload dtype the mapped path cannot widen in place must be refused
    /// loudly, naming the resident constructor — never widened wrong, never a
    /// silent fallback (FR-EX-08).
    #[test]
    fn mapped_bind_refuses_unsupported_dtype_and_points_at_the_resident_loader() {
        use vokra_core::gguf::{GgmlType, GgufBuilder};
        let cfg = crate::voxtral::test_support::gqa_config();
        let d = cfg.text.hidden_dim;
        let mut b = GgufBuilder::new();
        b.add_tensor(
            "model.embed_tokens.weight",
            GgmlType::F32,
            vec![cfg.text.vocab_size as u64, d as u64],
            vec![0u8; cfg.text.vocab_size * d * 4],
        )
        .unwrap();
        // F16 is correctly sized for any element count, so this reaches the
        // dtype arm rather than tripping the element-count check first. The
        // mapped path serves F32/BF16 only — F16 has no in-place widen here.
        b.add_tensor(
            "model.layers.0.input_layernorm.weight",
            GgmlType::F16,
            vec![d as u64],
            vec![0u8; d * GgmlType::F16.type_size()],
        )
        .unwrap();
        let file = Arc::new(GgufFile::parse(b.to_bytes().unwrap()).unwrap());
        let err = MappedTextBlocks::bind(Arc::clone(&file), &cfg, "model.").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("voxtral"),
            "names the model that refused: {msg}"
        );
        assert!(
            msg.contains("TextDecoder::load"),
            "points at the resident loader so the refusal is actionable: {msg}"
        );
    }

    /// A missing layer tensor is a loud bind-time error naming the tensor —
    /// the load fails, never a forward halfway through a stream.
    #[test]
    fn mapped_bind_names_a_missing_tensor() {
        use vokra_core::gguf::GgufBuilder;
        let cfg = crate::voxtral::test_support::gqa_config();
        let b = GgufBuilder::new();
        let file = Arc::new(GgufFile::parse(b.to_bytes().unwrap()).unwrap());
        let err = MappedTextBlocks::bind(file, &cfg, "model.").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("model.layers.0.input_layernorm.weight") && msg.contains("missing"),
            "must name the first missing tensor: {msg}"
        );
    }

    #[test]
    fn rms_norm_normalises_row_to_unit_rms() {
        // With gamma = 1, RMSNorm output should have unit RMS (per row).
        let d = 4;
        let x = vec![1.0f32, 2.0, 3.0, 4.0]; // mean(x^2) = 7.5.
        let gamma = vec![1.0f32; d];
        let mut out = vec![0.0f32; d];
        rms_norm(&x, &gamma, 0.0, 1, &mut out).unwrap();
        let mean_sq: f32 = out.iter().map(|v| v * v).sum::<f32>() / d as f32;
        assert!(
            (mean_sq - 1.0).abs() < 1e-5,
            "row RMS should be 1.0, got sqrt({mean_sq})"
        );
    }

    #[test]
    fn rms_norm_zero_row_stays_zero_with_epsilon() {
        // An all-zero row must not blow up (eps guards the divisor).
        let x = vec![0.0f32; 4];
        let gamma = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rms_norm(&x, &gamma, 1e-5, 1, &mut out).unwrap();
        assert!(out.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn silu_matches_reference_at_specific_points() {
        // silu(0)=0, silu(large positive)≈x, silu(large negative)≈0.
        let mut x = vec![0.0f32, 5.0, -5.0, 1.0];
        silu_inplace(&mut x);
        assert!((x[0]).abs() < 1e-6);
        assert!((x[1] - 5.0 * (1.0 / (1.0 + (-5.0f32).exp()))).abs() < 1e-5);
        assert!(x[2].abs() < 0.05); // small negative
        // silu(1) = 1 * sigmoid(1) ≈ 0.731
        assert!((x[3] - 0.731_058_6).abs() < 1e-3);
    }

    #[test]
    fn hadamard_multiplies_elementwise() {
        let mut a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        hadamard_inplace(&mut a, &b).unwrap();
        assert_eq!(a, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn hadamard_rejects_length_mismatch() {
        let mut a = vec![1.0f32, 2.0];
        let b = vec![1.0f32];
        assert!(matches!(
            hadamard_inplace(&mut a, &b),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rope_apply_position_zero_is_identity() {
        // At m=0, all angles are 0 → cos=1, sin=0 → unchanged.
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let orig = x.clone();
        rope_apply(&mut x, 1, 4, 10_000.0, 0).unwrap();
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_apply_rotation_preserves_norm() {
        // RoPE is a rotation, so it preserves the vector norm per row.
        let mut x = vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let orig_norms: Vec<f32> = x
            .chunks(4)
            .map(|c| c.iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect();
        rope_apply(&mut x, 2, 4, 10_000.0, 3).unwrap();
        let new_norms: Vec<f32> = x
            .chunks(4)
            .map(|c| c.iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect();
        for (a, b) in orig_norms.iter().zip(new_norms.iter()) {
            assert!((a - b).abs() < 1e-4, "norm changed: {a} -> {b}");
        }
    }

    #[test]
    fn rope_apply_rejects_odd_head_dim() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        assert!(matches!(
            rope_apply(&mut x, 1, 3, 10_000.0, 0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn text_decoder_step_advances_seq_len() {
        let mut s = TextDecoderStep::new();
        assert_eq!(s.seq_len, 0);
        s.advance();
        s.advance();
        assert_eq!(s.seq_len, 2);
    }

    // ---------- extended oracle tests (M3-10 structural completion) --------

    #[test]
    fn rms_norm_scales_by_gamma_per_channel() {
        // With a non-uniform γ, each column of the output should be scaled
        // exactly by the corresponding γ[c] after the row is normalised.
        // Craft a row whose RMS is a nice number so the effect of γ is
        // isolated from the divisor.
        let d = 4;
        // row [2, 2, 2, 2] has mean(x^2)=4 → RMS=2 → x / RMS = [1, 1, 1, 1].
        let x = vec![2.0f32; d];
        // γ = [10, 20, 30, 40] → out = γ * 1.
        let gamma = vec![10.0f32, 20.0, 30.0, 40.0];
        let mut out = vec![0.0f32; d];
        rms_norm(&x, &gamma, 0.0, 1, &mut out).unwrap();
        for (i, &g) in gamma.iter().enumerate() {
            assert!(
                (out[i] - g).abs() < 1e-4,
                "column {i}: expected {g}, got {}",
                out[i]
            );
        }
    }

    #[test]
    fn rms_norm_epsilon_prevents_divide_by_zero_and_scales_predictably() {
        // Non-zero row with a large ε: the divisor becomes sqrt(mean_sq + ε).
        // For row [2,2,2,2] mean_sq=4, ε=12 → divisor=sqrt(16)=4 → out = x/4 = [0.5,…].
        let x = vec![2.0f32; 4];
        let gamma = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rms_norm(&x, &gamma, 12.0, 1, &mut out).unwrap();
        for v in &out {
            assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn rms_norm_multirow_processes_each_row_independently() {
        // Two rows with different scales must be normalised to the same RMS.
        let d = 4;
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let gamma = vec![1.0f32; d];
        let mut out = vec![0.0f32; d * 2];
        rms_norm(&x, &gamma, 0.0, 2, &mut out).unwrap();
        for row in 0..2 {
            let slice = &out[row * d..(row + 1) * d];
            let rms = (slice.iter().map(|v| v * v).sum::<f32>() / d as f32).sqrt();
            assert!(
                (rms - 1.0).abs() < 1e-4,
                "row {row}: RMS should be 1, got {rms}"
            );
        }
    }

    #[test]
    fn rms_norm_shape_mismatch_is_error_not_panic() {
        // x/out length disagreeing with rows*d must surface as an error.
        let gamma = vec![1.0f32; 4];
        let x = vec![1.0f32; 3]; // should be 4 for one row
        let mut out = vec![0.0f32; 4];
        assert!(matches!(
            rms_norm(&x, &gamma, 0.0, 1, &mut out),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn silu_derivative_positive_at_origin() {
        // SiLU(0)=0 and SiLU'(0)=0.5. This is a small numerical check that
        // silu_inplace matches the math (numerical derivative via
        // (silu(h) - silu(-h)) / 2h).
        let h = 1e-3f32;
        let mut a = vec![h];
        let mut b = vec![-h];
        silu_inplace(&mut a);
        silu_inplace(&mut b);
        let d = (a[0] - b[0]) / (2.0 * h);
        assert!((d - 0.5).abs() < 1e-2, "silu'(0) ≈ 0.5, got {d}");
    }

    #[test]
    fn silu_asymptotic_saturation() {
        // silu(large positive x) ≈ x; silu(large negative x) ≈ 0.
        let mut pos = vec![50.0f32];
        let mut neg = vec![-50.0f32];
        silu_inplace(&mut pos);
        silu_inplace(&mut neg);
        assert!((pos[0] - 50.0).abs() < 1e-3, "silu(50)≈50, got {}", pos[0]);
        assert!(neg[0].abs() < 1e-10, "silu(-50)≈0, got {}", neg[0]);
    }

    #[test]
    fn swiglu_gate_up_roundtrip_pattern() {
        // SwiGLU is `silu(gate(x)) * up(x)`. Verify the pattern element-wise
        // using pre-computed gate and up projections on a small vector.
        // For x=[1,2,3,4] with an identity gate and up: silu(x)*x should be
        // silu-elementwise times x-elementwise.
        let gate_out = vec![1.0f32, 2.0, 3.0, 4.0];
        let up_out = vec![1.0f32, 2.0, 3.0, 4.0];
        // Apply silu to a copy of gate_out.
        let mut activated = gate_out.clone();
        silu_inplace(&mut activated);
        // Hadamard with up_out.
        let mut swiglu = activated.clone();
        hadamard_inplace(&mut swiglu, &up_out).unwrap();
        // Verify each element: silu(gate[i]) * up[i].
        for (i, ((&g, &u), &s)) in gate_out.iter().zip(&up_out).zip(&swiglu).enumerate() {
            let expected = g * (1.0 / (1.0 + (-g).exp())) * u;
            assert!(
                (s - expected).abs() < 1e-4,
                "swiglu[{i}] expected {expected}, got {s}"
            );
        }
    }

    #[test]
    fn rope_apply_frequency_formula_at_first_pair() {
        // Verify the first frequency pair (j=0) rotates by angle m * θ_0 =
        // m * rope_base^(-2*0/head_dim) = m * 1 = m radians (regardless of
        // rope_base). This is a bedrock property: the θ_0 pair rotates at
        // exactly the position rate.
        let head_dim = 4;
        let rope_base = 10_000.0f32;
        let m = 5.0f32;
        // Row [1, 0, 0, 0]: the (j=0) pair is (x[0]=1, x[2]=0).
        // After RoPE at position m: x[0]=cos(m*1)*1 = cos(m), x[2]=sin(m).
        let mut x = vec![1.0f32, 0.0, 0.0, 0.0];
        rope_apply(&mut x, 1, head_dim, rope_base, m as usize).unwrap();
        assert!(
            (x[0] - m.cos()).abs() < 1e-4,
            "x[0]={}, want cos({m})",
            x[0]
        );
        assert!(
            (x[2] - m.sin()).abs() < 1e-4,
            "x[2]={}, want sin({m})",
            x[2]
        );
    }

    #[test]
    fn rope_apply_second_pair_scales_frequency_with_rope_base() {
        // For the j=1 pair, θ_1 = rope_base^(-2/head_dim).
        // With head_dim=4 and rope_base=10_000, θ_1 = 10_000^(-0.5) = 0.01.
        // Row [0, 1, 0, 0]: the (j=1) pair is (x[1]=1, x[3]=0).
        // After RoPE at position m=1: x[1]=cos(θ_1), x[3]=sin(θ_1).
        let head_dim = 4;
        let rope_base = 10_000.0f32;
        let theta_1 = rope_base.powf(-2.0 / head_dim as f32);
        let mut x = vec![0.0f32, 1.0, 0.0, 0.0];
        rope_apply(&mut x, 1, head_dim, rope_base, 1).unwrap();
        assert!(
            (x[1] - theta_1.cos()).abs() < 1e-5,
            "x[1]={}, want cos({theta_1})",
            x[1]
        );
        assert!(
            (x[3] - theta_1.sin()).abs() < 1e-5,
            "x[3]={}, want sin({theta_1})",
            x[3]
        );
    }

    #[test]
    fn rope_apply_position_offset_advances_angles_by_one_row() {
        // A single row at offset m must equal the m-th row of a run at
        // offset 0 with m+1 rows. This is the incremental-decoding
        // invariant that KV-cache-append depends on.
        let head_dim = 4;
        let rope_base = 10_000.0f32;
        let orig = [1.0f32, 2.0, 3.0, 4.0];
        // Full-range run at offset 0, 5 rows: use row 3.
        let mut full = orig.repeat(5);
        rope_apply(&mut full, 5, head_dim, rope_base, 0).unwrap();
        let row_from_full = &full[3 * head_dim..4 * head_dim];
        // Single-row run at offset 3.
        let mut single = orig.to_vec();
        rope_apply(&mut single, 1, head_dim, rope_base, 3).unwrap();
        for (i, (&a, &b)) in single.iter().zip(row_from_full.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "offset invariance broken at index {i}: single={a}, cached={b}"
            );
        }
    }

    #[test]
    fn rope_apply_length_mismatch_is_error_not_panic() {
        let mut x = vec![1.0f32, 2.0, 3.0]; // 3 elements, seq_len*head_dim=4
        assert!(matches!(
            rope_apply(&mut x, 1, 4, 10_000.0, 0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn gqa_head_split_derivation_from_config() {
        // The REAL Voxtral-Mini-3B-2507 split (config.json, 2026-07-16
        // real-weight eval): 32 query heads, 8 KV heads, hidden_dim 3072,
        // explicit head_dim 128 — so q_hidden = 4096 ≠ hidden_dim, and each
        // K/V head is broadcast to 32/8 = 4 query heads. (The pre-fix test
        // asserted a hallucinated 24-head config chosen so that hidden/24
        // happened to be 128.)
        use crate::voxtral::config::TextDecoderConfig;
        let cfg = TextDecoderConfig {
            n_layer: 30,
            n_head_q: 32,
            n_head_kv: 8,
            head_dim: 128,
            hidden_dim: 3072,
            ffn_dim: 8192,
            vocab_size: 131_072,
            n_ctx: 131_072,
            rope_base: 100_000_000.0,
            rms_norm_eps: 1e-5,
        };
        assert_eq!(cfg.head_dim(), 128, "explicit head_dim wins");
        assert_eq!(cfg.q_hidden(), 4096, "decoupled from hidden_dim (3072)");
        assert_eq!(cfg.kv_hidden(), 1024);
        assert_eq!(
            cfg.n_head_q % cfg.n_head_kv,
            0,
            "GQA requires n_head_q % n_head_kv == 0"
        );
        // The number of query heads sharing one K/V head:
        assert_eq!(cfg.n_head_q / cfg.n_head_kv, 4);
    }

    #[test]
    fn lm_head_base_strips_only_the_model_segment() {
        // `language_model.model.` → head at `language_model.lm_head.weight`;
        // plain `model.` → root `lm_head.weight`. A bare `language_model.`
        // must NOT be corrupted to `language_` (the naive strip_suffix trap).
        assert_eq!(lm_head_base("language_model.model."), "language_model.");
        assert_eq!(lm_head_base("model."), "");
        assert_eq!(lm_head_base("language_model."), "language_model.");
    }

    // ---------- GQA decoupled-head_dim load + untied lm_head (P1 fix) -------

    mod gqa_load {
        use super::*;
        use crate::voxtral::test_support::gqa_config;
        use vokra_core::gguf::{GgmlType, GgufBuilder};

        /// Deterministic non-constant f32 payload (distinct per `seed` so the
        /// untied-vs-tied logits comparison is meaningful).
        fn f32_bytes(n: usize, seed: f32) -> Vec<u8> {
            (0..n)
                .flat_map(|i| (seed + 0.01 * ((i % 5) as f32 - 2.0)).to_le_bytes())
                .collect()
        }

        /// A GQA-shaped 1-layer GGUF matching [`gqa_config`]: `d = 6`,
        /// `head_dim = 4` (decoupled — `d / n_head_q` would be 3), `q_hidden
        /// = 8`, `kv_hidden = 4`. `lm_head` controls the untied head tensor:
        /// `None` = tied checkpoint, `Some(shape)` writes that shape.
        fn gqa_gguf(lm_head: Option<&[u64]>) -> GgufFile {
            let cfg = gqa_config();
            let d = cfg.text.hidden_dim as u64;
            let q = cfg.text.q_hidden() as u64;
            let kv = cfg.text.kv_hidden() as u64;
            let ffn = cfg.text.ffn_dim as u64;
            let vocab = cfg.text.vocab_size as u64;
            let p = "language_model.model.";
            let mut b = GgufBuilder::new();
            let mut add = |name: String, dims: Vec<u64>, seed: f32| {
                let n: u64 = dims.iter().product();
                b.add_tensor(&name, GgmlType::F32, dims, f32_bytes(n as usize, seed))
                    .unwrap();
            };
            add(format!("{p}embed_tokens.weight"), vec![vocab, d], 0.05);
            add(format!("{p}layers.0.input_layernorm.weight"), vec![d], 1.0);
            add(
                format!("{p}layers.0.self_attn.q_proj.weight"),
                vec![q, d],
                0.10,
            );
            add(
                format!("{p}layers.0.self_attn.k_proj.weight"),
                vec![kv, d],
                -0.07,
            );
            add(
                format!("{p}layers.0.self_attn.v_proj.weight"),
                vec![kv, d],
                0.05,
            );
            add(
                format!("{p}layers.0.self_attn.o_proj.weight"),
                vec![d, q],
                -0.04,
            );
            add(
                format!("{p}layers.0.post_attention_layernorm.weight"),
                vec![d],
                1.0,
            );
            add(
                format!("{p}layers.0.mlp.gate_proj.weight"),
                vec![ffn, d],
                0.06,
            );
            add(
                format!("{p}layers.0.mlp.up_proj.weight"),
                vec![ffn, d],
                -0.02,
            );
            add(
                format!("{p}layers.0.mlp.down_proj.weight"),
                vec![d, ffn],
                0.03,
            );
            add(format!("{p}norm.weight"), vec![d], 1.0);
            if let Some(shape) = lm_head {
                add(
                    "language_model.lm_head.weight".to_owned(),
                    shape.to_vec(),
                    -0.09,
                );
            }
            GgufFile::parse(b.to_bytes().unwrap()).unwrap()
        }

        #[test]
        fn load_binds_decoupled_gqa_projections() {
            // The pre-fix loader derived head_dim = d / n_head_q (= 3 here)
            // and expected a square [d, d] Q — rejecting exactly this shape
            // class ("shape [4096, 3072] != expected [3072, 3072]" on the
            // real mini). The decoupled shapes must now bind.
            let cfg = gqa_config();
            let file = gqa_gguf(None);
            let td = TextDecoder::load(&file, &cfg).unwrap();
            assert_eq!(td.n_layer(), 1);
            let attn = &td.blocks[0].attn;
            assert_eq!((attn.q.in_features, attn.q.out_features), (6, 8));
            assert_eq!((attn.k.in_features, attn.k.out_features), (6, 4));
            assert_eq!((attn.v.in_features, attn.v.out_features), (6, 4));
            assert_eq!((attn.o.in_features, attn.o.out_features), (8, 6));
        }

        #[test]
        fn load_binds_untied_lm_head_when_present() {
            let cfg = gqa_config();
            let vocab = cfg.text.vocab_size as u64;
            let d = cfg.text.hidden_dim as u64;
            let file = gqa_gguf(Some(&[vocab, d]));
            let td = TextDecoder::load(&file, &cfg).unwrap();
            assert!(td.has_untied_lm_head());
            // The head really is the separate tensor, not the embedding.
            assert_ne!(td.output_head(), td.token_emb.as_slice());
        }

        #[test]
        fn load_ties_embedding_only_when_lm_head_genuinely_absent() {
            let cfg = gqa_config();
            let file = gqa_gguf(None);
            let td = TextDecoder::load(&file, &cfg).unwrap();
            assert!(!td.has_untied_lm_head());
            // Tied semantics: the output head IS the token embedding.
            assert_eq!(td.output_head(), td.token_emb.as_slice());
        }

        #[test]
        fn load_rejects_mis_shaped_lm_head_instead_of_silent_tie() {
            // A PRESENT lm_head with the wrong shape must hard-error — never
            // silently fall back to the tied embedding (FR-EX-08).
            let cfg = gqa_config();
            let vocab = cfg.text.vocab_size as u64;
            let d = cfg.text.hidden_dim as u64;
            let file = gqa_gguf(Some(&[vocab, d + 1]));
            let err = match TextDecoder::load(&file, &cfg) {
                Ok(_) => panic!("mis-shaped lm_head must not load"),
                Err(e) => e,
            };
            assert!(
                err.to_string().contains("lm_head"),
                "error must name the tensor: {err}"
            );
        }

        #[test]
        fn forward_on_decoupled_gqa_shapes_produces_finite_logits() {
            // End-to-end through the session (the KV cache width and every
            // scratch stride must use q_hidden/kv_hidden, not d).
            use crate::voxtral::TextDecoderSession;
            let cfg = gqa_config();
            let file = gqa_gguf(Some(&[
                cfg.text.vocab_size as u64,
                cfg.text.hidden_dim as u64,
            ]));
            let td = TextDecoder::load(&file, &cfg).unwrap();
            let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
            sess.step_into(&[1u32, 2, 0]).unwrap();
            let full: Vec<f32> = sess.last_logits_row().to_vec();
            assert_eq!(full.len(), cfg.text.vocab_size);
            assert!(full.iter().all(|v| v.is_finite()));

            // Incremental decode with the KV cache must agree with the
            // full-prefix step (same tolerance as the MHA sibling test).
            sess.reset();
            sess.step_into(&[1u32]).unwrap();
            sess.step_into(&[2u32]).unwrap();
            sess.step_into(&[0u32]).unwrap();
            for (i, (&f, &c)) in full.iter().zip(sess.last_logits_row()).enumerate() {
                assert!((f - c).abs() < 5e-4, "idx {i}: full {f} vs cached {c}");
            }
        }

        #[test]
        fn untied_lm_head_changes_logits_vs_tied() {
            use crate::voxtral::TextDecoderSession;
            let cfg = gqa_config();
            let vocab = cfg.text.vocab_size as u64;
            let d = cfg.text.hidden_dim as u64;

            let tied = TextDecoder::load(&gqa_gguf(None), &cfg).unwrap();
            let untied = TextDecoder::load(&gqa_gguf(Some(&[vocab, d])), &cfg).unwrap();

            let mut s_tied = TextDecoderSession::cpu(&cfg, &tied).unwrap();
            let mut s_untied = TextDecoderSession::cpu(&cfg, &untied).unwrap();
            s_tied.step_into(&[1u32, 2]).unwrap();
            s_untied.step_into(&[1u32, 2]).unwrap();
            let diff: f32 = s_tied
                .last_logits_row()
                .iter()
                .zip(s_untied.last_logits_row())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                diff > 1e-6,
                "untied lm_head (different values) must change the logits"
            );
        }
    }

    // ---------- property: GQA with n_head_kv == n_head_q ≡ MHA --------------

    mod gqa_reduction {
        use super::*;
        use crate::voxtral::TextDecoderSession;
        use crate::voxtral::config::{AudioEncoderConfig, TextDecoderConfig};

        fn cfg_with_kv_heads(n_head_kv: usize) -> VoxtralConfig {
            VoxtralConfig {
                audio: AudioEncoderConfig {
                    n_layer: 1,
                    n_head: 2,
                    hidden_dim: 6,
                    n_ctx: 4,
                    n_mels: 2,
                    ffn_dim: 8,
                },
                text: TextDecoderConfig {
                    n_layer: 1,
                    n_head_q: 2,
                    n_head_kv,
                    head_dim: 4, // decoupled: d / n_head_q would be 3
                    hidden_dim: 6,
                    ffn_dim: 8,
                    vocab_size: 8,
                    n_ctx: 16,
                    rope_base: 10_000.0,
                    rms_norm_eps: 1e-5,
                },
                cross_attn_hidden_dim: 6,
                mode: "asr".to_owned(),
                s2s_codec_type: "none".to_owned(),
            }
        }

        fn pattern_linear(rows: usize, cols: usize, base: f32) -> Linear {
            let mut w_t = vec![0.0f32; rows * cols];
            for (i, v) in w_t.iter_mut().enumerate() {
                *v = base + 0.01 * ((i as i32 % 5) - 2) as f32;
            }
            Linear {
                w_t,
                in_features: rows,
                out_features: cols,
            }
        }

        /// Duplicates a `[in, head_dim]` K/V projection into `[in, groups *
        /// head_dim]` by repeating the column block — the MHA weights whose
        /// per-head K/V equal the single GQA head.
        fn duplicate_kv(base: &Linear, groups: usize) -> Linear {
            let (rows, cols) = (base.in_features, base.out_features);
            let mut w_t = vec![0.0f32; rows * cols * groups];
            for r in 0..rows {
                for g in 0..groups {
                    for c in 0..cols {
                        w_t[r * cols * groups + g * cols + c] = base.w_t[r * cols + c];
                    }
                }
            }
            Linear {
                w_t,
                in_features: rows,
                out_features: cols * groups,
            }
        }

        fn decoder_with_kv(kv: Linear, v: Linear, d: usize, cfg: &VoxtralConfig) -> TextDecoder {
            let ffn = cfg.text.ffn_dim;
            let vocab = cfg.text.vocab_size;
            let q_hidden = cfg.text.q_hidden();
            let mut token_emb = vec![0.0f32; vocab * d];
            for (i, val) in token_emb.iter_mut().enumerate() {
                *val = ((i as i32 % 7) - 3) as f32 * 0.05;
            }
            TextDecoder {
                token_emb,
                lm_head: None,
                mapped: None,
                mapped_heads: None,
                blocks: vec![DecoderBlock {
                    attn_norm_gamma: vec![1.0f32; d],
                    attn: GqaAttention {
                        q: pattern_linear(d, q_hidden, 0.10),
                        k: kv,
                        v,
                        o: pattern_linear(q_hidden, d, -0.04),
                    },
                    ffn_norm_gamma: vec![1.0f32; d],
                    ffn: SwiGluFfn {
                        gate: pattern_linear(d, ffn, 0.06),
                        up: pattern_linear(d, ffn, -0.02),
                        down: pattern_linear(ffn, d, 0.03),
                    },
                }],
                final_norm_gamma: vec![1.0f32; d],
                prefix: "",
            }
        }

        /// GQA with a single K/V head must be **bit-identical** to plain MHA
        /// (`n_head_kv == n_head_q`) whose per-head K/V weights are copies of
        /// that single head: every dot product, softmax row and accumulation
        /// runs in the same order over the same values, so the grouped
        /// broadcast is exactly the head-duplication it claims to be.
        #[test]
        fn gqa_broadcast_reduces_to_mha_bit_identically() {
            let d = 6;
            let head_dim = 4;
            let groups = 2; // n_head_q / n_head_kv

            let cfg_gqa = cfg_with_kv_heads(1);
            let cfg_mha = cfg_with_kv_heads(2);

            let k1 = pattern_linear(d, head_dim, -0.07);
            let v1 = pattern_linear(d, head_dim, 0.05);
            let dec_gqa = decoder_with_kv(
                Linear {
                    w_t: k1.w_t.clone(),
                    in_features: k1.in_features,
                    out_features: k1.out_features,
                },
                Linear {
                    w_t: v1.w_t.clone(),
                    in_features: v1.in_features,
                    out_features: v1.out_features,
                },
                d,
                &cfg_gqa,
            );
            let dec_mha = decoder_with_kv(
                duplicate_kv(&k1, groups),
                duplicate_kv(&v1, groups),
                d,
                &cfg_mha,
            );

            let mut s_gqa = TextDecoderSession::cpu(&cfg_gqa, &dec_gqa).unwrap();
            let mut s_mha = TextDecoderSession::cpu(&cfg_mha, &dec_mha).unwrap();

            // Multi-token prefix + two incremental steps: covers both the
            // full-prefix attention and the KV-cache append path.
            s_gqa.step_into(&[1u32, 2, 3]).unwrap();
            s_mha.step_into(&[1u32, 2, 3]).unwrap();
            assert_eq!(
                s_gqa.last_logits_row(),
                s_mha.last_logits_row(),
                "prefix step must be bit-identical"
            );
            for tok in [0u32, 4] {
                s_gqa.step_into(&[tok]).unwrap();
                s_mha.step_into(&[tok]).unwrap();
                assert_eq!(
                    s_gqa.last_logits_row(),
                    s_mha.last_logits_row(),
                    "incremental step (tok {tok}) must be bit-identical"
                );
            }
        }
    }
}
