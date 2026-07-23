//! Moshi Helium temporal backbone — pre-norm MHA transformer over summed
//! 17-channel token frames (M4-06-T09 weights + T10 forward + T13 paged KV).
//!
//! # Architecture (ADR M4-06 §D2 — transcribed, never invented)
//!
//! `kyutai-labs/moshi` `lm.py` `forward_text` + `transformer.py`:
//!
//! - input embedding = **sum** over the audio channels' embeddings plus
//!   the text channel's ([`MoshiBackbone::embed_step`]) — one sequence
//!   position per 12.5 Hz frame;
//! - `n_layer` pre-norm blocks: `rms_norm_f32` (ε from config, 1e-8
//!   upstream) → MHA with interleaved-pair RoPE (max_period 10 000,
//!   `csm::rope::rope_apply_adjacent` — the two repos share the
//!   adjacent-pair convention) → residual; norm → SiLU **gating** FFN
//!   (`linear_in` split into gate/up halves — gating.py) → residual;
//! - `out_norm` (`rms_norm_f32`) is applied to the transformer output
//!   **before both heads** (`text_linear` and the depformer input —
//!   lm.py `forward_text`), so this backbone returns the normed hidden;
//! - **sliding-window causal attention**: position `q` sees key `k` iff
//!   `0 <= q - k < context` (transformer.py `attn_bias = (delta >= 0) &
//!   (delta < context)`).
//!
//! # KV: M3-03 paged cache, single stream (T13 — ADR M4-06 §D1-(e))
//!
//! The user/moshi duplex directions live on the **channel axis** of the
//! summed embedding, not on separate KV streams (upstream runs one
//! transformer); the paged `[time, stream, codebook]` layout is used with
//! `n_stream = 1` (`BlockSize::Two`, 12.5 Hz-native) and its stream axis
//! stays available for future multi-session serving (FR-SV-06). Positions
//! past `max_ctx` are a **loud error** (FR-EX-08 — no silent wrap-around);
//! upstream's RingKVCache releases memory past `context` but its math is
//! exactly the sliding-window mask applied here, so semantics match while
//! memory is O(max_ctx) (eviction = follow-up optimization).
//!
//! # No silent fallback (FR-EX-08)
//!
//! Missing GGUF tensors → [`VokraError::ModelLoad`] naming the tensor;
//! out-of-range tokens / positions → [`VokraError::InvalidArgument`].

use std::sync::{Arc, Mutex};

use vokra_core::cache::paged::{BlockSize, KvDims, PagedKvCache};
use vokra_core::cache::ring::RingKvCache;
use vokra_core::gguf::{GgufFile, GgufTensorInfo};
use vokra_core::rng::SplitMix64;
use vokra_core::{BackendKind, Result, VokraError};

use super::config::MoshiConfig;
use crate::compute::{Compute, HotOp};
use crate::cosyvoice2::llm::LlmBlockWeights;
use crate::csm::backbone::xavier_uniform;
use crate::csm::rope::{llama3_inv_freqs, rope_apply_adjacent};
use crate::mapped_weights::{MappedModel, lock_scratch, mapped_info, transpose_widen, widen_into};

/// Error identity for the Moshi mapped store: the resident alternative is
/// `MoshiEngine::from_gguf_with_policy`, which dequantizes block-wise and so
/// accepts payload dtypes the mapped path cannot address per element.
const MOSHI_MAPPED: MappedModel = MappedModel {
    name: "moshi",
    resident_entry: "MoshiEngine::from_gguf_with_policy",
};
use crate::voxtral::text_decoder::{hadamard_inplace, rms_norm, silu_inplace};

/// Compute-seam hot ops the Moshi backbone + depformer dispatch (the CSM
/// set — GEMM projections/FFN, GEMV heads, softmax attention; RMSNorm /
/// gating glue is scalar host code). FA v3 is **never** part of this set
/// (M4-07 red-line).
pub(crate) const MOSHI_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::Gemv, HotOp::Softmax];

/// Sentinel for "no input on this channel": embeds as a **zero vector**
/// (`ScaledEmbedding.zero_idx = -1` upstream — lm_utils.py). Regular
/// vocabulary ids (including the initial tokens `card` / `text_card`) are
/// real embedding rows; this sentinel is the only non-row id accepted.
pub const MOSHI_ZERO_TOKEN: u32 = u32::MAX;

/// Seed for the synthesized fixture the GGUF loader falls back to on the
/// shape-only converter path (T29 real binding pending). Distinct from
/// the CSM / CosyVoice2 / Voxtral fixture seeds.
pub const MOSHI_FROM_GGUF_DEFAULT_SEED: u64 = 0x0405_1500_0405_1500;

/// All Helium temporal-backbone weights (ADR M4-06 §D2 manifest table).
///
/// Layouts:
/// - `text_emb` — `[(text_card + 1), d]` row-major (`text_emb.weight`;
///   row `text_card` = the text initial token);
/// - `audio_emb` — `[n_q_in, (audio_card + 1), d]` flattened (upstream
///   `emb.{k}.weight` tables concatenated in channel order; row
///   `audio_card` = the audio initial token);
/// - `blocks` — the shared [`LlmBlockWeights`] bundle at `n_head_kv ==
///   n_head` (MHA — `in_proj_weight` is split into Q/K/V thirds and
///   transposed, `gating.linear_in` into gate/up halves; ADR §D2);
/// - `out_norm_gamma` — `[d]` (`out_norm.alpha` flattened);
/// - `text_linear` — `[text_card, d]` row-major (GEMV layout,
///   `text_linear.weight` verbatim).
#[derive(Debug, Clone)]
pub struct MoshiBackboneWeights {
    /// Text-channel embedding `[(text_card + 1) * d]`.
    pub text_emb: Vec<f32>,
    /// Audio-channel embeddings `[n_q_in * (audio_card + 1) * d]`.
    pub audio_emb: Vec<f32>,
    /// Per-layer transformer blocks (shared bundle type — one fix lands
    /// for CosyVoice2 / Voxtral / CSM / Moshi).
    pub blocks: Vec<LlmBlockWeights>,
    /// `out_norm` γ `[d]` — applied before both heads (module docs).
    pub out_norm_gamma: Vec<f32>,
    /// Text head `[text_card, d]` row-major (GEMV layout).
    pub text_linear: Vec<f32>,
    /// `true` when built by [`Self::synthesized`]; real-checkpoint parity
    /// assertions gate on `false`.
    pub is_synthesized: bool,
}

impl MoshiBackboneWeights {
    /// Builds a synthesized (seed-deterministic) weight store — Xavier
    /// projections/embeddings, γ = 1 (the M3-09 recipe).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config` fails validation.
    pub fn synthesized(config: &MoshiConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let d = config.temporal.d_model;
        let h = config.temporal.ffn_hidden;
        let mut rng = SplitMix64::new(seed);
        let text_emb = xavier_uniform(&mut rng, (config.text_card + 1) * d, d, d);
        let audio_emb = xavier_uniform(&mut rng, config.n_q_in * (config.audio_card + 1) * d, d, d);
        let mut blocks = Vec::with_capacity(config.temporal.n_layer);
        for _ in 0..config.temporal.n_layer {
            blocks.push(LlmBlockWeights {
                attn_norm_gamma: vec![1.0f32; d],
                q_w_t: xavier_uniform(&mut rng, d * d, d, d),
                // Helium attention is bias-less (fused in_proj has no bias
                // tensor); the shared bias fields serve the Qwen2 family.
                q_b: None,
                k_w_t: xavier_uniform(&mut rng, d * d, d, d),
                k_b: None,
                v_w_t: xavier_uniform(&mut rng, d * d, d, d),
                v_b: None,
                o_w_t: xavier_uniform(&mut rng, d * d, d, d),
                ffn_norm_gamma: vec![1.0f32; d],
                ffn_gate_w_t: xavier_uniform(&mut rng, d * h, d, h),
                ffn_up_w_t: xavier_uniform(&mut rng, d * h, d, h),
                ffn_down_w_t: xavier_uniform(&mut rng, h * d, h, d),
            });
        }
        let out_norm_gamma = vec![1.0f32; d];
        let text_linear = xavier_uniform(&mut rng, config.text_card * d, d, config.text_card);
        Ok(Self {
            text_emb,
            audio_emb,
            blocks,
            out_norm_gamma,
            text_linear,
            is_synthesized: true,
        })
    }

    /// Binds real weights from a Moshi GGUF (upstream-verbatim tensor
    /// names — the T02 manifest is the single naming source; the packed
    /// `in_proj_weight` / `gating.linear_in.weight` tensors are split per
    /// the upstream `_load_hook` / gating conventions, ADR M4-06 §D2).
    ///
    /// This is the fully **resident** binding (every block widened to f32
    /// up front — ~30 GiB for the full-7B model). For bounded-memory
    /// loading of large checkpoints see [`Self::head_from_gguf`] +
    /// [`MappedTemporalBlocks`] (the `MoshiEngine::from_path` mmap path).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming any missing / mis-shaped tensor
    /// (never a silent zero-fill — FR-EX-08).
    pub fn from_gguf(file: &GgufFile, config: &MoshiConfig) -> Result<Self> {
        let mut this = Self::head_from_gguf(file, config)?;
        let d = config.temporal.d_model;
        let h = config.temporal.ffn_hidden;
        let mut blocks = Vec::with_capacity(config.temporal.n_layer);
        for i in 0..config.temporal.n_layer {
            let p = format!("transformer.layers.{i}");
            // Packed [3d, d] Q/K/V rows (manifest `in_proj_weight`).
            let in_proj = tensor_f32(file, &format!("{p}.self_attn.in_proj_weight"), 3 * d * d)?;
            let q_w_t = transpose(&in_proj[0..d * d], d, d);
            let k_w_t = transpose(&in_proj[d * d..2 * d * d], d, d);
            let v_w_t = transpose(&in_proj[2 * d * d..3 * d * d], d, d);
            let out_proj = tensor_f32(file, &format!("{p}.self_attn.out_proj.weight"), d * d)?;
            // Packed [2h, d] gate/up rows (gating.py view(B,T,2,h) split).
            let lin_in = tensor_f32(file, &format!("{p}.gating.linear_in.weight"), 2 * h * d)?;
            let ffn_gate_w_t = transpose(&lin_in[0..h * d], h, d);
            let ffn_up_w_t = transpose(&lin_in[h * d..2 * h * d], h, d);
            let lin_out = tensor_f32(file, &format!("{p}.gating.linear_out.weight"), d * h)?;
            blocks.push(LlmBlockWeights {
                attn_norm_gamma: tensor_f32(file, &format!("{p}.norm1.alpha"), d)?,
                q_w_t,
                // The Moshi checkpoint ships a fused `in_proj_weight` with
                // no bias tensor (Helium attention is bias-less).
                q_b: None,
                k_w_t,
                k_b: None,
                v_w_t,
                v_b: None,
                o_w_t: transpose(&out_proj, d, d),
                ffn_norm_gamma: tensor_f32(file, &format!("{p}.norm2.alpha"), d)?,
                ffn_gate_w_t,
                ffn_up_w_t,
                ffn_down_w_t: transpose(&lin_out, d, h),
            });
        }
        this.blocks = blocks;
        Ok(this)
    }

    /// Binds only the **head** weights (channel embeddings, `out_norm`,
    /// text head) from a Moshi GGUF, leaving `blocks` empty — the eager
    /// half of the bounded-memory load: the head is read on every step
    /// (embedding lookups + the text GEMV), so it stays resident f32,
    /// while the temporal blocks (the ~86% bulk of the 7B model) stay
    /// mapped and materialize per layer through [`MappedTemporalBlocks`].
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming any missing / mis-shaped tensor.
    pub fn head_from_gguf(file: &GgufFile, config: &MoshiConfig) -> Result<Self> {
        config.validate_for_forward()?;
        let d = config.temporal.d_model;

        let text_emb = tensor_f32(file, "text_emb.weight", (config.text_card + 1) * d)?;
        let mut audio_emb = Vec::with_capacity(config.n_q_in * (config.audio_card + 1) * d);
        for k in 0..config.n_q_in {
            let t = tensor_f32(
                file,
                &format!("emb.{k}.weight"),
                (config.audio_card + 1) * d,
            )?;
            audio_emb.extend_from_slice(&t);
        }
        Ok(Self {
            text_emb,
            audio_emb,
            blocks: Vec::new(),
            out_norm_gamma: tensor_f32(file, "out_norm.alpha", d)?,
            text_linear: tensor_f32(file, "text_linear.weight", config.text_card * d)?,
            is_synthesized: false,
        })
    }
}

/// Reads a named tensor as f32, enforcing the element count (loud
/// [`VokraError::ModelLoad`] on absence / size mismatch — FR-EX-08).
pub(crate) fn tensor_f32(file: &GgufFile, name: &str, want: usize) -> Result<Vec<f32>> {
    let v = file
        .tensor_f32(name)
        .map_err(|e| VokraError::ModelLoad(format!("moshi: tensor `{name}`: {e}")))?;
    if v.len() != want {
        return Err(VokraError::ModelLoad(format!(
            "moshi: tensor `{name}` has {} elements, expected {want}",
            v.len()
        )));
    }
    Ok(v)
}

/// Transposes a `[rows, cols]` row-major matrix into `[cols, rows]`
/// (torch `Linear.weight` `[out, in]` → the Compute-seam GEMM `w_t`
/// `[in, out]` layout).
pub(crate) fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(m.len(), rows * cols);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = m[r * cols + c];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Mapped-lazy temporal blocks (the bounded-memory full-7B load path)
// ---------------------------------------------------------------------------

/// One temporal layer's resolved GGUF tensor descriptors (validated at
/// bind — name, shape, dtype; payload bytes stay in the mapping).
#[derive(Debug)]
struct MappedLayerLocs {
    /// `{p}.self_attn.in_proj_weight` `[3d, d]` packed Q/K/V rows.
    in_proj: GgufTensorInfo,
    /// `{p}.self_attn.out_proj.weight` `[d, d]`.
    out_proj: GgufTensorInfo,
    /// `{p}.norm1.alpha` `[d]`.
    norm1: GgufTensorInfo,
    /// `{p}.norm2.alpha` `[d]`.
    norm2: GgufTensorInfo,
    /// `{p}.gating.linear_in.weight` `[2h, d]` packed gate/up rows.
    lin_in: GgufTensorInfo,
    /// `{p}.gating.linear_out.weight` `[d, h]`.
    lin_out: GgufTensorInfo,
}

/// The reused per-engine materialization target: exactly one temporal
/// layer's f32 block lives here at a time (~0.8 GiB at the 7B shape —
/// versus ~26 GiB for all 32 layers resident).
#[derive(Debug)]
struct MappedBlockScratch {
    block: LlmBlockWeights,
}

/// Temporal transformer blocks left **in the GGUF mapping** and widened
/// to f32 one layer at a time during the forward pass (M4 cc-06: the
/// full-7B `MoshiEngine::from_path` on a 16 GB machine).
///
/// - **Bind time** ([`Self::bind`]): every layer's six tensors are
///   resolved and shape/dtype-validated up front, so a malformed GGUF
///   fails at load, not mid-stream (FR-EX-08).
/// - **Step time** ([`Self::materialize_into`]): the requested layer is
///   widened + transposed straight out of the mapped bytes into the
///   reused scratch block, with **bit-identical** f32 values to the
///   resident [`MoshiBackboneWeights::from_gguf`] binding (same BF16
///   widen formula as `gguf::quant::dequantize`, same transpose index
///   math — pinned by `mapped_blocks_match_resident_bitwise`). The
///   trade is recurring per-step decode bandwidth for bounded memory.
/// - Supported payload dtypes: `F32` and `BF16` (the two the Moshi
///   converter emits). Anything else is a loud bind-time error naming
///   the resident loader as the alternative.
///
/// Holds the mapping alive via `Arc<GgufFile>`; the scratch sits behind a
/// `Mutex` so the owning engine stays `Send + Sync` (concurrent sessions
/// on one engine serialize their backbone forwards on this lock — the
/// bounded-memory choice; per-session scratch would multiply the ~0.8 GiB
/// buffer per session).
pub struct MappedTemporalBlocks {
    file: Arc<GgufFile>,
    locs: Vec<MappedLayerLocs>,
    scratch: Mutex<MappedBlockScratch>,
    d: usize,
    h: usize,
}

impl std::fmt::Debug for MappedTemporalBlocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedTemporalBlocks")
            .field("n_layer", &self.locs.len())
            .field("d", &self.d)
            .field("h", &self.h)
            .finish()
    }
}

impl MappedTemporalBlocks {
    /// Resolves and validates every temporal layer's tensors against
    /// `config` (upstream-verbatim names — the same manifest the
    /// resident loader binds).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming any missing tensor, any element
    /// count that disagrees with the config shapes, or any payload dtype
    /// outside `{F32, BF16}`.
    pub fn bind(file: Arc<GgufFile>, config: &MoshiConfig) -> Result<Self> {
        config.validate_for_forward()?;
        let d = config.temporal.d_model;
        let h = config.temporal.ffn_hidden;
        let mut locs = Vec::with_capacity(config.temporal.n_layer);
        for i in 0..config.temporal.n_layer {
            let p = format!("transformer.layers.{i}");
            locs.push(MappedLayerLocs {
                in_proj: mapped_info(
                    &file,
                    &format!("{p}.self_attn.in_proj_weight"),
                    3 * d * d,
                    MOSHI_MAPPED,
                )?,
                out_proj: mapped_info(
                    &file,
                    &format!("{p}.self_attn.out_proj.weight"),
                    d * d,
                    MOSHI_MAPPED,
                )?,
                norm1: mapped_info(&file, &format!("{p}.norm1.alpha"), d, MOSHI_MAPPED)?,
                norm2: mapped_info(&file, &format!("{p}.norm2.alpha"), d, MOSHI_MAPPED)?,
                lin_in: mapped_info(
                    &file,
                    &format!("{p}.gating.linear_in.weight"),
                    2 * h * d,
                    MOSHI_MAPPED,
                )?,
                lin_out: mapped_info(
                    &file,
                    &format!("{p}.gating.linear_out.weight"),
                    d * h,
                    MOSHI_MAPPED,
                )?,
            });
        }
        let scratch = Mutex::new(MappedBlockScratch {
            block: LlmBlockWeights {
                attn_norm_gamma: Vec::new(),
                q_w_t: Vec::new(),
                q_b: None,
                k_w_t: Vec::new(),
                k_b: None,
                v_w_t: Vec::new(),
                v_b: None,
                o_w_t: Vec::new(),
                ffn_norm_gamma: Vec::new(),
                ffn_gate_w_t: Vec::new(),
                ffn_up_w_t: Vec::new(),
                ffn_down_w_t: Vec::new(),
            },
        });
        Ok(Self {
            file,
            locs,
            scratch,
            d,
            h,
        })
    }

    /// Number of bound layers.
    #[must_use]
    pub fn n_layer(&self) -> usize {
        self.locs.len()
    }

    /// Locks the shared materialization scratch (whole-forward hold).
    fn lock_scratch(&self) -> Result<std::sync::MutexGuard<'_, MappedBlockScratch>> {
        lock_scratch(&self.scratch, MOSHI_MAPPED)
    }

    /// Widens + transposes layer `layer` out of the mapping into
    /// `scratch`, returning the materialized block (values bit-identical
    /// to the resident binding — type docs).
    fn materialize_into<'a>(
        &self,
        scratch: &'a mut MappedBlockScratch,
        layer: usize,
    ) -> Result<&'a LlmBlockWeights> {
        let locs = &self.locs[layer];
        let (d, h) = (self.d, self.h);
        let b = &mut scratch.block;

        // Packed [3d, d] Q/K/V thirds (resident: transpose of widened
        // sub-ranges; here the same sub-ranges are addressed in bytes).
        let bytes = self.file.tensor_bytes(&locs.in_proj);
        let esz = locs.in_proj.dtype.type_size();
        transpose_widen(
            &bytes[..d * d * esz],
            locs.in_proj.dtype,
            d,
            d,
            &mut b.q_w_t,
            MOSHI_MAPPED,
        )?;
        transpose_widen(
            &bytes[d * d * esz..2 * d * d * esz],
            locs.in_proj.dtype,
            d,
            d,
            &mut b.k_w_t,
            MOSHI_MAPPED,
        )?;
        transpose_widen(
            &bytes[2 * d * d * esz..3 * d * d * esz],
            locs.in_proj.dtype,
            d,
            d,
            &mut b.v_w_t,
            MOSHI_MAPPED,
        )?;
        transpose_widen(
            self.file.tensor_bytes(&locs.out_proj),
            locs.out_proj.dtype,
            d,
            d,
            &mut b.o_w_t,
            MOSHI_MAPPED,
        )?;
        widen_into(
            self.file.tensor_bytes(&locs.norm1),
            locs.norm1.dtype,
            &mut b.attn_norm_gamma,
            MOSHI_MAPPED,
        )?;
        widen_into(
            self.file.tensor_bytes(&locs.norm2),
            locs.norm2.dtype,
            &mut b.ffn_norm_gamma,
            MOSHI_MAPPED,
        )?;
        // Packed [2h, d] gate/up halves.
        let bytes = self.file.tensor_bytes(&locs.lin_in);
        let esz = locs.lin_in.dtype.type_size();
        transpose_widen(
            &bytes[..h * d * esz],
            locs.lin_in.dtype,
            h,
            d,
            &mut b.ffn_gate_w_t,
            MOSHI_MAPPED,
        )?;
        transpose_widen(
            &bytes[h * d * esz..2 * h * d * esz],
            locs.lin_in.dtype,
            h,
            d,
            &mut b.ffn_up_w_t,
            MOSHI_MAPPED,
        )?;
        transpose_widen(
            self.file.tensor_bytes(&locs.lin_out),
            locs.lin_out.dtype,
            d,
            h,
            &mut b.ffn_down_w_t,
            MOSHI_MAPPED,
        )?;
        // Helium attention is bias-less (fused in_proj has no bias
        // tensor); pinned here so a stale scratch can never leak one.
        b.q_b = None;
        b.k_b = None;
        b.v_b = None;
        Ok(&scratch.block)
    }
}

/// Pre-allocated per-state scratch, sized once for `t_cap` positions
/// (step path `t = 1` reuses it — FR-EX-05).
#[derive(Debug)]
struct Scratch {
    t_cap: usize,
    norm: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    rope_buf: Vec<f32>,
    k_hist: Vec<f32>,
    v_hist: Vec<f32>,
    scores: Vec<f32>,
    probs: Vec<f32>,
    attn_out: Vec<f32>,
    attn_o: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
    h: Vec<f32>,
}

impl Scratch {
    fn new(config: &MoshiConfig, t_cap: usize) -> Self {
        let d = config.temporal.d_model;
        let head_dim = config.temporal.head_dim();
        let h = config.temporal.ffn_hidden;
        let n_ctx = config.max_ctx;
        Self {
            t_cap,
            norm: vec![0.0; t_cap * d],
            q_proj: vec![0.0; t_cap * d],
            k_proj: vec![0.0; t_cap * d],
            v_proj: vec![0.0; t_cap * d],
            rope_buf: vec![0.0; t_cap * head_dim],
            k_hist: vec![0.0; n_ctx * d],
            v_hist: vec![0.0; n_ctx * d],
            scores: vec![0.0; t_cap * n_ctx],
            probs: vec![0.0; t_cap * n_ctx],
            attn_out: vec![0.0; t_cap * d],
            attn_o: vec![0.0; t_cap * d],
            ffn_gate: vec![0.0; t_cap * h],
            ffn_up: vec![0.0; t_cap * h],
            ffn_down: vec![0.0; t_cap * d],
            h: vec![0.0; t_cap * d],
        }
    }

    fn empty() -> Self {
        Self {
            t_cap: 0,
            norm: Vec::new(),
            q_proj: Vec::new(),
            k_proj: Vec::new(),
            v_proj: Vec::new(),
            rope_buf: Vec::new(),
            k_hist: Vec::new(),
            v_hist: Vec::new(),
            scores: Vec::new(),
            probs: Vec::new(),
            attn_out: Vec::new(),
            attn_o: Vec::new(),
            ffn_gate: Vec::new(),
            ffn_up: Vec::new(),
            ffn_down: Vec::new(),
            h: Vec::new(),
        }
    }
}

/// KV store backing a Moshi session (M4-06 #16).
///
/// - [`MoshiBackboneState::new`] uses the M3-03 [`PagedKvCache`] — it keeps the
///   full `max_ctx` history, which the shared bulk `forward` needs because a
///   bulk step reads an arbitrarily wide window (up to `context + t - 1`
///   positions once the clock is past the first window).
/// - [`MoshiBackboneState::new_bounded`] uses the M4-06 sliding-window
///   [`RingKvCache`] with `capacity = context`, so a **streaming** (one new
///   position per step) full-duplex session uses `O(context)` memory instead of
///   `O(max_ctx)`. A bulk step whose read span exceeds `capacity` surfaces a
///   loud error rather than silently reading evicted keys (FR-EX-08).
///
/// Both expose the same append / read / advance / reset surface, so
/// `forward_impl` is store-agnostic.
enum MoshiKvStore {
    Paged(PagedKvCache<f32>),
    Ring(RingKvCache<f32>),
}

impl MoshiKvStore {
    #[inline]
    fn append_step(
        &mut self,
        layer: usize,
        t: usize,
        s: usize,
        c: usize,
        k_row: &[f32],
        v_row: &[f32],
    ) -> Result<()> {
        match self {
            Self::Paged(p) => p.append_step(layer, t, s, c, k_row, v_row),
            Self::Ring(r) => r.append_step(layer, t, s, c, k_row, v_row),
        }
    }

    #[inline]
    fn read_step(&self, layer: usize, t: usize, s: usize, c: usize) -> Option<(&[f32], &[f32])> {
        match self {
            Self::Paged(p) => p.read_step(layer, t, s, c),
            Self::Ring(r) => r.read_step(layer, t, s, c),
        }
    }

    #[inline]
    fn advance(&mut self, n: usize) {
        match self {
            Self::Paged(p) => p.advance(n),
            Self::Ring(r) => r.advance(n),
        }
    }

    #[inline]
    fn reset(&mut self) {
        match self {
            Self::Paged(p) => p.reset(),
            Self::Ring(r) => r.reset(),
        }
    }

    #[inline]
    fn pages_in_use(&self) -> usize {
        match self {
            Self::Paged(p) => p.pages_in_use(),
            Self::Ring(r) => r.live_len(),
        }
    }
}

/// Autoregressive backbone state: position clock + KV store (single
/// stream — module docs) + step scratch. The KV arena is fully
/// pre-allocated; the decode loop only pops pages / ring slots.
pub struct MoshiBackboneState {
    seq_len: usize,
    kv: MoshiKvStore,
    scratch: Scratch,
}

impl std::fmt::Debug for MoshiBackboneState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoshiBackboneState")
            .field("seq_len", &self.seq_len)
            .field("pages_in_use", &self.kv.pages_in_use())
            .finish()
    }
}

impl MoshiBackboneState {
    /// KV shape for this config (single stream / codebook, `max_ctx` bound).
    fn kv_dims(config: &MoshiConfig) -> KvDims {
        KvDims {
            n_layer: config.temporal.n_layer,
            n_head: config.temporal.n_head,
            d_head: config.temporal.head_dim(),
            n_stream: 1,
            n_codebook: 1,
            max_time: config.max_ctx,
        }
    }

    /// Pre-allocates the paged arena (`max_ctx` positions) and scratch. This is
    /// the general-purpose state: it serves both bulk `forward` (wide-window
    /// priming) and incremental `step_into`.
    ///
    /// # Errors
    ///
    /// Propagates config validation and arena allocation errors.
    pub fn new(config: &MoshiConfig) -> Result<Self> {
        config.validate_for_forward()?;
        let kv = PagedKvCache::pre_allocate(Self::kv_dims(config), BlockSize::Two)?;
        Ok(Self {
            seq_len: 0,
            kv: MoshiKvStore::Paged(kv),
            scratch: Scratch::new(config, 1),
        })
    }

    /// Pre-allocates a **bounded** sliding-window ring KV (`capacity = context`)
    /// and scratch — `O(context)` memory instead of `O(max_ctx)`, for a
    /// long-running full-duplex **streaming** session (one new position per
    /// `step_into`). Reads over the streaming window are byte-identical to the
    /// paged history (the out-of-window keys the paged cache still holds
    /// contribute exactly zero to the sliding-window softmax — proven in
    /// [`RingKvCache`]'s tests and this module's `bounded_stream_matches_paged`).
    ///
    /// A bulk `forward` whose read span exceeds `context` (only reachable once
    /// the clock is past the first window with `t > 1`) is rejected loudly by
    /// `forward_impl` rather than reading evicted keys (FR-EX-08); use
    /// [`Self::new`] for wide-window bulk priming.
    ///
    /// # Errors
    ///
    /// Propagates config validation and ring allocation errors.
    pub fn new_bounded(config: &MoshiConfig) -> Result<Self> {
        config.validate_for_forward()?;
        let kv = RingKvCache::pre_allocate(Self::kv_dims(config), config.context)?;
        Ok(Self {
            seq_len: 0,
            kv: MoshiKvStore::Ring(kv),
            scratch: Scratch::new(config, 1),
        })
    }

    /// Frame positions consumed so far.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Rewinds to position 0, returning every page to the pre-allocated
    /// free list (no realloc — fast barge-in reset, T18).
    pub fn reset(&mut self) {
        self.seq_len = 0;
        self.kv.reset();
    }

    /// Paged-cache observability (tests / FR-EX-03 assertions).
    #[must_use]
    pub fn pages_in_use(&self) -> usize {
        self.kv.pages_in_use()
    }
}

/// The Helium temporal backbone (config + weights + backend selection).
pub struct MoshiBackbone {
    config: MoshiConfig,
    /// Head weights always resident; in mapped mode `weights.blocks` is
    /// **empty** and the temporal blocks come from `mapped` instead.
    weights: MoshiBackboneWeights,
    /// `Some` on the bounded-memory path ([`Self::new_mapped`]): the
    /// temporal blocks stay in the GGUF mapping and materialize one layer
    /// at a time during the forward (bit-identical values — type docs).
    mapped: Option<MappedTemporalBlocks>,
    backend: BackendKind,
    /// Plain (unscaled) interleaved-pair RoPE frequencies `[head_dim/2]`
    /// at the config max_period (ADR M4-06 §D2 gap analysis: the
    /// `csm::rope` helpers with `scaling = None` are exactly the moshi
    /// `apply_rope(interleave=True)` convention).
    inv_freqs: Vec<f32>,
}

impl std::fmt::Debug for MoshiBackbone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoshiBackbone")
            .field("config", &self.config)
            .field("weights.is_synthesized", &self.weights.is_synthesized)
            .field("mapped", &self.mapped.is_some())
            .field("backend", &self.backend)
            .finish()
    }
}

/// Per-forward block dispatch: resident slice or mapped materialization
/// (the guard is held for the whole forward so consecutive layers reuse
/// one scratch allocation).
enum BlockSource<'a> {
    Resident(&'a [LlmBlockWeights]),
    Mapped(
        &'a MappedTemporalBlocks,
        std::sync::MutexGuard<'a, MappedBlockScratch>,
    ),
}

impl BlockSource<'_> {
    /// The block for `layer` (materializing it on the mapped path).
    fn block(&mut self, layer: usize) -> Result<&LlmBlockWeights> {
        match self {
            Self::Resident(blocks) => Ok(&blocks[layer]),
            Self::Mapped(m, guard) => m.materialize_into(guard, layer),
        }
    }
}

impl MoshiBackbone {
    /// Builds a backbone from an explicit weight store (CPU backend).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on config / weight-shape mismatch.
    pub fn new(config: MoshiConfig, weights: MoshiBackboneWeights) -> Result<Self> {
        config.validate_for_forward()?;
        validate_head_shapes(&config, &weights)?;
        validate_block_shapes(&config, &weights)?;
        let inv_freqs = llama3_inv_freqs(config.temporal.head_dim(), config.rope_max_period, None)?;
        Ok(Self {
            config,
            weights,
            mapped: None,
            backend: BackendKind::Cpu,
            inv_freqs,
        })
    }

    /// Builds a **bounded-memory** backbone: `head` carries the resident
    /// head weights ([`MoshiBackboneWeights::head_from_gguf`] — `blocks`
    /// must be empty), while the temporal blocks stay in the GGUF mapping
    /// (`mapped`, validated at bind) and materialize per layer during the
    /// forward. Numerically **bit-identical** to the resident path
    /// ([`MappedTemporalBlocks`] docs); the trade is per-step decode
    /// bandwidth for ~1/8 the resident footprint at the 7B shape.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on head-shape mismatch, a
    /// non-empty `head.blocks` (ambiguous double weight source — loud,
    /// FR-EX-08), or a layer-count mismatch.
    pub fn new_mapped(
        config: MoshiConfig,
        head: MoshiBackboneWeights,
        mapped: MappedTemporalBlocks,
    ) -> Result<Self> {
        config.validate_for_forward()?;
        validate_head_shapes(&config, &head)?;
        if !head.blocks.is_empty() {
            return Err(VokraError::InvalidArgument(format!(
                "moshi MoshiBackbone::new_mapped: head store carries {} resident \
                 block(s) while a mapped block store is also supplied — ambiguous \
                 double source; bind the head with head_from_gguf (FR-EX-08)",
                head.blocks.len()
            )));
        }
        if mapped.n_layer() != config.temporal.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "moshi MoshiBackbone::new_mapped: mapped store has {} layers, \
                 config expects {}",
                mapped.n_layer(),
                config.temporal.n_layer
            )));
        }
        let inv_freqs = llama3_inv_freqs(config.temporal.head_dim(), config.rope_max_period, None)?;
        Ok(Self {
            config,
            weights: head,
            mapped: Some(mapped),
            backend: BackendKind::Cpu,
            inv_freqs,
        })
    }

    /// Whether the temporal blocks are mapped-lazy (`from_path` load)
    /// rather than resident f32.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        self.mapped.is_some()
    }

    /// Synthesized-fixture constructor.
    ///
    /// # Errors
    ///
    /// Propagates [`MoshiBackboneWeights::synthesized`].
    pub fn synthesized(config: MoshiConfig, seed: u64) -> Result<Self> {
        let weights = MoshiBackboneWeights::synthesized(&config, seed)?;
        Self::new(config, weights)
    }

    /// Selects the Compute-seam backend (explicit; unsupported ops on the
    /// selected backend fail loudly — FR-EX-08, no silent CPU fallback).
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// The resolved config.
    #[must_use]
    pub fn config(&self) -> &MoshiConfig {
        &self.config
    }

    /// The weight store (parity / shape assertions). On the mapped path
    /// ([`Self::new_mapped`]) this carries the resident **head** only —
    /// `blocks` is empty; the temporal blocks live in the mapping.
    #[must_use]
    pub fn weights(&self) -> &MoshiBackboneWeights {
        &self.weights
    }

    fn compute(&self) -> Result<Compute> {
        Compute::for_backend(self.backend, MOSHI_HOT_OPS)
    }

    /// Embeds one step's channel tokens into `out = [d]`: the **sum** of
    /// every non-[`MOSHI_ZERO_TOKEN`] channel's embedding row (lm.py
    /// `forward_text`; `tokens[0]` = text, `tokens[1..]` = audio channels
    /// in `emb.{k}` order).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a wrong channel count, an
    /// out-of-range token (valid ids include the initial tokens
    /// `card` / `text_card`), or a wrong-sized `out`.
    pub fn embed_step(&self, tokens: &[u32], out: &mut [f32]) -> Result<()> {
        let d = self.config.temporal.d_model;
        if out.len() != d {
            return Err(VokraError::InvalidArgument(format!(
                "moshi embed_step: out len {} != d_model {d}",
                out.len()
            )));
        }
        if tokens.len() != self.config.n_channels() {
            return Err(VokraError::InvalidArgument(format!(
                "moshi embed_step: {} tokens for {} channels",
                tokens.len(),
                self.config.n_channels()
            )));
        }
        out.iter_mut().for_each(|v| *v = 0.0);
        // Text channel (index 0).
        let text_tok = tokens[0];
        if text_tok != MOSHI_ZERO_TOKEN {
            let rows = self.config.text_card + 1;
            let tok = text_tok as usize;
            if tok >= rows {
                return Err(VokraError::InvalidArgument(format!(
                    "moshi embed_step: text token {tok} >= text rows {rows}"
                )));
            }
            let row = &self.weights.text_emb[tok * d..(tok + 1) * d];
            for (dst, src) in out.iter_mut().zip(row) {
                *dst += *src;
            }
        }
        // Audio channels (emb.{k} tables).
        let rows = self.config.audio_card + 1;
        for (k, &tok) in tokens[1..].iter().enumerate() {
            if tok == MOSHI_ZERO_TOKEN {
                continue;
            }
            let tok = tok as usize;
            if tok >= rows {
                return Err(VokraError::InvalidArgument(format!(
                    "moshi embed_step: audio token {tok} on channel {k} >= audio rows {rows}"
                )));
            }
            let base = (k * rows + tok) * d;
            let row = &self.weights.audio_emb[base..base + d];
            for (dst, src) in out.iter_mut().zip(row) {
                *dst += *src;
            }
        }
        Ok(())
    }

    /// Bulk forward over `steps` (each a `[n_channels]` token row),
    /// appending K/V to `state` and returning the **out_norm-applied**
    /// hidden states `[t, d]` (module docs — both heads read this).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on empty `steps`, a position past
    /// `max_ctx`, or any embed error; Compute-seam errors verbatim.
    pub fn forward(&self, steps: &[Vec<u32>], state: &mut MoshiBackboneState) -> Result<Vec<f32>> {
        if steps.is_empty() {
            return Err(VokraError::InvalidArgument(
                "moshi backbone forward: steps must be non-empty".into(),
            ));
        }
        let t = steps.len();
        let mut scratch = Scratch::new(&self.config, t);
        let mut hidden = vec![0.0f32; t * self.config.temporal.d_model];
        self.forward_impl(steps, state, &mut scratch, &mut hidden)?;
        Ok(hidden)
    }

    /// One autoregressive step with zero heap allocation (state scratch +
    /// pre-allocated pages only — FR-EX-05). `hidden_out = [d]` receives
    /// the out_norm-applied hidden state.
    ///
    /// # Errors
    ///
    /// Same surface as [`Self::forward`].
    pub fn step_into(
        &self,
        state: &mut MoshiBackboneState,
        tokens: &[u32],
        hidden_out: &mut [f32],
    ) -> Result<()> {
        if hidden_out.len() != self.config.temporal.d_model {
            return Err(VokraError::InvalidArgument(format!(
                "moshi backbone step: hidden_out len {} != d_model {}",
                hidden_out.len(),
                self.config.temporal.d_model
            )));
        }
        let steps = [tokens.to_vec()];
        let mut scratch = std::mem::replace(&mut state.scratch, Scratch::empty());
        let result = self.forward_impl(&steps, state, &mut scratch, hidden_out);
        state.scratch = scratch;
        result
    }

    /// Text-head logits from an out_norm-applied hidden state:
    /// `out = [text_card]` (`text_linear` GEMV).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on shape mismatch; Compute-seam
    /// errors verbatim.
    pub fn text_logits_into(&self, hidden: &[f32], out: &mut [f32]) -> Result<()> {
        let d = self.config.temporal.d_model;
        let vocab = self.config.text_card;
        if hidden.len() != d || out.len() != vocab {
            return Err(VokraError::InvalidArgument(format!(
                "moshi text_logits: hidden len {} (want {d}) / out len {} (want {vocab})",
                hidden.len(),
                out.len()
            )));
        }
        let compute = self.compute()?;
        compute.gemv_f32(vocab, d, &self.weights.text_linear, hidden, None, out)
    }

    /// The shared forward body (bulk t = N / step t = 1): pre-norm MHA
    /// blocks with sliding-window causal attention over the paged KV
    /// history, then `out_norm` into `hidden_out = [t, d]`.
    fn forward_impl(
        &self,
        steps: &[Vec<u32>],
        state: &mut MoshiBackboneState,
        scratch: &mut Scratch,
        hidden_out: &mut [f32],
    ) -> Result<()> {
        let t = steps.len();
        let cfg = &self.config.temporal;
        let d = cfg.d_model;
        let n_head = cfg.n_head;
        let head_dim = cfg.head_dim();
        let ffn_h = cfg.ffn_hidden;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let eps = self.config.rms_norm_eps;
        let context = self.config.context;
        let position_offset = state.seq_len;

        if position_offset + t > self.config.max_ctx {
            return Err(VokraError::InvalidArgument(format!(
                "moshi backbone: position {} + t {} > max_ctx {} (FR-EX-08 — no \
                 silent wrap-around; re-open or reset the session)",
                position_offset, t, self.config.max_ctx
            )));
        }
        if scratch.t_cap < t {
            return Err(VokraError::InvalidArgument(format!(
                "moshi backbone: scratch capacity {} < t {t} (internal sizing bug)",
                scratch.t_cap
            )));
        }
        if hidden_out.len() != t * d {
            return Err(VokraError::InvalidArgument(format!(
                "moshi backbone: hidden_out len {} != t*d {}",
                hidden_out.len(),
                t * d
            )));
        }
        let compute = self.compute()?;

        // Summed channel embeddings → h [t, d]. embed_step writes the
        // row in place (no extra buffer — the sum starts from zero).
        for (i, tokens) in steps.iter().enumerate() {
            self.embed_step(tokens, &mut scratch.h[i * d..(i + 1) * d])?;
        }

        let t_kv = position_offset + t;
        // Oldest key any query row in this step attends: the first (oldest)
        // row sits at `position_offset` and its window starts at
        // `(position_offset + 1) - context`. Keys below this are masked to
        // `-inf` for *every* row, so the snapshot skips them — which is what
        // lets a bounded ring cache (M4-06) back a streaming session: it holds
        // exactly `[t_kv - context, t_kv)`, and this lower bound never dips
        // below `t_kv - context` for a single-position step. (For the paged
        // cache this is a pure work-saving no-op; the skipped rows were masked.)
        let win_lo_global = (position_offset + 1).saturating_sub(context);
        // Resident blocks iterate the owned store; mapped blocks widen one
        // layer at a time into the engine's shared scratch (guard held for
        // the whole forward — BlockSource docs). Values are bit-identical
        // between the two sources (MappedTemporalBlocks docs).
        let mut blocks = match &self.mapped {
            Some(m) => BlockSource::Mapped(m, m.lock_scratch()?),
            None => BlockSource::Resident(&self.weights.blocks),
        };
        for layer_idx in 0..self.config.temporal.n_layer {
            let block = blocks.block(layer_idx)?;
            // ---------- Pre-norm MHA attention ----------
            rms_norm(
                &scratch.h[..t * d],
                &block.attn_norm_gamma,
                eps,
                t,
                &mut scratch.norm[..t * d],
            )?;
            // Q/K/V biases are `None` on every Moshi store (Helium is
            // bias-less), but honor the shared-struct contract so a bias,
            // if ever bound, is never silently dropped (FR-EX-08).
            compute.gemm_f32(
                t,
                d,
                d,
                &scratch.norm[..t * d],
                &block.q_w_t,
                block.q_b.as_deref(),
                &mut scratch.q_proj[..t * d],
            )?;
            compute.gemm_f32(
                t,
                d,
                d,
                &scratch.norm[..t * d],
                &block.k_w_t,
                block.k_b.as_deref(),
                &mut scratch.k_proj[..t * d],
            )?;
            compute.gemm_f32(
                t,
                d,
                d,
                &scratch.norm[..t * d],
                &block.v_w_t,
                block.v_b.as_deref(),
                &mut scratch.v_proj[..t * d],
            )?;

            // Interleaved-pair RoPE per head on Q and K (max_period from
            // config; adjacent-pair convention = upstream interleave=True).
            for h_i in 0..n_head {
                for i in 0..t {
                    let src = &scratch.q_proj[i * d + h_i * head_dim..i * d + (h_i + 1) * head_dim];
                    scratch.rope_buf[i * head_dim..(i + 1) * head_dim].copy_from_slice(src);
                }
                rope_apply_adjacent(
                    &mut scratch.rope_buf[..t * head_dim],
                    t,
                    head_dim,
                    &self.inv_freqs,
                    position_offset,
                )?;
                for i in 0..t {
                    let dst =
                        &mut scratch.q_proj[i * d + h_i * head_dim..i * d + (h_i + 1) * head_dim];
                    dst.copy_from_slice(&scratch.rope_buf[i * head_dim..(i + 1) * head_dim]);
                }
                for i in 0..t {
                    let src = &scratch.k_proj[i * d + h_i * head_dim..i * d + (h_i + 1) * head_dim];
                    scratch.rope_buf[i * head_dim..(i + 1) * head_dim].copy_from_slice(src);
                }
                rope_apply_adjacent(
                    &mut scratch.rope_buf[..t * head_dim],
                    t,
                    head_dim,
                    &self.inv_freqs,
                    position_offset,
                )?;
                for i in 0..t {
                    let dst =
                        &mut scratch.k_proj[i * d + h_i * head_dim..i * d + (h_i + 1) * head_dim];
                    dst.copy_from_slice(&scratch.rope_buf[i * head_dim..(i + 1) * head_dim]);
                }
            }

            // Append new K/V rows to the paged cache, snapshot the history.
            for i in 0..t {
                state.kv.append_step(
                    layer_idx,
                    position_offset + i,
                    0,
                    0,
                    &scratch.k_proj[i * d..(i + 1) * d],
                    &scratch.v_proj[i * d..(i + 1) * d],
                )?;
            }
            for j in win_lo_global..t_kv {
                let (k_row, v_row) = state.kv.read_step(layer_idx, j, 0, 0).ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "moshi backbone: KV history hole at layer {layer_idx} t {j} (window \
                         [{win_lo_global}, {t_kv})) — a bounded ring session cannot serve a \
                         bulk step wider than context={context}; use MoshiBackboneState::new \
                         for wide-window priming (FR-EX-08 — no silent eviction)"
                    ))
                })?;
                scratch.k_hist[j * d..(j + 1) * d].copy_from_slice(k_row);
                scratch.v_hist[j * d..(j + 1) * d].copy_from_slice(v_row);
            }

            // MHA attention with the sliding-window causal mask:
            // key j visible iff 0 <= (q_pos - j) < context.
            let scores = &mut scratch.scores[..t * t_kv];
            let probs = &mut scratch.probs[..t * t_kv];
            for h_i in 0..n_head {
                for i in 0..t {
                    let q_row =
                        &scratch.q_proj[i * d + h_i * head_dim..i * d + (h_i + 1) * head_dim];
                    let row_start = i * t_kv;
                    let cur_pos = position_offset + i;
                    // Window start: keys older than `context` are masked
                    // (transformer.py `delta < context`).
                    let win_lo = (cur_pos + 1).saturating_sub(context);
                    for j in 0..t_kv {
                        if j < win_lo || j > cur_pos {
                            scores[row_start + j] = f32::NEG_INFINITY;
                            continue;
                        }
                        let k_row =
                            &scratch.k_hist[j * d + h_i * head_dim..j * d + (h_i + 1) * head_dim];
                        let mut s = 0.0f32;
                        for c in 0..head_dim {
                            s += q_row[c] * k_row[c];
                        }
                        scores[row_start + j] = s * scale;
                    }
                }
                compute.softmax_f32(scores, probs, t, t_kv)?;
                for i in 0..t {
                    let cur_pos = position_offset + i;
                    let win_lo = (cur_pos + 1).saturating_sub(context);
                    let out_dst =
                        &mut scratch.attn_out[i * d + h_i * head_dim..i * d + (h_i + 1) * head_dim];
                    for (c, out) in out_dst.iter_mut().enumerate() {
                        let mut sum = 0.0f32;
                        for j in win_lo..=cur_pos {
                            sum += probs[i * t_kv + j] * scratch.v_hist[j * d + h_i * head_dim + c];
                        }
                        *out = sum;
                    }
                }
            }

            compute.gemm_f32(
                t,
                d,
                d,
                &scratch.attn_out[..t * d],
                &block.o_w_t,
                None,
                &mut scratch.attn_o[..t * d],
            )?;
            for i in 0..t * d {
                scratch.h[i] += scratch.attn_o[i];
            }

            // ---------- Pre-norm SiLU-gating FFN ----------
            rms_norm(
                &scratch.h[..t * d],
                &block.ffn_norm_gamma,
                eps,
                t,
                &mut scratch.norm[..t * d],
            )?;
            compute.gemm_f32(
                t,
                ffn_h,
                d,
                &scratch.norm[..t * d],
                &block.ffn_gate_w_t,
                None,
                &mut scratch.ffn_gate[..t * ffn_h],
            )?;
            compute.gemm_f32(
                t,
                ffn_h,
                d,
                &scratch.norm[..t * d],
                &block.ffn_up_w_t,
                None,
                &mut scratch.ffn_up[..t * ffn_h],
            )?;
            silu_inplace(&mut scratch.ffn_gate[..t * ffn_h]);
            hadamard_inplace(
                &mut scratch.ffn_gate[..t * ffn_h],
                &scratch.ffn_up[..t * ffn_h],
            )?;
            compute.gemm_f32(
                t,
                d,
                ffn_h,
                &scratch.ffn_gate[..t * ffn_h],
                &block.ffn_down_w_t,
                None,
                &mut scratch.ffn_down[..t * d],
            )?;
            for i in 0..t * d {
                scratch.h[i] += scratch.ffn_down[i];
            }
        }
        state.kv.advance(t);
        state.seq_len += t;

        // out_norm into the caller's buffer (read by both heads).
        rms_norm(
            &scratch.h[..t * d],
            &self.weights.out_norm_gamma,
            eps,
            t,
            hidden_out,
        )?;
        Ok(())
    }
}

/// Head-weight shape checks (embeddings / out_norm / text head) — shared
/// by the resident and mapped constructors.
fn validate_head_shapes(config: &MoshiConfig, weights: &MoshiBackboneWeights) -> Result<()> {
    let d = config.temporal.d_model;
    let checks = [
        (
            "text_emb",
            weights.text_emb.len(),
            (config.text_card + 1) * d,
        ),
        (
            "audio_emb",
            weights.audio_emb.len(),
            config.n_q_in * (config.audio_card + 1) * d,
        ),
        ("out_norm_gamma", weights.out_norm_gamma.len(), d),
        (
            "text_linear",
            weights.text_linear.len(),
            config.text_card * d,
        ),
    ];
    for (name, got, want) in checks {
        if got != want {
            return Err(VokraError::InvalidArgument(format!(
                "moshi MoshiBackbone::new: {name} len {got} != expected {want}"
            )));
        }
    }
    Ok(())
}

/// Resident-block shape checks (the mapped path validates blocks at bind
/// through [`MappedTemporalBlocks::bind`] instead).
fn validate_block_shapes(config: &MoshiConfig, weights: &MoshiBackboneWeights) -> Result<()> {
    let d = config.temporal.d_model;
    let h = config.temporal.ffn_hidden;
    if weights.blocks.len() != config.temporal.n_layer {
        return Err(VokraError::InvalidArgument(format!(
            "moshi MoshiBackbone::new: blocks {} != n_layer {}",
            weights.blocks.len(),
            config.temporal.n_layer
        )));
    }
    for (i, b) in weights.blocks.iter().enumerate() {
        let checks = [
            ("attn_norm_gamma", b.attn_norm_gamma.len(), d),
            ("q_w_t", b.q_w_t.len(), d * d),
            ("k_w_t", b.k_w_t.len(), d * d),
            ("v_w_t", b.v_w_t.len(), d * d),
            ("o_w_t", b.o_w_t.len(), d * d),
            ("ffn_norm_gamma", b.ffn_norm_gamma.len(), d),
            ("ffn_gate_w_t", b.ffn_gate_w_t.len(), d * h),
            ("ffn_up_w_t", b.ffn_up_w_t.len(), d * h),
            ("ffn_down_w_t", b.ffn_down_w_t.len(), h * d),
        ];
        for (name, got, want) in checks {
            if got != want {
                return Err(VokraError::InvalidArgument(format!(
                    "moshi MoshiBackbone::new: block[{i}].{name} len {got} != expected {want}"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use vokra_core::gguf::GgmlType;

    use super::*;

    fn backbone() -> MoshiBackbone {
        MoshiBackbone::synthesized(MoshiConfig::tiny_for_tests(), 11).expect("backbone")
    }

    /// A valid step-token row: real ids on every channel.
    fn step_tokens(cfg: &MoshiConfig, seed: u32) -> Vec<u32> {
        let mut v = Vec::with_capacity(cfg.n_channels());
        v.push((seed as usize % cfg.text_card) as u32);
        for k in 0..cfg.n_q_in {
            v.push(((seed as usize + 3 * k + 1) % cfg.audio_card) as u32);
        }
        v
    }

    #[test]
    fn synthesized_weights_have_config_shapes() {
        let b = backbone();
        let cfg = b.config().clone();
        let w = b.weights();
        assert_eq!(w.text_emb.len(), (cfg.text_card + 1) * cfg.temporal.d_model);
        assert_eq!(
            w.audio_emb.len(),
            cfg.n_q_in * (cfg.audio_card + 1) * cfg.temporal.d_model
        );
        assert_eq!(w.blocks.len(), cfg.temporal.n_layer);
        assert!(w.is_synthesized);
    }

    #[test]
    fn embed_step_sums_channels_and_honors_zero_token() {
        let b = backbone();
        let cfg = b.config().clone();
        let d = cfg.temporal.d_model;
        let tokens = step_tokens(&cfg, 5);

        // Full sum.
        let mut full = vec![0.0f32; d];
        b.embed_step(&tokens, &mut full).unwrap();

        // Text-only (audio channels zeroed) + audio-only must add up.
        let mut text_only_tokens = tokens.clone();
        for t in text_only_tokens.iter_mut().skip(1) {
            *t = MOSHI_ZERO_TOKEN;
        }
        let mut text_only = vec![0.0f32; d];
        b.embed_step(&text_only_tokens, &mut text_only).unwrap();

        let mut audio_only_tokens = tokens.clone();
        audio_only_tokens[0] = MOSHI_ZERO_TOKEN;
        let mut audio_only = vec![0.0f32; d];
        b.embed_step(&audio_only_tokens, &mut audio_only).unwrap();

        for i in 0..d {
            assert!(
                (full[i] - (text_only[i] + audio_only[i])).abs() < 1e-6,
                "sum contract at {i}"
            );
        }

        // All-zero row embeds to the zero vector (upstream zero_idx).
        let all_zero = vec![MOSHI_ZERO_TOKEN; cfg.n_channels()];
        let mut z = vec![1.0f32; d];
        b.embed_step(&all_zero, &mut z).unwrap();
        assert!(z.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn initial_tokens_are_valid_embedding_rows() {
        // `_get_initial_token`: audio initial = card, text initial =
        // text_card — the "+1" table rows (lm.py).
        let b = backbone();
        let cfg = b.config().clone();
        let mut tokens = vec![cfg.text_initial_token()];
        tokens.extend(std::iter::repeat_n(cfg.audio_initial_token(), cfg.n_q_in));
        let mut out = vec![0.0f32; cfg.temporal.d_model];
        b.embed_step(&tokens, &mut out).unwrap();
        assert!(out.iter().any(|v| *v != 0.0), "initial rows are real rows");
    }

    #[test]
    fn embed_rejects_out_of_range_and_wrong_arity() {
        let b = backbone();
        let cfg = b.config().clone();
        let d = cfg.temporal.d_model;
        let mut out = vec![0.0f32; d];
        // One past the initial token is out of range.
        let mut bad = step_tokens(&cfg, 0);
        bad[0] = cfg.text_initial_token() + 1;
        assert!(b.embed_step(&bad, &mut out).is_err());
        let mut bad = step_tokens(&cfg, 0);
        bad[1] = cfg.audio_initial_token() + 1;
        assert!(b.embed_step(&bad, &mut out).is_err());
        // Wrong channel arity.
        assert!(
            b.embed_step(&bad[..cfg.n_channels() - 1], &mut out)
                .is_err()
        );
    }

    #[test]
    fn forward_is_finite_and_deterministic() {
        let b = backbone();
        let cfg = b.config().clone();
        let steps: Vec<Vec<u32>> = (0..3).map(|i| step_tokens(&cfg, i)).collect();
        let mut s1 = MoshiBackboneState::new(&cfg).unwrap();
        let h1 = b.forward(&steps, &mut s1).unwrap();
        assert_eq!(h1.len(), steps.len() * cfg.temporal.d_model);
        assert!(h1.iter().all(|v| v.is_finite()));
        let mut s2 = MoshiBackboneState::new(&cfg).unwrap();
        let h2 = b.forward(&steps, &mut s2).unwrap();
        assert_eq!(h1, h2, "same weights + input → bit-identical");
    }

    #[test]
    fn forward_matches_step_by_step() {
        // Bulk causal forward vs incremental steps through the paged KV —
        // the M3-09 property (T15 anchor).
        let b = backbone();
        let cfg = b.config().clone();
        let steps: Vec<Vec<u32>> = (0..4).map(|i| step_tokens(&cfg, i * 7 + 1)).collect();
        let d = cfg.temporal.d_model;
        let mut bulk_state = MoshiBackboneState::new(&cfg).unwrap();
        let bulk = b.forward(&steps, &mut bulk_state).unwrap();
        let mut step_state = MoshiBackboneState::new(&cfg).unwrap();
        let mut last = vec![0.0f32; d];
        for s in &steps {
            b.step_into(&mut step_state, s, &mut last).unwrap();
        }
        let bulk_last = &bulk[(steps.len() - 1) * d..];
        let max_delta = bulk_last
            .iter()
            .zip(last.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_delta <= 1e-4, "bulk vs step max |Δ| = {max_delta}");
        assert_eq!(bulk_state.seq_len(), step_state.seq_len());
    }

    #[test]
    fn sliding_window_context_masks_old_positions() {
        // Same weights, two configs differing only in `context`: once the
        // clock passes the window, the constrained config must diverge
        // from the unconstrained one (the transformer.py `delta < context`
        // mask actually bites); before that, they agree.
        let mut narrow_cfg = MoshiConfig::tiny_for_tests();
        narrow_cfg.context = 2;
        let wide_cfg = MoshiConfig::tiny_for_tests(); // context = 32 ≥ steps
        let narrow = MoshiBackbone::synthesized(narrow_cfg.clone(), 99).unwrap();
        let wide = MoshiBackbone::synthesized(wide_cfg.clone(), 99).unwrap();

        let steps: Vec<Vec<u32>> = (0..4).map(|i| step_tokens(&wide_cfg, i + 2)).collect();
        let d = wide_cfg.temporal.d_model;

        let mut sn = MoshiBackboneState::new(&narrow_cfg).unwrap();
        let mut sw = MoshiBackboneState::new(&wide_cfg).unwrap();
        let hn = narrow.forward(&steps, &mut sn).unwrap();
        let hw = wide.forward(&steps, &mut sw).unwrap();

        // Position 0 and 1 see identical windows (delta < 2 covers both).
        for i in 0..2 * d {
            assert!(
                (hn[i] - hw[i]).abs() <= 1e-6,
                "inside the window the mask is inert (idx {i})"
            );
        }
        // Position 3 attends to {2, 3} only under context=2 — must differ.
        let tail_delta = hn[3 * d..]
            .iter()
            .zip(hw[3 * d..].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            tail_delta > 1e-6,
            "context mask must change the out-of-window position (Δ = {tail_delta})"
        );
    }

    #[test]
    fn text_logits_shape_and_error_paths() {
        let b = backbone();
        let cfg = b.config().clone();
        let mut state = MoshiBackboneState::new(&cfg).unwrap();
        let hidden = {
            let mut h = vec![0.0f32; cfg.temporal.d_model];
            b.step_into(&mut state, &step_tokens(&cfg, 1), &mut h)
                .unwrap();
            h
        };
        let mut logits = vec![0.0f32; cfg.text_card];
        b.text_logits_into(&hidden, &mut logits).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
        assert!(b.text_logits_into(&hidden[1..], &mut logits).is_err());
    }

    #[test]
    fn position_past_max_ctx_is_a_loud_error() {
        let mut cfg = MoshiConfig::tiny_for_tests();
        cfg.max_ctx = 2;
        let b = MoshiBackbone::synthesized(cfg.clone(), 3).unwrap();
        let mut state = MoshiBackboneState::new(&cfg).unwrap();
        let mut h = vec![0.0f32; cfg.temporal.d_model];
        b.step_into(&mut state, &step_tokens(&cfg, 0), &mut h)
            .unwrap();
        b.step_into(&mut state, &step_tokens(&cfg, 1), &mut h)
            .unwrap();
        let err = b
            .step_into(&mut state, &step_tokens(&cfg, 2), &mut h)
            .unwrap_err();
        assert!(
            err.to_string().contains("max_ctx"),
            "names the bound: {err}"
        );
    }

    #[test]
    fn reset_reuses_the_preallocated_arena_and_reproduces() {
        let b = backbone();
        let cfg = b.config().clone();
        let mut state = MoshiBackboneState::new(&cfg).unwrap();
        let mut h1 = vec![0.0f32; cfg.temporal.d_model];
        b.step_into(&mut state, &step_tokens(&cfg, 4), &mut h1)
            .unwrap();
        assert!(state.pages_in_use() > 0);
        state.reset();
        assert_eq!(state.pages_in_use(), 0);
        assert_eq!(state.seq_len(), 0);
        let mut h2 = vec![0.0f32; cfg.temporal.d_model];
        b.step_into(&mut state, &step_tokens(&cfg, 4), &mut h2)
            .unwrap();
        assert_eq!(h1, h2, "post-reset step reproduces a fresh state");
    }

    /// Serializes a synthesized backbone into a GGUF carrying
    /// upstream-verbatim tensor names in the *packed* layouts
    /// (in_proj_weight `[3d, d]`, gating.linear_in `[2h, d]`), with every
    /// float payload encoded as `dtype` (`F32` verbatim; `BF16` by bit
    /// truncation — the resident and mapped loaders then decode the SAME
    /// stored bits, which is what the bitwise-equality tests compare).
    fn packed_backbone_gguf(
        cfg: &MoshiConfig,
        w: &MoshiBackboneWeights,
        dtype: GgmlType,
    ) -> Vec<u8> {
        use vokra_core::gguf::GgufBuilder;
        let d = cfg.temporal.d_model;
        let h = cfg.temporal.ffn_hidden;
        let enc = |v: &[f32]| -> Vec<u8> {
            match dtype {
                GgmlType::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                GgmlType::BF16 => v
                    .iter()
                    .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
                other => panic!("test encoder supports F32/BF16 only, got {other:?}"),
            }
        };

        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "moshi");
        b.add_tensor(
            "text_emb.weight",
            dtype,
            vec![(cfg.text_card + 1) as u64, d as u64],
            enc(&w.text_emb),
        )
        .unwrap();
        for k in 0..cfg.n_q_in {
            let rows = cfg.audio_card + 1;
            b.add_tensor(
                &format!("emb.{k}.weight"),
                dtype,
                vec![rows as u64, d as u64],
                enc(&w.audio_emb[k * rows * d..(k + 1) * rows * d]),
            )
            .unwrap();
        }
        for (i, blk) in w.blocks.iter().enumerate() {
            let p = format!("transformer.layers.{i}");
            // Re-pack Q/K/V into the upstream [3d, d] row-major layout.
            let mut in_proj = Vec::with_capacity(3 * d * d);
            in_proj.extend(transpose(&blk.q_w_t, d, d));
            in_proj.extend(transpose(&blk.k_w_t, d, d));
            in_proj.extend(transpose(&blk.v_w_t, d, d));
            b.add_tensor(
                &format!("{p}.self_attn.in_proj_weight"),
                dtype,
                vec![3 * d as u64, d as u64],
                enc(&in_proj),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.self_attn.out_proj.weight"),
                dtype,
                vec![d as u64, d as u64],
                enc(&transpose(&blk.o_w_t, d, d)),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.norm1.alpha"),
                dtype,
                vec![1, 1, d as u64],
                enc(&blk.attn_norm_gamma),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.norm2.alpha"),
                dtype,
                vec![1, 1, d as u64],
                enc(&blk.ffn_norm_gamma),
            )
            .unwrap();
            let mut lin_in = Vec::with_capacity(2 * h * d);
            lin_in.extend(transpose(&blk.ffn_gate_w_t, d, h));
            lin_in.extend(transpose(&blk.ffn_up_w_t, d, h));
            b.add_tensor(
                &format!("{p}.gating.linear_in.weight"),
                dtype,
                vec![2 * h as u64, d as u64],
                enc(&lin_in),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.gating.linear_out.weight"),
                dtype,
                vec![d as u64, h as u64],
                enc(&transpose(&blk.ffn_down_w_t, h, d)),
            )
            .unwrap();
        }
        b.add_tensor(
            "out_norm.alpha",
            dtype,
            vec![1, 1, d as u64],
            enc(&w.out_norm_gamma),
        )
        .unwrap();
        b.add_tensor(
            "text_linear.weight",
            dtype,
            vec![cfg.text_card as u64, d as u64],
            enc(&w.text_linear),
        )
        .unwrap();
        b.to_bytes().unwrap()
    }

    /// Builds a mapped-lazy backbone over `file` (head eager + blocks in
    /// the mapping) — the from_path assembly, at unit-test scale.
    fn mapped_backbone(file: Arc<GgufFile>, cfg: &MoshiConfig) -> MoshiBackbone {
        let head = MoshiBackboneWeights::head_from_gguf(&file, cfg).expect("head bind");
        let mapped = MappedTemporalBlocks::bind(file, cfg).expect("mapped bind");
        MoshiBackbone::new_mapped(cfg.clone(), head, mapped).expect("mapped backbone")
    }

    #[test]
    fn from_gguf_binds_upstream_named_tensors_round_trip() {
        // Pack a synthesized store into GGUF, bind it back through the
        // resident loader, and verify the exact forward reproduces.
        let cfg = MoshiConfig::tiny_for_tests();
        let src = MoshiBackbone::synthesized(cfg.clone(), 21).unwrap();

        let file =
            GgufFile::parse(packed_backbone_gguf(&cfg, src.weights(), GgmlType::F32)).unwrap();
        let loaded = MoshiBackboneWeights::from_gguf(&file, &cfg).expect("bind");
        assert!(!loaded.is_synthesized);
        let reloaded = MoshiBackbone::new(cfg.clone(), loaded).unwrap();

        let steps: Vec<Vec<u32>> = (0..2).map(|i| step_tokens(&cfg, i + 6)).collect();
        let mut s1 = MoshiBackboneState::new(&cfg).unwrap();
        let mut s2 = MoshiBackboneState::new(&cfg).unwrap();
        let h_src = src.forward(&steps, &mut s1).unwrap();
        let h_re = reloaded.forward(&steps, &mut s2).unwrap();
        assert_eq!(h_src, h_re, "pack → GGUF → unpack is exact");
    }

    #[test]
    fn mapped_blocks_match_resident_bitwise_over_a_real_mmap() {
        // The bounded-memory contract's numerical half: per-layer
        // materialization out of a REAL read-only mapping must reproduce
        // the resident f32 forward BIT-identically, step by step (same
        // widen formula, same transpose index math).
        let cfg = MoshiConfig::tiny_for_tests();
        let src = MoshiBackbone::synthesized(cfg.clone(), 33).unwrap();
        let bytes = packed_backbone_gguf(&cfg, src.weights(), GgmlType::F32);

        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-moshi-mapped-f32-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();

        let resident = {
            let file = GgufFile::parse(bytes).unwrap();
            let w = MoshiBackboneWeights::from_gguf(&file, &cfg).unwrap();
            MoshiBackbone::new(cfg.clone(), w).unwrap()
        };
        let mmapped = Arc::new(vokra_mmap::open_gguf(&path).expect("real mmap"));
        let mapped = mapped_backbone(mmapped, &cfg);
        assert!(mapped.is_mapped());
        assert!(
            mapped.weights().blocks.is_empty(),
            "mapped mode keeps no resident blocks"
        );

        let d = cfg.temporal.d_model;
        let mut s_res = MoshiBackboneState::new(&cfg).unwrap();
        let mut s_map = MoshiBackboneState::new(&cfg).unwrap();
        let mut h_res = vec![0.0f32; d];
        let mut h_map = vec![0.0f32; d];
        for i in 0..6u32 {
            let tokens = step_tokens(&cfg, i * 3 + 1);
            resident.step_into(&mut s_res, &tokens, &mut h_res).unwrap();
            mapped.step_into(&mut s_map, &tokens, &mut h_map).unwrap();
            assert_eq!(h_res, h_map, "step {i}: mapped must be bit-identical");
        }
        // The text head reads the shared resident head — also identical.
        let mut l_res = vec![0.0f32; cfg.text_card];
        let mut l_map = vec![0.0f32; cfg.text_card];
        resident.text_logits_into(&h_res, &mut l_res).unwrap();
        mapped.text_logits_into(&h_map, &mut l_map).unwrap();
        assert_eq!(l_res, l_map);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mapped_blocks_bf16_match_resident_bitwise() {
        // Same property over a BF16 GGUF (the streaming converter's real
        // output dtype): both loaders widen the same stored bf16 bits.
        let cfg = MoshiConfig::tiny_for_tests();
        let src = MoshiBackbone::synthesized(cfg.clone(), 55).unwrap();
        let bytes = packed_backbone_gguf(&cfg, src.weights(), GgmlType::BF16);
        let file = Arc::new(GgufFile::parse(bytes).unwrap());

        let resident = {
            let w = MoshiBackboneWeights::from_gguf(&file, &cfg).unwrap();
            MoshiBackbone::new(cfg.clone(), w).unwrap()
        };
        let mapped = mapped_backbone(Arc::clone(&file), &cfg);

        let d = cfg.temporal.d_model;
        let mut s_res = MoshiBackboneState::new(&cfg).unwrap();
        let mut s_map = MoshiBackboneState::new(&cfg).unwrap();
        let mut h_res = vec![0.0f32; d];
        let mut h_map = vec![0.0f32; d];
        for i in 0..4u32 {
            let tokens = step_tokens(&cfg, i * 7 + 2);
            resident.step_into(&mut s_res, &tokens, &mut h_res).unwrap();
            mapped.step_into(&mut s_map, &tokens, &mut h_map).unwrap();
            assert_eq!(h_res, h_map, "step {i}: bf16 mapped must be bit-identical");
        }
    }

    #[test]
    fn mapped_bind_rejects_unsupported_dtype_and_missing_tensors() {
        use vokra_core::gguf::GgufBuilder;
        let cfg = MoshiConfig::tiny_for_tests();
        let src = MoshiBackbone::synthesized(cfg.clone(), 9).unwrap();

        // An F16 layer tensor: bind must refuse loudly, naming the dtype
        // and the resident alternative (FR-EX-08 — never a silent widen).
        let bytes = packed_backbone_gguf(&cfg, src.weights(), GgmlType::F32);
        let base = GgufFile::parse(bytes).unwrap();
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "moshi");
        for t in base.tensors() {
            let name = t.name.clone();
            if name == "transformer.layers.0.self_attn.in_proj_weight" {
                // Re-encode this one tensor as F16 (payload content is
                // irrelevant — bind rejects on dtype before reading it).
                let n = t.element_count().unwrap();
                b.add_tensor(
                    &name,
                    GgmlType::F16,
                    t.dimensions.clone(),
                    vec![0u8; (n * 2) as usize],
                )
                .unwrap();
            } else {
                b.add_tensor(
                    &name,
                    t.dtype,
                    t.dimensions.clone(),
                    base.tensor_bytes(t).to_vec(),
                )
                .unwrap();
            }
        }
        let file = Arc::new(GgufFile::parse(b.to_bytes().unwrap()).unwrap());
        let err = MappedTemporalBlocks::bind(Arc::clone(&file), &cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("F16"), "names the dtype: {msg}");
        assert!(
            msg.contains("from_gguf_with_policy"),
            "points at the resident loader: {msg}"
        );

        // A missing layer tensor is a loud ModelLoad naming it.
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "moshi");
        let file = Arc::new(GgufFile::parse(b.to_bytes().unwrap()).unwrap());
        let err = MappedTemporalBlocks::bind(file, &cfg).unwrap_err();
        assert!(
            err.to_string()
                .contains("transformer.layers.0.self_attn.in_proj_weight"),
            "names the missing tensor: {err}"
        );
    }

    #[test]
    fn new_mapped_rejects_a_double_block_source() {
        // A head store still carrying resident blocks alongside a mapped
        // store is ambiguous — loud, never silently preferring one.
        let cfg = MoshiConfig::tiny_for_tests();
        let src = MoshiBackbone::synthesized(cfg.clone(), 13).unwrap();
        let file = Arc::new(
            GgufFile::parse(packed_backbone_gguf(&cfg, src.weights(), GgmlType::F32)).unwrap(),
        );
        let full = MoshiBackboneWeights::from_gguf(&file, &cfg).unwrap();
        let mapped = MappedTemporalBlocks::bind(Arc::clone(&file), &cfg).unwrap();
        let err = MoshiBackbone::new_mapped(cfg.clone(), full, mapped).unwrap_err();
        assert!(err.to_string().contains("ambiguous"), "{err}");
    }

    #[test]
    fn from_gguf_missing_tensor_is_a_loud_model_load_error() {
        use vokra_core::gguf::GgufBuilder;
        let cfg = MoshiConfig::tiny_for_tests();
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "moshi");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let err = MoshiBackboneWeights::from_gguf(&file, &cfg).unwrap_err();
        assert!(matches!(err, VokraError::ModelLoad(_)));
        assert!(
            err.to_string().contains("text_emb.weight"),
            "names the missing tensor: {err}"
        );
    }

    #[test]
    fn bounded_ring_streaming_matches_paged_and_is_memory_bounded() {
        // context < max_ctx so the ring actually evicts (the tiny default has
        // context = max_ctx = 32, which never wraps).
        let mut cfg = MoshiConfig::tiny_for_tests();
        cfg.context = 4; // max_ctx stays 32
        let b = MoshiBackbone::synthesized(cfg.clone(), 77).unwrap();
        let d = cfg.temporal.d_model;

        let mut paged = MoshiBackboneState::new(&cfg).unwrap();
        let mut ring = MoshiBackboneState::new_bounded(&cfg).unwrap();

        // Stream well past the window (20 ≫ context 4).
        for i in 0..20u32 {
            let tokens = step_tokens(&cfg, i * 5 + 1);
            let mut h_paged = vec![0.0f32; d];
            let mut h_ring = vec![0.0f32; d];
            b.step_into(&mut paged, &tokens, &mut h_paged).unwrap();
            b.step_into(&mut ring, &tokens, &mut h_ring).unwrap();
            // Sliding-window equivalence: the ring holds exactly the attendable
            // window, so the streamed hidden is byte-identical to the full
            // paged history (the out-of-window keys the paged cache still holds
            // contribute exactly 0 to the sliding-window softmax).
            assert_eq!(h_ring, h_paged, "ring vs paged diverged at step {i}");
            // Bounded memory: the ring's live set never exceeds context.
            assert!(
                ring.pages_in_use() <= cfg.context,
                "ring live set {} exceeded context {} at step {i}",
                ring.pages_in_use(),
                cfg.context
            );
        }
        // The ring saturates at its capacity; the paged cache has spilled into
        // many more pages (the unbounded-in-practice growth the ring avoids).
        assert_eq!(ring.pages_in_use(), cfg.context, "ring pinned at capacity");
        assert!(
            paged.pages_in_use() > cfg.context,
            "paged history grew past the window ({} pages)",
            paged.pages_in_use()
        );
        assert_eq!(ring.seq_len(), 20);
    }

    #[test]
    fn bounded_ring_rejects_wide_bulk_forward() {
        // A bulk step wider than the window needs keys the ring has already
        // evicted; that must be a loud FR-EX-08 error, never a silent wrong
        // answer from a stale slot.
        let mut cfg = MoshiConfig::tiny_for_tests();
        cfg.context = 2;
        let b = MoshiBackbone::synthesized(cfg.clone(), 5).unwrap();
        // Bulk forward of 3 positions at offset 0: row 0 needs key 0, but a
        // capacity-2 ring evicts key 0 the moment key 2 is appended.
        let steps: Vec<Vec<u32>> = (0..3).map(|i| step_tokens(&cfg, i + 1)).collect();
        let mut ring = MoshiBackboneState::new_bounded(&cfg).unwrap();
        let err = b.forward(&steps, &mut ring).unwrap_err();
        assert!(
            err.to_string().contains("context") || err.to_string().contains("ring"),
            "names the bounded-ring limitation: {err}"
        );
        // The identical bulk forward on the paged state is fine (full history).
        let mut paged = MoshiBackboneState::new(&cfg).unwrap();
        assert!(
            b.forward(&steps, &mut paged).is_ok(),
            "paged serves wide bulk"
        );
    }
}
