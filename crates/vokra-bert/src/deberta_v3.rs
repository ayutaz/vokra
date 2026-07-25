//! DeBERTa v3 encoder — clean-room per arXiv:2111.09543.
//!
//! DeBERTa v3 differs from v2 in three ways, only one of which matters at
//! inference time:
//!
//! - **RTD** (Replaced Token Detection) pre-training objective — training
//!   only, irrelevant for inference.
//! - **ELECTRA-style generator/discriminator** training setup — training
//!   only, irrelevant for inference.
//! - **Shared position embedding** — v2 gives every layer its own
//!   `pos_embed` tensor; v3 shares a single position embedding table across
//!   all layers (§3.1, "gradient-disentangled embedding sharing"). This is
//!   the only difference that changes the inference-time computation graph.
//!
//! Structurally this module reuses [`crate::deberta_v2`]'s
//! [`AttnWeights`], [`DisentangledAttention`], [`EncoderLayer`],
//! [`FfnBlock`], and [`LayerNorm`] verbatim — v3's per-layer forward math is
//! identical to v2's, only how `pos_embed` is populated at load time
//! differs (one shared `Vec<f32>` cloned into each layer instead of one
//! tensor read per layer).
//!
//! # References (permissive only)
//!
//! - He, Gao, Chen 2021 (arXiv:2111.09543)
//! - HuggingFace transformers `deberta_v3` (Apache-2.0)
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
//! - github.com/fishaudio/Bert-VITS2 (AGPL-3.0)
//! - Any AGPL derivative of the above.

use crate::deberta_v2::{AttnWeights, DisentangledAttention, EncoderLayer, FfnBlock, LayerNorm};
use vokra_core::gguf::GgufFile;
use vokra_core::VokraError;

/// Full DeBERTa v3 encoder: token embedding lookup → embed LayerNorm →
/// N-layer transformer stack, with a single position embedding table
/// shared across all layers (v3's only inference-relevant delta vs v2).
pub struct DebertaV3Encoder {
    layers: Vec<EncoderLayer>,
    embed: Vec<f32>, // [vocab, d_model]
    embed_ln: LayerNorm,
    d_model: usize,
    vocab_size: usize,
}

impl DebertaV3Encoder {
    pub fn forward(&self, ids: &[u32]) -> Vec<f32> {
        let seq_len = ids.len();
        let mut hidden = vec![0.0_f32; seq_len * self.d_model];
        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            assert!(
                id < self.vocab_size,
                "token id {id} out of vocab {}",
                self.vocab_size
            );
            for d in 0..self.d_model {
                hidden[i * self.d_model + d] = self.embed[id * self.d_model + d];
            }
        }
        hidden = self.embed_ln.forward(&hidden, seq_len, self.d_model);
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }
        hidden
    }

    /// Builds a `DebertaV3Encoder` with deterministic synthetic weights, for
    /// structure/shape tests only (no real checkpoint involved).
    ///
    /// Builds one shared `pos_embed` table and clones it into every layer's
    /// [`AttnWeights`] — mirroring how `from_gguf` populates layers from a
    /// single `bert.encoder.pos_embed.weight` tensor.
    #[doc(hidden)]
    pub fn synthetic_for_test(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        vocab: usize,
        n_pos_buckets: i32,
    ) -> Self {
        let head_dim = d_model / n_heads;
        let shared_pos_embed: Vec<f32> = vec![0.001_f32; n_pos_buckets as usize * d_model];
        let make_layer = |pos_embed: Vec<f32>| {
            let w = AttnWeights {
                wq: vec![0.01_f32; d_model * d_model],
                wk: vec![0.01_f32; d_model * d_model],
                wv: vec![0.01_f32; d_model * d_model],
                wq_pos: vec![0.01_f32; d_model * d_model],
                wk_pos: vec![0.01_f32; d_model * d_model],
                w_out: vec![0.01_f32; d_model * d_model],
                pos_embed,
                bq: vec![0.0_f32; d_model],
                bk: vec![0.0_f32; d_model],
                bv: vec![0.0_f32; d_model],
                bout: vec![0.0_f32; d_model],
            };
            EncoderLayer {
                attn: DisentangledAttention::new(w, d_model, n_heads, head_dim, n_pos_buckets, 512),
                ffn: FfnBlock::new(
                    vec![0.01_f32; 4 * d_model * d_model],
                    vec![0.0; 4 * d_model],
                    vec![0.01_f32; d_model * 4 * d_model],
                    vec![0.0; d_model],
                    d_model,
                    4 * d_model,
                ),
                ln1: LayerNorm::new(vec![1.0; d_model], vec![0.0; d_model], 1e-7),
                ln2: LayerNorm::new(vec![1.0; d_model], vec![0.0; d_model], 1e-7),
            }
        };
        Self {
            layers: (0..n_layers)
                .map(|_| make_layer(shared_pos_embed.clone()))
                .collect(),
            embed: vec![0.01_f32; vocab * d_model],
            embed_ln: LayerNorm::new(vec![1.0; d_model], vec![0.0; d_model], 1e-7),
            d_model,
            vocab_size: vocab,
        }
    }

    /// Loads a `DebertaV3Encoder` from a GGUF file written by the SBV2
    /// converter.
    ///
    /// # Metadata keys (`vokra.bert.deberta_v3.*`)
    ///
    /// - `n_layers` (required), `vocab_size` (required)
    /// - `d_model` (default 1024), `n_heads` (default 16),
    ///   `n_pos_buckets` (default 512), `max_pos_dist` (default 512)
    ///
    /// # Tensor names
    ///
    /// - `bert.embed.weight`, `bert.embed.ln.{gamma,beta}`
    /// - `bert.encoder.pos_embed.weight` — single shared table (v3's
    ///   inference-relevant delta vs v2: read **once**, cloned into every
    ///   layer's `AttnWeights.pos_embed` below).
    /// - `bert.encoder.layer.<i>.attn.{wq,wk,wv,wq_pos,wk_pos,w_out}.weight`
    /// - `bert.encoder.layer.<i>.attn.{wq,wk,wv,w_out}.bias`
    /// - `bert.encoder.layer.<i>.ffn.{w1,w2}.{weight,bias}`
    /// - `bert.encoder.layer.<i>.ln{1,2}.{gamma,beta}`
    ///
    /// # Known limitation
    ///
    /// Same as [`crate::deberta_v2::DebertaV2Encoder::from_gguf`]:
    /// [`AttnWeights`] has no dedicated `bq_pos`/`bk_pos` fields, so the
    /// position-aware Q/K projections (`wq_pos`/`wk_pos`) are applied with
    /// the *content* biases (`bq`/`bk`) in
    /// [`DisentangledAttention::forward`]. No `wq_pos.bias`/`wk_pos.bias`
    /// tensors are read here.
    pub fn from_gguf(g: &GgufFile) -> Result<Self, VokraError> {
        let meta_u32 =
            |key: &str| -> Option<u32> { g.get(key).and_then(|v| v.as_u64()).map(|u| u as u32) };
        let require_u32 = |key: &str| -> Result<u32, VokraError> {
            meta_u32(key)
                .ok_or_else(|| VokraError::ModelLoad(format!("missing GGUF metadata key: {key}")))
        };

        let n_layers = require_u32("vokra.bert.deberta_v3.n_layers")? as usize;
        let d_model = meta_u32("vokra.bert.deberta_v3.d_model").unwrap_or(1024) as usize;
        let n_heads = meta_u32("vokra.bert.deberta_v3.n_heads").unwrap_or(16) as usize;
        let vocab_size = require_u32("vokra.bert.deberta_v3.vocab_size")? as usize;
        let n_pos_buckets = meta_u32("vokra.bert.deberta_v3.n_pos_buckets").unwrap_or(512) as i32;
        let max_pos_dist = meta_u32("vokra.bert.deberta_v3.max_pos_dist").unwrap_or(512) as i32;

        if n_heads == 0 || !d_model.is_multiple_of(n_heads) {
            return Err(VokraError::ModelLoad(format!(
                "vokra.bert.deberta_v3: d_model ({d_model}) not divisible by n_heads ({n_heads})"
            )));
        }
        let head_dim = d_model / n_heads;

        let load_tensor_f32 = |name: &str| -> Result<Vec<f32>, VokraError> {
            g.tensor_f32(name)
                .map_err(|e| VokraError::ModelLoad(format!("{name}: {e}")))
        };

        let embed = load_tensor_f32("bert.embed.weight")?;
        let embed_ln = LayerNorm::new(
            load_tensor_f32("bert.embed.ln.gamma")?,
            load_tensor_f32("bert.embed.ln.beta")?,
            1e-7,
        );

        // v3's only inference-relevant delta vs v2: one shared position
        // embedding table, read once and cloned into every layer below
        // (v2 reads a fresh `<layer>.attn.pos_embed.weight` tensor per layer).
        let shared_pos_embed = load_tensor_f32("bert.encoder.pos_embed.weight")?;

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("bert.encoder.layer.{i}");
            let w = AttnWeights {
                wq: load_tensor_f32(&format!("{p}.attn.wq.weight"))?,
                wk: load_tensor_f32(&format!("{p}.attn.wk.weight"))?,
                wv: load_tensor_f32(&format!("{p}.attn.wv.weight"))?,
                wq_pos: load_tensor_f32(&format!("{p}.attn.wq_pos.weight"))?,
                wk_pos: load_tensor_f32(&format!("{p}.attn.wk_pos.weight"))?,
                w_out: load_tensor_f32(&format!("{p}.attn.w_out.weight"))?,
                pos_embed: shared_pos_embed.clone(),
                bq: load_tensor_f32(&format!("{p}.attn.wq.bias"))?,
                bk: load_tensor_f32(&format!("{p}.attn.wk.bias"))?,
                bv: load_tensor_f32(&format!("{p}.attn.wv.bias"))?,
                bout: load_tensor_f32(&format!("{p}.attn.w_out.bias"))?,
            };
            let ffn = FfnBlock::new(
                load_tensor_f32(&format!("{p}.ffn.w1.weight"))?,
                load_tensor_f32(&format!("{p}.ffn.w1.bias"))?,
                load_tensor_f32(&format!("{p}.ffn.w2.weight"))?,
                load_tensor_f32(&format!("{p}.ffn.w2.bias"))?,
                d_model,
                4 * d_model,
            );
            let ln1 = LayerNorm::new(
                load_tensor_f32(&format!("{p}.ln1.gamma"))?,
                load_tensor_f32(&format!("{p}.ln1.beta"))?,
                1e-7,
            );
            let ln2 = LayerNorm::new(
                load_tensor_f32(&format!("{p}.ln2.gamma"))?,
                load_tensor_f32(&format!("{p}.ln2.beta"))?,
                1e-7,
            );
            layers.push(EncoderLayer {
                attn: DisentangledAttention::new(
                    w,
                    d_model,
                    n_heads,
                    head_dim,
                    n_pos_buckets,
                    max_pos_dist,
                ),
                ffn,
                ln1,
                ln2,
            });
        }

        Ok(Self {
            layers,
            embed,
            embed_ln,
            d_model,
            vocab_size,
        })
    }
}
