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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgufTensorSchema {
    /// Current converter output: inference-ready `bert.*` names and a
    /// pre-normalized shared relative-position table.
    Canonical,
    /// Early public release: verbatim HF `deberta.*` tensor names. The
    /// loader performs the same lossless rename/QK-sharing and one-time
    /// relative-embedding LayerNorm as the offline converter.
    LegacyHf,
}

fn detect_tensor_schema(g: &GgufFile) -> Result<GgufTensorSchema, VokraError> {
    let canonical = g.tensor_info("bert.embed.weight").is_some();
    let legacy = g
        .tensor_info("deberta.embeddings.word_embeddings.weight")
        .is_some();
    match (canonical, legacy) {
        (true, false) => Ok(GgufTensorSchema::Canonical),
        (false, true) => Ok(GgufTensorSchema::LegacyHf),
        (true, true) => Err(VokraError::ModelLoad(
            "deberta_v3 GGUF mixes canonical `bert.*` and legacy HF `deberta.*` tensor schemas; refusing ambiguous precedence (FR-EX-08)"
                .to_owned(),
        )),
        (false, false) => Err(VokraError::ModelLoad(
            "deberta_v3 GGUF contains neither canonical `bert.embed.weight` nor legacy HF `deberta.embeddings.word_embeddings.weight`"
                .to_owned(),
        )),
    }
}

fn legacy_shared_position_embedding(g: &GgufFile, d_model: usize) -> Result<Vec<f32>, VokraError> {
    let load = |name: &str| {
        g.tensor_f32(name)
            .map_err(|error| VokraError::ModelLoad(format!("{name}: {error}")))
    };
    let rel = load("deberta.encoder.rel_embeddings.weight")?;
    if d_model == 0 || rel.len() % d_model != 0 {
        return Err(VokraError::ModelLoad(format!(
            "deberta.encoder.rel_embeddings.weight has {} elements, not a positive multiple of d_model {d_model}",
            rel.len()
        )));
    }
    let gamma_present = g.tensor_info("deberta.encoder.LayerNorm.weight").is_some();
    let beta_present = g.tensor_info("deberta.encoder.LayerNorm.bias").is_some();
    match (gamma_present, beta_present) {
        (false, false) => Ok(rel),
        (true, true) => {
            let gamma = load("deberta.encoder.LayerNorm.weight")?;
            let beta = load("deberta.encoder.LayerNorm.bias")?;
            if gamma.len() != d_model || beta.len() != d_model {
                return Err(VokraError::ModelLoad(format!(
                    "deberta.encoder.LayerNorm gamma/beta lengths ({}/{}) do not match d_model {d_model}",
                    gamma.len(),
                    beta.len()
                )));
            }
            // HF `DebertaV2Encoder.get_rel_embedding` applies this once per
            // forward. The canonical converter performs the identical f32
            // operation offline and stores its result under
            // `bert.encoder.pos_embed.weight`.
            Ok(LayerNorm::new(gamma, beta, 1e-7).forward(
                &rel,
                rel.len() / d_model,
                d_model,
            ))
        }
        _ => Err(VokraError::ModelLoad(
            "legacy deberta_v3 GGUF has a partial `deberta.encoder.LayerNorm.{weight,bias}` pair; refusing to synthesize the missing half (FR-EX-08)"
                .to_owned(),
        )),
    }
}

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

    pub fn get_d_model(&self) -> usize {
        self.d_model
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
                bq_pos: None,
                bk_pos: None,
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

    /// Loads a `DebertaV3Encoder` from a current converter GGUF or the early
    /// public GGUF layout that retained verbatim Hugging Face tensor names.
    ///
    /// Schema detection is fail-closed: files that mix both layouts or contain
    /// neither layout are rejected. For the legacy layout, this loader performs
    /// the same Q/K sharing and shared-relative-embedding LayerNorm as the
    /// current offline converter.
    ///
    /// # Metadata keys (`vokra.bert.deberta_v3.*`)
    ///
    /// - `n_layers` (required), `vocab_size` (required)
    /// - `d_model` (default 1024), `n_heads` (default 16),
    ///   `n_pos_buckets` (default 512), `max_pos_dist` (default 512)
    ///
    /// # Canonical tensor names
    ///
    /// - `bert.embed.weight`, `bert.embed.ln.{gamma,beta}`
    /// - `bert.encoder.pos_embed.weight` — single shared table (v3's
    ///   inference-relevant delta vs v2: read **once**, cloned into every
    ///   layer's `AttnWeights.pos_embed` below).
    /// - `bert.encoder.layer.<i>.attn.{wq,wk,wv,wq_pos,wk_pos,w_out}.weight`
    /// - `bert.encoder.layer.<i>.attn.{wq,wk,wv,w_out}.bias`
    /// - `bert.encoder.layer.<i>.attn.{wq_pos,wk_pos}.bias` — **optional**
    ///   (WP-15). Loaded into [`AttnWeights::bq_pos`] / [`AttnWeights::bk_pos`]
    ///   when present; when absent, forward falls back to the content
    ///   biases `bq` / `bk` (backward-compat with pre-WP-15 GGUFs and
    ///   upstream `share_att_key=True` configs — see the [`AttnWeights`]
    ///   struct-level "Position-aware biases" section).
    /// - `bert.encoder.layer.<i>.ffn.{w1,w2}.{weight,bias}`
    /// - `bert.encoder.layer.<i>.ln{1,2}.{gamma,beta}`
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

        if n_heads == 0 || d_model % n_heads != 0 {
            return Err(VokraError::ModelLoad(format!(
                "vokra.bert.deberta_v3: d_model ({d_model}) not divisible by n_heads ({n_heads})"
            )));
        }
        let head_dim = d_model / n_heads;
        let tensor_schema = detect_tensor_schema(g)?;

        let load_tensor_f32 = |name: &str| -> Result<Vec<f32>, VokraError> {
            g.tensor_f32(name)
                .map_err(|e| VokraError::ModelLoad(format!("{name}: {e}")))
        };
        // WP-15: `wq_pos.bias` / `wk_pos.bias` are optional (see
        // `AttnWeights` "Position-aware biases"). Same probe pattern as
        // `DebertaV2Encoder::from_gguf`.
        let load_optional_tensor_f32 = |name: &str| -> Result<Option<Vec<f32>>, VokraError> {
            if g.tensor_info(name).is_some() {
                Ok(Some(load_tensor_f32(name)?))
            } else {
                Ok(None)
            }
        };

        let (embed, embed_ln) = match tensor_schema {
            GgufTensorSchema::Canonical => (
                load_tensor_f32("bert.embed.weight")?,
                LayerNorm::new(
                    load_tensor_f32("bert.embed.ln.gamma")?,
                    load_tensor_f32("bert.embed.ln.beta")?,
                    1e-7,
                ),
            ),
            GgufTensorSchema::LegacyHf => (
                load_tensor_f32("deberta.embeddings.word_embeddings.weight")?,
                LayerNorm::new(
                    load_tensor_f32("deberta.embeddings.LayerNorm.weight")?,
                    load_tensor_f32("deberta.embeddings.LayerNorm.bias")?,
                    1e-7,
                ),
            ),
        };

        // v3's only inference-relevant delta vs v2: one shared position
        // embedding table, read once and cloned into every layer below
        // (v2 reads a fresh `<layer>.attn.pos_embed.weight` tensor per layer).
        let shared_pos_embed = match tensor_schema {
            GgufTensorSchema::Canonical => load_tensor_f32("bert.encoder.pos_embed.weight")?,
            GgufTensorSchema::LegacyHf => legacy_shared_position_embedding(g, d_model)?,
        };

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("bert.encoder.layer.{i}");
            let legacy_p = format!("deberta.encoder.layer.{i}");
            let (wq, wk, wv, wq_pos, wk_pos, w_out, bq, bk, bv, bout, bq_pos, bk_pos) =
                match tensor_schema {
                    GgufTensorSchema::Canonical => (
                        load_tensor_f32(&format!("{p}.attn.wq.weight"))?,
                        load_tensor_f32(&format!("{p}.attn.wk.weight"))?,
                        load_tensor_f32(&format!("{p}.attn.wv.weight"))?,
                        load_tensor_f32(&format!("{p}.attn.wq_pos.weight"))?,
                        load_tensor_f32(&format!("{p}.attn.wk_pos.weight"))?,
                        load_tensor_f32(&format!("{p}.attn.w_out.weight"))?,
                        load_tensor_f32(&format!("{p}.attn.wq.bias"))?,
                        load_tensor_f32(&format!("{p}.attn.wk.bias"))?,
                        load_tensor_f32(&format!("{p}.attn.wv.bias"))?,
                        load_tensor_f32(&format!("{p}.attn.w_out.bias"))?,
                        load_optional_tensor_f32(&format!("{p}.attn.wq_pos.bias"))?,
                        load_optional_tensor_f32(&format!("{p}.attn.wk_pos.bias"))?,
                    ),
                    GgufTensorSchema::LegacyHf => {
                        let wq = load_tensor_f32(&format!(
                            "{legacy_p}.attention.self.query_proj.weight"
                        ))?;
                        let wk =
                            load_tensor_f32(&format!("{legacy_p}.attention.self.key_proj.weight"))?;
                        let bq =
                            load_tensor_f32(&format!("{legacy_p}.attention.self.query_proj.bias"))?;
                        let bk =
                            load_tensor_f32(&format!("{legacy_p}.attention.self.key_proj.bias"))?;
                        (
                            wq.clone(),
                            wk.clone(),
                            load_tensor_f32(&format!(
                                "{legacy_p}.attention.self.value_proj.weight"
                            ))?,
                            wq,
                            wk,
                            load_tensor_f32(&format!("{legacy_p}.attention.output.dense.weight"))?,
                            bq.clone(),
                            bk.clone(),
                            load_tensor_f32(&format!("{legacy_p}.attention.self.value_proj.bias"))?,
                            load_tensor_f32(&format!("{legacy_p}.attention.output.dense.bias"))?,
                            Some(bq),
                            Some(bk),
                        )
                    }
                };
            let w = AttnWeights {
                wq,
                wk,
                wv,
                wq_pos,
                wk_pos,
                w_out,
                pos_embed: shared_pos_embed.clone(),
                bq,
                bk,
                bv,
                bout,
                bq_pos,
                bk_pos,
            };
            let (ffn, ln1, ln2) = match tensor_schema {
                GgufTensorSchema::Canonical => (
                    FfnBlock::new(
                        load_tensor_f32(&format!("{p}.ffn.w1.weight"))?,
                        load_tensor_f32(&format!("{p}.ffn.w1.bias"))?,
                        load_tensor_f32(&format!("{p}.ffn.w2.weight"))?,
                        load_tensor_f32(&format!("{p}.ffn.w2.bias"))?,
                        d_model,
                        4 * d_model,
                    ),
                    LayerNorm::new(
                        load_tensor_f32(&format!("{p}.ln1.gamma"))?,
                        load_tensor_f32(&format!("{p}.ln1.beta"))?,
                        1e-7,
                    ),
                    LayerNorm::new(
                        load_tensor_f32(&format!("{p}.ln2.gamma"))?,
                        load_tensor_f32(&format!("{p}.ln2.beta"))?,
                        1e-7,
                    ),
                ),
                GgufTensorSchema::LegacyHf => (
                    FfnBlock::new(
                        load_tensor_f32(&format!("{legacy_p}.intermediate.dense.weight"))?,
                        load_tensor_f32(&format!("{legacy_p}.intermediate.dense.bias"))?,
                        load_tensor_f32(&format!("{legacy_p}.output.dense.weight"))?,
                        load_tensor_f32(&format!("{legacy_p}.output.dense.bias"))?,
                        d_model,
                        4 * d_model,
                    ),
                    LayerNorm::new(
                        load_tensor_f32(&format!("{legacy_p}.attention.output.LayerNorm.weight"))?,
                        load_tensor_f32(&format!("{legacy_p}.attention.output.LayerNorm.bias"))?,
                        1e-7,
                    ),
                    LayerNorm::new(
                        load_tensor_f32(&format!("{legacy_p}.output.LayerNorm.weight"))?,
                        load_tensor_f32(&format!("{legacy_p}.output.LayerNorm.bias"))?,
                        1e-7,
                    ),
                ),
            };
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
