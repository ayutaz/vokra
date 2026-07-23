//! Zonos-v0.1 (transformer) — Zyphra's text-to-audio TTS with typed prefix
//! conditioning (SoTA plan Phase 1-5, 2026-07-24).
//!
//! # What Zonos-v0.1-transformer is (primary source)
//!
//! Zonos-v0.1 is Zyphra's open (Apache 2.0 code + weight) TTS that generates
//! discrete audio tokens autoregressively over a **single** GQA transformer
//! stack, with a **typed prefix conditioner** (espeak phonemes + speaker
//! embedding + Fourier / integer control conditioners) prepended to the
//! sequence. Architecture per
//! `huggingface.co/Zyphra/Zonos-v0.1-transformer/raw/main/config.json`
//! (fetched verbatim into this module — CLAUDE.md「ハルシネーション厳禁」):
//!
//! - **Backbone** (`config.backbone`): a single uniform stack of
//!   `n_layer=26` GQA transformer blocks. `d_model=2048`,
//!   `attn_mlp_d_intermediate=8192` (SwiGLU inner width),
//!   `norm_epsilon=1e-05`. **`rms_norm=false`**: Zonos uses
//!   `LayerNorm(weight + bias)`, **not** RMSNorm — this is the config's
//!   own toggle and diverges from the family default (Dia / CosyVoice2
//!   both use RMSNorm).
//! - **Attention** (`config.backbone.attn_cfg`): `causal=true`,
//!   `num_heads=16`, `num_heads_kv=4` (GQA broadcast 4:1),
//!   `rotary_emb_dim=128` per head, `rotary_emb_interleaved=true`,
//!   `qkv_proj_bias=false`, `out_proj_bias=false`. **All 26 layers are
//!   attention** (`attn_layer_idx = [0..26]`) — the transformer variant
//!   contains no SSM layers.
//! - **SwiGLU MLP** (upstream `zonos/backbone/_torch.py`):
//!   `y, gate = fc1(x).chunk(2, dim=-1); fc2(y * silu(gate))`. `fc1` has
//!   width `2 * d_intermediate` (packed for the chunk split).
//! - **Prefix conditioner** (`config.prefix_conditioner`): 7 typed
//!   conditioners consumed positionally before the codebook tokens —
//!   espeak phonemes, speaker embedding (`cond_dim=128`), and 5 Fourier /
//!   integer scalars (`emotion` `input_dim=8`, `fmax` [0, 24000],
//!   `pitch_std` [0, 400], `speaking_rate` [0, 40],
//!   `language_id` [-1, 126]). Each has a learned unconditional token
//!   (`uncond_type=learned`). This module carries the **descriptor**
//!   (names + type + numeric bounds) so a real conversion can bind the
//!   projection tensors; the projection weights themselves live in the
//!   scaffold's opaque `prefix_conditioner_state` slot until a follow-up
//!   wave decodes the tensor manifest.
//! - **Codebook I/O**: `embeddings` = 9 × `Embedding(1026, d_model)` (one
//!   per DAC codebook), `heads` = 9 × `Linear(d_model, 1025, bias=false)`.
//!   Special ids: `eos_token_id=1024`, `masked_token_id=1025` (the vocab
//!   `1026 = 1024 audio + eos + masked`; heads emit only `1025` because
//!   `masked` is never a valid output).
//! - **Delay pattern** (upstream `zonos/codebook_pattern.py::apply_delay_pattern`):
//!   codebook `k` is rolled by `k + 1` steps → the staircase
//!   `[1, 2, 3, 4, 5, 6, 7, 8, 9]`, one delay per DAC codebook. This
//!   diverges from Dia's `[0, 8, 9, 10, 11, 12, 13, 14, 15]` and is
//!   material for the parallel-teacher-forcing / AR-sampling paths.
//!
//! # Terminal codec (upstream primary source)
//!
//! Zonos decodes to PCM via **DAC 44.1 kHz** (`descript/dac_44khz`, loaded
//! upstream in `zonos/autoencoder.py::DACAutoencoder.__init__` via
//! `DacModel.from_pretrained("descript/dac_44khz")` — HF transformers
//! path, `dac.config.sampling_rate = 44_100`). 9 codebook channels
//! (`num_codebooks = 9`) match Dia's shape 1:1, so the same
//! [`DacCodecGguf`] binder handles both models — the codec GGUF is what
//! carries the sample rate (`vokra.dac.sample_rate`), and the runtime
//! cross-checks it against [`ZonosConfig::sample_rate`] in
//! [`ZonosTts::with_dac`].
//!
//! # What lands in this Phase 1-5 slice
//!
//! - [`ZonosConfig`] — every hparam transcribed from the primary source
//!   (no hardcoded fabrication; sample rate is inherited from DAC 44.1 kHz
//!   per the upstream autoencoder).
//! - [`ZonosWeights`] — a backbone + codebook weight store with a
//!   deterministic [`ZonosWeights::synthesized`] fixture (SplitMix64 +
//!   Xavier) so shape / dtype / size flow can be exercised without the
//!   real HF checkpoint. The prefix-conditioner projection tensors are
//!   left as opaque per-conditioner buffers (real conversion binds them
//!   once the tensor manifest is fetched).
//! - [`ZonosTts`] — engine handle carrying config + weights + optional
//!   DAC bind. [`ZonosTts::synthesize`] returns
//!   [`VokraError::NotImplemented`] until real weights are bound (the
//!   real forward — prefix conditioner → codebook-embedding sum →
//!   pre-norm GQA + SwiGLU stack → per-head logits → delayed AR sampling
//!   per channel → DAC decode → PCM — is a follow-up wave gated on the
//!   real-checkpoint tensor manifest).
//!
//! Real-checkpoint parity is deferred exactly like CosyVoice2 T02 / CSM
//! T29 / Dia (Phase 1-4): this scaffold sets the seam so the follow-up
//! lands drop-in.
//!
//! # No ONNX (permanent)
//!
//! Zonos ships as safetensors / a Python pipeline; the pipeline is
//! re-implemented natively (whisper.cpp 型, CLAUDE.md 設計判断 4). This
//! module never touches ONNX.

use vokra_core::rng::SplitMix64;
use vokra_core::{Result, VokraError};

use crate::codec::DacCodecGguf;

/// `vokra.model.arch` a Zonos GGUF must carry. Written by
/// `vokra-convert::models::zonos::ARCH`; the compliance registry
/// (`vokra_core::compliance`) knows `zonos` / `zonos-v0.1` as
/// [`LicenseClass::Permissive`](vokra_core::LicenseClass::Permissive)
/// (Apache 2.0 code + weight), so a stock Zonos GGUF passes the M2-13
/// gate without a research flag.
pub const EXPECTED_ARCH: &str = "zonos";

/// PCM sample rate Zonos emits. Not written in the upstream `config.json`;
/// inherited from **DAC 44.1 kHz** (`descript/dac_44khz` loaded upstream
/// in `zonos/autoencoder.py`).
pub const ZONOS_SAMPLE_RATE: u32 = 44_100;

/// Number of DAC codebook channels the Zonos-v0.1 decoder emits per step.
/// Wired to `DACAutoencoder.num_codebooks` (upstream constructor) — the
/// same 9 codebook channels as Dia so the two share the [`DacCodecGguf`]
/// bind.
pub const ZONOS_NUM_CODEBOOKS: usize = 9;

// ---------------------------------------------------------------------------
// Prefix conditioner descriptor
// ---------------------------------------------------------------------------

/// One typed conditioner in the prefix stack (primary source:
/// `config.prefix_conditioner.conditioners[i]`).
///
/// Zonos-v0.1 prepends 7 typed conditioning tokens before the codebook
/// token stream. Each entry has a **type** (which decides the projection
/// shape at real-conversion time) and, for the numeric conditioners, a
/// **numeric domain** (min/max). The projection weights themselves live in
/// [`ZonosWeights::prefix_conditioner_state`] as an opaque per-conditioner
/// `Vec<f32>` — the real converter binds them from the upstream tensor
/// manifest once fetched.
#[derive(Debug, Clone, PartialEq)]
pub enum ZonosConditionerKind {
    /// eSpeak-NG phoneme conditioner (`type=EspeakPhonemeConditioner`).
    /// eSpeak stays out of the Vokra runtime (GPL-3.0 — see CLAUDE.md 設計
    /// 判断 4); the descriptor only records that upstream Zonos consumed
    /// the phoneme id sequence here.
    EspeakPhoneme,
    /// Speaker-embedding pass-through with a linear projection
    /// (`type=PassthroughConditioner`, `cond_dim=128`).
    Speaker {
        /// `cond_dim` — width of the speaker embedding upstream passes in.
        cond_dim: u32,
    },
    /// Fourier feature encoder over a bounded scalar
    /// (`type=FourierConditioner`).
    Fourier {
        /// `input_dim` when explicitly written (`emotion`); otherwise `1`
        /// for the scalar conditioners (`fmax`, `pitch_std`,
        /// `speaking_rate`). Upstream defaults `input_dim=1` when the key
        /// is absent (`fourier_conditioner.py`).
        input_dim: u32,
        /// `min_val` of the numeric domain. Emotion carries no explicit
        /// range in the config (raw activation vector); default `0.0`.
        min_val: f32,
        /// `max_val` of the numeric domain. Emotion default `0.0`.
        max_val: f32,
    },
    /// Integer-embedding conditioner over a bounded id range
    /// (`type=IntegerConditioner`).
    Integer {
        /// `min_val` — inclusive lower id (Zonos `language_id` = -1 for
        /// "unset").
        min_val: i32,
        /// `max_val` — inclusive upper id.
        max_val: i32,
    },
}

/// A named prefix conditioner (name from `config.prefix_conditioner
/// .conditioners[i].name`).
///
/// Ordering follows the config verbatim; the runtime concatenates the
/// projected tokens in the same order in front of the codebook token
/// stream.
#[derive(Debug, Clone, PartialEq)]
pub struct ZonosConditioner {
    /// The `name` field from the config (e.g. `"speaker"`, `"language_id"`).
    pub name: String,
    /// The typed descriptor.
    pub kind: ZonosConditionerKind,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Backbone hparams (primary source: `config.backbone` + `config.backbone
/// .attn_cfg`).
///
/// Zonos-v0.1-transformer is a uniform stack: **every** layer is a GQA
/// attention block (`attn_layer_idx = [0..26]`); the transformer variant
/// contains no SSM layers.
#[derive(Debug, Clone, PartialEq)]
pub struct ZonosBackboneConfig {
    /// `n_layer` — 26 transformer blocks.
    pub n_layer: usize,
    /// `d_model` — hidden width, 2048.
    pub d_model: usize,
    /// `attn_mlp_d_intermediate` — SwiGLU FFN inner width, 8192. Note the
    /// packed fc1 width is `2 * d_intermediate` because SwiGLU chunks the
    /// pre-activation into `(y, gate)`.
    pub d_intermediate: usize,
    /// `attn_cfg.num_heads` — Q-heads (GQA), 16.
    pub num_heads: usize,
    /// `attn_cfg.num_heads_kv` — KV-heads (GQA broadcast), 4.
    pub num_heads_kv: usize,
    /// `attn_cfg.rotary_emb_dim` — RoPE per-head width, 128.
    pub rotary_emb_dim: usize,
    /// `attn_cfg.rotary_emb_interleaved` — RoPE variant (upstream
    /// `_torch.py` uses the interleaved fused kernel path when true).
    pub rotary_emb_interleaved: bool,
    /// `attn_cfg.causal` — always true for AR generation.
    pub causal: bool,
    /// `attn_cfg.qkv_proj_bias` — false for Zonos.
    pub qkv_proj_bias: bool,
    /// `attn_cfg.out_proj_bias` — false for Zonos.
    pub out_proj_bias: bool,
    /// `norm_epsilon` — LayerNorm ε (1e-5).
    pub norm_epsilon: f32,
    /// `rms_norm` — **false** for Zonos (LayerNorm with weight + bias).
    /// Kept as a config field so a future Zonos flavor toggling this to
    /// `true` does not need a new config type; the weight store keys off
    /// the same flag.
    pub rms_norm: bool,
}

impl ZonosBackboneConfig {
    /// GQA + head-dim + LayerNorm sanity: `d_model == num_heads *
    /// head_dim`, `num_heads % num_heads_kv == 0`, `rotary_emb_dim ==
    /// head_dim` (Zonos passes the same width to RoPE and the head).
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        self.n_layer != 0
            && self.d_model != 0
            && self.d_intermediate != 0
            && self.num_heads != 0
            && self.num_heads_kv != 0
            && self.rotary_emb_dim != 0
            && self.num_heads % self.num_heads_kv == 0
            && self.d_model % self.num_heads == 0
            && self.d_model / self.num_heads == self.rotary_emb_dim
    }

    /// Per-head width, `d_model / num_heads`.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads.max(1)
    }

    /// Q hidden width (rows of the Q projection), `num_heads * head_dim
    /// == d_model` for Zonos-v0.1.
    #[must_use]
    pub fn q_hidden(&self) -> usize {
        self.num_heads * self.head_dim()
    }

    /// KV hidden width, `num_heads_kv * head_dim`. For Zonos-v0.1 that
    /// is `4 * 128 = 512`.
    #[must_use]
    pub fn kv_hidden(&self) -> usize {
        self.num_heads_kv * self.head_dim()
    }

    /// Packed fc1 output width — SwiGLU chunks into `(y, gate)`, so the
    /// fc1 emits `2 * d_intermediate`.
    #[must_use]
    pub fn mlp_fc1_out(&self) -> usize {
        2 * self.d_intermediate
    }
}

/// Resolved Zonos hparam snapshot — every field is transcribed from the
/// upstream `config.json` (module docstring) or from the DAC codec Zonos
/// depends on (`sample_rate`).
#[derive(Debug, Clone, PartialEq)]
pub struct ZonosConfig {
    /// Transformer backbone hparams.
    pub backbone: ZonosBackboneConfig,
    /// Ordered prefix conditioners (verbatim from
    /// `config.prefix_conditioner.conditioners`).
    pub conditioners: Vec<ZonosConditioner>,
    /// Number of DAC codebook channels the decoder emits.
    /// `= ZONOS_NUM_CODEBOOKS = 9` (upstream
    /// `DACAutoencoder.num_codebooks`; matches Dia's shape 1:1 so the
    /// same [`DacCodecGguf`] binder works).
    pub num_codebooks: usize,
    /// Per-codebook input vocab (`Embedding(1026, d_model)` upstream —
    /// `1024 audio + eos_token_id + masked_token_id`).
    pub codebook_vocab: usize,
    /// Per-codebook head width (`Linear(d_model, 1025, bias=false)`
    /// upstream — `1024 audio + eos_token_id`; the masked id never
    /// emits, and upstream `_compute_logits` explicitly masks
    /// `logits[..., 1025:]` to `-inf`).
    pub head_vocab: usize,
    /// `eos_token_id` — 1024.
    pub eos_token_id: u32,
    /// `masked_token_id` — 1025 (never a valid emission; upstream
    /// clamps it out).
    pub masked_token_id: u32,
    /// Delay pattern from `zonos/codebook_pattern.py::apply_delay_pattern`:
    /// codebook `k` is rolled by `k + 1` steps → `[1, 2, ..., num_codebooks]`.
    pub delay_pattern: Vec<usize>,
    /// PCM sample rate — 44_100 (inherited from DAC 44.1 kHz, **not**
    /// written in the upstream `config.json`).
    pub sample_rate: u32,
}

impl ZonosConfig {
    /// Primary-source Zonos-v0.1-transformer config (every value
    /// transcribed from `huggingface.co/Zyphra/Zonos-v0.1-transformer/
    /// raw/main/config.json`).
    #[must_use]
    pub fn zonos_v0_1_transformer() -> Self {
        Self {
            backbone: ZonosBackboneConfig {
                n_layer: 26,
                d_model: 2048,
                d_intermediate: 8192,
                num_heads: 16,
                num_heads_kv: 4,
                rotary_emb_dim: 128,
                rotary_emb_interleaved: true,
                causal: true,
                qkv_proj_bias: false,
                out_proj_bias: false,
                norm_epsilon: 1e-5,
                rms_norm: false,
            },
            conditioners: vec![
                ZonosConditioner {
                    name: "espeak".to_owned(),
                    kind: ZonosConditionerKind::EspeakPhoneme,
                },
                ZonosConditioner {
                    name: "speaker".to_owned(),
                    kind: ZonosConditionerKind::Speaker { cond_dim: 128 },
                },
                ZonosConditioner {
                    name: "emotion".to_owned(),
                    kind: ZonosConditionerKind::Fourier {
                        input_dim: 8,
                        min_val: 0.0,
                        max_val: 0.0,
                    },
                },
                ZonosConditioner {
                    name: "fmax".to_owned(),
                    kind: ZonosConditionerKind::Fourier {
                        input_dim: 1,
                        min_val: 0.0,
                        max_val: 24_000.0,
                    },
                },
                ZonosConditioner {
                    name: "pitch_std".to_owned(),
                    kind: ZonosConditionerKind::Fourier {
                        input_dim: 1,
                        min_val: 0.0,
                        max_val: 400.0,
                    },
                },
                ZonosConditioner {
                    name: "speaking_rate".to_owned(),
                    kind: ZonosConditionerKind::Fourier {
                        input_dim: 1,
                        min_val: 0.0,
                        max_val: 40.0,
                    },
                },
                ZonosConditioner {
                    name: "language_id".to_owned(),
                    kind: ZonosConditionerKind::Integer {
                        min_val: -1,
                        max_val: 126,
                    },
                },
            ],
            num_codebooks: ZONOS_NUM_CODEBOOKS,
            codebook_vocab: 1026,
            head_vocab: 1025,
            eos_token_id: 1024,
            masked_token_id: 1025,
            // Zonos codebook_pattern.py — codebook k rolled by k+1.
            delay_pattern: (1..=ZONOS_NUM_CODEBOOKS).collect(),
            sample_rate: ZONOS_SAMPLE_RATE,
        }
    }

    /// Miniature well-formed config for shape / stability tests. Dims are
    /// tiny so synthesized-weight builds fit in KB; the shape relationships
    /// (GQA split, `d_model == num_heads * head_dim`, num_codebooks ==
    /// delay_pattern.len()) mirror the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            backbone: ZonosBackboneConfig {
                n_layer: 2,
                d_model: 16,
                d_intermediate: 32,
                num_heads: 4,
                num_heads_kv: 2,
                rotary_emb_dim: 4,
                rotary_emb_interleaved: true,
                causal: true,
                qkv_proj_bias: false,
                out_proj_bias: false,
                norm_epsilon: 1e-5,
                rms_norm: false,
            },
            conditioners: vec![
                ZonosConditioner {
                    name: "espeak".to_owned(),
                    kind: ZonosConditionerKind::EspeakPhoneme,
                },
                ZonosConditioner {
                    name: "speaker".to_owned(),
                    kind: ZonosConditionerKind::Speaker { cond_dim: 8 },
                },
                ZonosConditioner {
                    name: "fmax".to_owned(),
                    kind: ZonosConditionerKind::Fourier {
                        input_dim: 1,
                        min_val: 0.0,
                        max_val: 100.0,
                    },
                },
                ZonosConditioner {
                    name: "language_id".to_owned(),
                    kind: ZonosConditionerKind::Integer {
                        min_val: -1,
                        max_val: 3,
                    },
                },
            ],
            num_codebooks: 3,
            codebook_vocab: 12,
            head_vocab: 10,
            eos_token_id: 8,
            masked_token_id: 9,
            delay_pattern: vec![1, 2, 3],
            sample_rate: ZONOS_SAMPLE_RATE,
        }
    }

    /// Rejects `0`-placeholder / GQA-ill-formed configs before any forward
    /// runs (FR-EX-08 — a shape-only converter path fails loudly here,
    /// not deep inside a GEMM).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        if !self.backbone.is_well_formed() {
            return Err(VokraError::InvalidArgument(format!(
                "zonos config: backbone ill-formed (n_layer={}, d_model={}, \
                 d_intermediate={}, num_heads={}, num_heads_kv={}, \
                 rotary_emb_dim={}) — expected GQA well-formed \
                 (num_heads % num_heads_kv == 0, d_model % num_heads == 0, \
                 rotary_emb_dim == d_model / num_heads)",
                self.backbone.n_layer,
                self.backbone.d_model,
                self.backbone.d_intermediate,
                self.backbone.num_heads,
                self.backbone.num_heads_kv,
                self.backbone.rotary_emb_dim,
            )));
        }
        if self.backbone.head_dim() % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "zonos config: RoPE requires even head_dim (got {})",
                self.backbone.head_dim(),
            )));
        }
        if self.num_codebooks == 0 || self.codebook_vocab == 0 || self.head_vocab == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "zonos config: zero-size hparam (num_codebooks={}, \
                 codebook_vocab={}, head_vocab={})",
                self.num_codebooks, self.codebook_vocab, self.head_vocab,
            )));
        }
        if self.head_vocab > self.codebook_vocab {
            return Err(VokraError::InvalidArgument(format!(
                "zonos config: head_vocab={} > codebook_vocab={} — the head \
                 vocab is a subset of the embedding vocab (upstream drops the \
                 masked id from the emission surface)",
                self.head_vocab, self.codebook_vocab,
            )));
        }
        if self.delay_pattern.len() != self.num_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "zonos config: delay_pattern.len()={} != num_codebooks={}",
                self.delay_pattern.len(),
                self.num_codebooks,
            )));
        }
        // Special ids: eos must fit within `head_vocab` (it is emitted);
        // masked_token_id must fit within `codebook_vocab` but not within
        // `head_vocab` (upstream masks it out of the head).
        if (self.eos_token_id as usize) >= self.head_vocab {
            return Err(VokraError::InvalidArgument(format!(
                "zonos config: eos_token_id={} does not fit in head_vocab={}",
                self.eos_token_id, self.head_vocab,
            )));
        }
        if (self.masked_token_id as usize) >= self.codebook_vocab {
            return Err(VokraError::InvalidArgument(format!(
                "zonos config: masked_token_id={} does not fit in codebook_vocab={}",
                self.masked_token_id, self.codebook_vocab,
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Per-block backbone weights (pre-norm GQA attention + pre-norm SwiGLU
/// FFN with LayerNorm).
///
/// Field names track the upstream block shape: `norm_1_{w,b}` before
/// attention (LayerNorm has both γ and β because `rms_norm=false`),
/// `attn.qkv_proj` (fused), `attn.o_proj`, `norm_2_{w,b}` before FFN,
/// `mlp.fc1` / `mlp.fc2` for the SwiGLU stage.
///
/// The QKV projection is **fused** at the checkpoint layer: upstream
/// `_torch.py` packs `q + k + v` widths into a single Linear (matches the
/// mamba_ssm reference block). Total fused width =
/// `q_hidden + 2 * kv_hidden`; both projection biases are absent
/// (`qkv_proj_bias=false`, `out_proj_bias=false`).
#[derive(Debug, Clone)]
pub struct ZonosBlockWeights {
    /// Pre-attention LayerNorm γ, shape `[d_model]`.
    pub norm_1_w: Vec<f32>,
    /// Pre-attention LayerNorm β, shape `[d_model]`.
    pub norm_1_b: Vec<f32>,
    /// Fused QKV projection (transposed), shape
    /// `[d_model, q_hidden + 2 * kv_hidden]`.
    pub qkv_proj: Vec<f32>,
    /// Output projection (transposed), shape `[q_hidden, d_model]`.
    pub o_proj: Vec<f32>,
    /// Pre-FFN LayerNorm γ, shape `[d_model]`.
    pub norm_2_w: Vec<f32>,
    /// Pre-FFN LayerNorm β, shape `[d_model]`.
    pub norm_2_b: Vec<f32>,
    /// SwiGLU fc1 (transposed), shape `[d_model, 2 * d_intermediate]`.
    /// Chunked into `(y, gate)` at forward.
    pub mlp_fc1: Vec<f32>,
    /// SwiGLU fc2 (transposed), shape `[d_intermediate, d_model]`.
    pub mlp_fc2: Vec<f32>,
}

/// Zonos weight store: per-conditioner projection state (opaque),
/// per-codebook input embeddings, backbone blocks, and per-codebook logit
/// heads.
///
/// # Prefix-conditioner state
///
/// Upstream Zonos wraps each conditioner in its own `nn.Module` (espeak
/// tokenizer + text embedding, speaker linear projection, Fourier /
/// integer embedding tables, and a learned unconditional token). The
/// tensor manifest for these is not shape-derivable from the config
/// alone (the language-embedding table's row count is a runtime constant,
/// the Fourier feature widths depend on the packing convention). This
/// module reserves one opaque `Vec<f32>` per conditioner so real
/// conversion can populate them once the manifest is fetched — the
/// scaffold's shape gate only checks that the slot count matches
/// `cfg.conditioners.len()`.
///
/// # Real-checkpoint binding
///
/// [`Self::synthesized`] builds a deterministic fixture (SplitMix64 +
/// Xavier) against `config` so shape / dtype / size can be exercised
/// without the real HF checkpoint. Real-checkpoint binding is a follow-up
/// (T29-equivalent — tensor-name manifest fetch from the upstream
/// release).
#[derive(Debug, Clone)]
pub struct ZonosWeights {
    /// One opaque buffer per conditioner in `cfg.conditioners`. The
    /// scaffold leaves these `Vec::new()`; the real converter binds the
    /// projection tensors here. Preserving one slot per conditioner keeps
    /// the ordering handshake with `cfg.conditioners`.
    pub prefix_conditioner_state: Vec<Vec<f32>>,
    /// Per-codebook input embeddings, `num_codebooks` tables each of
    /// shape `[codebook_vocab, d_model]`.
    pub codebook_embeddings: Vec<Vec<f32>>,
    /// Backbone blocks in order.
    pub blocks: Vec<ZonosBlockWeights>,
    /// Per-codebook logit heads (transposed), `num_codebooks` tables each
    /// of shape `[d_model, head_vocab]`.
    pub logit_heads: Vec<Vec<f32>>,
    /// `true` when built by [`Self::synthesized`] — never a real upstream
    /// checkpoint. Real-checkpoint bindings set this to `false`.
    pub is_synthesized: bool,
}

impl ZonosWeights {
    /// Builds a deterministic synthesized fixture from `config` and `seed`.
    ///
    /// Draws are Xavier-uniform `± sqrt(6 / (fan_in + fan_out))` via a
    /// [`SplitMix64`] stream — reproducible, allocation-only, zero-dep.
    /// Every LayerNorm γ starts at `1.0`, every LayerNorm β at `0.0`.
    /// Prefix-conditioner slots start empty (`Vec::new()`) — the scaffold
    /// deliberately does not fabricate conditioner weights.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &ZonosConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        let bb = &config.backbone;
        let q_hidden = bb.q_hidden();
        let kv_hidden = bb.kv_hidden();
        let qkv_out = q_hidden + 2 * kv_hidden;
        let mlp_fc1_out = bb.mlp_fc1_out();

        let mut codebook_embeddings = Vec::with_capacity(config.num_codebooks);
        for _ in 0..config.num_codebooks {
            codebook_embeddings.push(xavier(
                &mut rng,
                config.codebook_vocab * bb.d_model,
                config.codebook_vocab,
                bb.d_model,
            ));
        }

        let mut blocks = Vec::with_capacity(bb.n_layer);
        for _ in 0..bb.n_layer {
            blocks.push(ZonosBlockWeights {
                norm_1_w: vec![1.0; bb.d_model],
                norm_1_b: vec![0.0; bb.d_model],
                qkv_proj: xavier(&mut rng, bb.d_model * qkv_out, bb.d_model, qkv_out),
                o_proj: xavier(&mut rng, q_hidden * bb.d_model, q_hidden, bb.d_model),
                norm_2_w: vec![1.0; bb.d_model],
                norm_2_b: vec![0.0; bb.d_model],
                mlp_fc1: xavier(&mut rng, bb.d_model * mlp_fc1_out, bb.d_model, mlp_fc1_out),
                mlp_fc2: xavier(
                    &mut rng,
                    bb.d_intermediate * bb.d_model,
                    bb.d_intermediate,
                    bb.d_model,
                ),
            });
        }

        let mut logit_heads = Vec::with_capacity(config.num_codebooks);
        for _ in 0..config.num_codebooks {
            logit_heads.push(xavier(
                &mut rng,
                bb.d_model * config.head_vocab,
                bb.d_model,
                config.head_vocab,
            ));
        }

        Ok(Self {
            prefix_conditioner_state: vec![Vec::new(); config.conditioners.len()],
            codebook_embeddings,
            blocks,
            logit_heads,
            is_synthesized: true,
        })
    }
}

/// Xavier-uniform draw of `count` `f32`s in `[-a, +a]` where
/// `a = sqrt(6 / (fan_in + fan_out))`. Deterministic under a fixed `rng`.
fn xavier(rng: &mut SplitMix64, count: usize, fan_in: usize, fan_out: usize) -> Vec<f32> {
    let a = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        // Map the top 24 bits of the u64 stream to a f32 in [0, 1).
        let raw = (rng.next_u64() >> 40) as u32;
        let u01 = (raw as f32) / ((1u32 << 24) as f32);
        out.push((u01 * 2.0 - 1.0) * a);
    }
    out
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Zonos TTS engine handle.
///
/// Carries the resolved config, weight store, and an optional DAC codec
/// bind ([`DacCodecGguf`] — MIT). [`Self::synthesize`] is the primary
/// text → PCM entry point; until real weights are bound (see the module
/// docstring) it returns [`VokraError::NotImplemented`] with a message
/// naming the blocker (FR-EX-08 — never a silent zero-fill fallback).
#[derive(Debug, Clone)]
pub struct ZonosTts {
    cfg: ZonosConfig,
    weights: ZonosWeights,
    /// Optional DAC codec bind. Injected via [`Self::with_dac`]; the real
    /// synth path consumes the DAC factorized RVQ decode + neural chain
    /// to produce 44.1 kHz PCM.
    dac: Option<DacCodecGguf>,
}

impl ZonosTts {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// weight-store shapes against `cfg` (block count, per-codebook
    /// counts, per-tensor sizes, conditioner slot count) so a mismatched
    /// pair fails loudly here rather than deep inside a forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape mismatch.
    pub fn new(cfg: ZonosConfig, weights: ZonosWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let bb = &cfg.backbone;

        // Prefix-conditioner slot count.
        if weights.prefix_conditioner_state.len() != cfg.conditioners.len() {
            return Err(VokraError::InvalidArgument(format!(
                "zonos weights: prefix_conditioner_state.len()={} != \
                 cfg.conditioners.len()={}",
                weights.prefix_conditioner_state.len(),
                cfg.conditioners.len(),
            )));
        }

        // Codebook embedding shapes.
        if weights.codebook_embeddings.len() != cfg.num_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "zonos weights: codebook_embeddings.len()={} != num_codebooks={}",
                weights.codebook_embeddings.len(),
                cfg.num_codebooks,
            )));
        }
        for (i, tbl) in weights.codebook_embeddings.iter().enumerate() {
            let expected = cfg.codebook_vocab * bb.d_model;
            if tbl.len() != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "zonos weights: codebook_embeddings[{i}].len()={} != {expected}",
                    tbl.len(),
                )));
            }
        }

        // Backbone block shapes.
        if weights.blocks.len() != bb.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "zonos weights: blocks.len()={} != backbone.n_layer={}",
                weights.blocks.len(),
                bb.n_layer,
            )));
        }
        let q_hidden = bb.q_hidden();
        let kv_hidden = bb.kv_hidden();
        let qkv_out = q_hidden + 2 * kv_hidden;
        let mlp_fc1_out = bb.mlp_fc1_out();
        for (i, blk) in weights.blocks.iter().enumerate() {
            for (name, len, expected) in [
                ("norm_1_w", blk.norm_1_w.len(), bb.d_model),
                ("norm_1_b", blk.norm_1_b.len(), bb.d_model),
                ("qkv_proj", blk.qkv_proj.len(), bb.d_model * qkv_out),
                ("o_proj", blk.o_proj.len(), q_hidden * bb.d_model),
                ("norm_2_w", blk.norm_2_w.len(), bb.d_model),
                ("norm_2_b", blk.norm_2_b.len(), bb.d_model),
                ("mlp_fc1", blk.mlp_fc1.len(), bb.d_model * mlp_fc1_out),
                ("mlp_fc2", blk.mlp_fc2.len(), bb.d_intermediate * bb.d_model),
            ] {
                if len != expected {
                    return Err(VokraError::InvalidArgument(format!(
                        "zonos weights: block {i} `{name}` len={len} != {expected}",
                    )));
                }
            }
        }

        // Head shapes.
        if weights.logit_heads.len() != cfg.num_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "zonos weights: logit_heads.len()={} != num_codebooks={}",
                weights.logit_heads.len(),
                cfg.num_codebooks,
            )));
        }
        for (i, tbl) in weights.logit_heads.iter().enumerate() {
            let expected = bb.d_model * cfg.head_vocab;
            if tbl.len() != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "zonos weights: logit_heads[{i}].len()={} != {expected}",
                    tbl.len(),
                )));
            }
        }
        Ok(Self {
            cfg,
            weights,
            dac: None,
        })
    }

    /// Injects a [`DacCodecGguf`] — the terminal factorized RVQ codes →
    /// PCM decoder.
    ///
    /// Zonos's decoder outputs `num_codebooks` (9) DAC codes per step;
    /// the DAC 44.1 kHz codec reduces them to a PCM waveform. Without a
    /// DAC bind [`Self::synthesize`] cannot honestly return audio
    /// (FR-EX-08).
    ///
    /// Cross-checks that the DAC codec has at least as many codebooks as
    /// Zonos emits channels — a mismatch would misroute channel indices
    /// at decode time — and that its sample rate matches
    /// [`ZonosConfig::sample_rate`].
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a codebook / sample-rate mismatch.
    pub fn with_dac(mut self, dac: DacCodecGguf) -> Result<Self> {
        if dac.attrs.n_codebooks < self.cfg.num_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "zonos with_dac: dac has {} codebooks but Zonos emits {} channels",
                dac.attrs.n_codebooks, self.cfg.num_codebooks,
            )));
        }
        if dac.sample_rate != self.cfg.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "zonos with_dac: dac sample_rate {} Hz != Zonos config sample_rate \
                 {} Hz (Zonos-v0.1 is bound to descript/dac_44khz)",
                dac.sample_rate, self.cfg.sample_rate,
            )));
        }
        self.dac = Some(dac);
        Ok(self)
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &ZonosConfig {
        &self.cfg
    }

    /// The bound DAC codec, if any.
    #[must_use]
    pub fn dac(&self) -> Option<&DacCodecGguf> {
        self.dac.as_ref()
    }

    /// True iff the weight store was built by [`ZonosWeights::synthesized`]
    /// (never a real upstream checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Synthesizes PCM given a phoneme-id sequence.
    ///
    /// `phoneme_ids` is an eSpeak-NG phoneme id sequence (upstream Zonos
    /// consumes the same ids the `EspeakPhonemeConditioner` would). No
    /// specific vocab range is enforced by this scaffold — the real
    /// conversion binds the phoneme-table row count from the tensor
    /// manifest and enforces the range then. What this method **does**
    /// enforce today is FR-EX-08: never a silent zero-fill / silence
    /// fallback.
    ///
    /// This is the primary text → PCM entry point. **Real weights required**:
    /// synthesized-weight builds cannot produce meaningful audio (they'd
    /// be noise or a hallucinated "silence"), so this returns
    /// [`VokraError::NotImplemented`] naming the blocker. Callers verify
    /// the shape flow through [`ZonosTts::new`] +
    /// [`ZonosWeights::synthesized`] today; a follow-up wave binds the
    /// real HF checkpoint tensor names and wires the forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on empty `phoneme_ids` or a
    ///   negative id (upstream eSpeak ids are always ≥ 0).
    /// - [`VokraError::NotImplemented`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn synthesize(&self, phoneme_ids: &[i64]) -> Result<Vec<f32>> {
        if phoneme_ids.is_empty() {
            return Err(VokraError::InvalidArgument(
                "zonos synthesize: phoneme_ids is empty".to_owned(),
            ));
        }
        for (i, id) in phoneme_ids.iter().enumerate() {
            if *id < 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "zonos synthesize: phoneme_ids[{i}]={id} < 0 \
                     (eSpeak-NG phoneme ids are non-negative)",
                )));
            }
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "zonos synthesize: this engine holds synthesized weights (deterministic \
                 fixture from ZonosWeights::synthesized) — synthesized-weight PCM would \
                 be noise, not speech. Bind real Zonos-v0.1-transformer weights \
                 (Apache 2.0, Zyphra/Zonos-v0.1-transformer) before invoking synthesize. \
                 The shape flow (config validation, weight-store construction) is \
                 exercised through ZonosTts::new; the real-checkpoint tensor-name \
                 manifest lands in a follow-up wave (T29-equivalent).",
            ));
        }
        if self.dac.is_none() {
            return Err(VokraError::NotImplemented(
                "zonos synthesize: no DAC codec has been bound — call \
                 `.with_dac(DacCodecGguf::from_gguf(&dac_gguf)?)?` first. Zonos's \
                 decoder emits 9 DAC codebook channels per step which the DAC 44.1 kHz \
                 codec reduces to PCM; without it there is nothing honest to return \
                 (FR-EX-08).",
            ));
        }
        Err(VokraError::NotImplemented(
            "zonos synthesize: real weights are bound and a DAC codec is present, but \
             the prefix-conditioner projection tensors and the delayed-AR decoder \
             forward path have not landed yet. Follow-up wave: transcribe the upstream \
             tensor manifest (backbone.blocks.*, embeddings.i.*, heads.i.*, \
             prefix_conditioner.conditioners.*) and wire the pre-norm GQA + SwiGLU \
             forward through the `Compute` seam (CosyVoice2 T07/T08 pattern), plus the \
             prefix-conditioner projection stack (espeak phoneme lookup + speaker \
             linear + Fourier / integer feature embeddings, each with a learned \
             unconditional token).",
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Every hparam matches the primary source
    /// (`huggingface.co/Zyphra/Zonos-v0.1-transformer/raw/main/config.json`)
    /// verbatim.
    #[test]
    fn zonos_v0_1_transformer_matches_primary_source_config_json() {
        let c = ZonosConfig::zonos_v0_1_transformer();
        // config.backbone
        assert_eq!(c.backbone.n_layer, 26);
        assert_eq!(c.backbone.d_model, 2048);
        assert_eq!(c.backbone.d_intermediate, 8192);
        assert_eq!(c.backbone.num_heads, 16);
        assert_eq!(c.backbone.num_heads_kv, 4);
        assert_eq!(c.backbone.rotary_emb_dim, 128);
        assert!(c.backbone.rotary_emb_interleaved);
        assert!(c.backbone.causal);
        assert!(!c.backbone.qkv_proj_bias);
        assert!(!c.backbone.out_proj_bias);
        assert_eq!(c.backbone.norm_epsilon, 1e-5);
        assert!(
            !c.backbone.rms_norm,
            "Zonos-v0.1 uses LayerNorm (rms_norm=false in config.json)"
        );

        // Derived shape helpers.
        assert_eq!(c.backbone.head_dim(), 128);
        assert_eq!(c.backbone.q_hidden(), 2048);
        assert_eq!(c.backbone.kv_hidden(), 512);
        assert_eq!(c.backbone.mlp_fc1_out(), 16_384);

        // config.prefix_conditioner.conditioners (in order).
        assert_eq!(c.conditioners.len(), 7);
        let names: Vec<&str> = c.conditioners.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "espeak",
                "speaker",
                "emotion",
                "fmax",
                "pitch_std",
                "speaking_rate",
                "language_id"
            ]
        );
        // Individual conditioner types.
        assert!(matches!(
            c.conditioners[0].kind,
            ZonosConditionerKind::EspeakPhoneme
        ));
        assert_eq!(
            c.conditioners[1].kind,
            ZonosConditionerKind::Speaker { cond_dim: 128 }
        );
        assert_eq!(
            c.conditioners[2].kind,
            ZonosConditionerKind::Fourier {
                input_dim: 8,
                min_val: 0.0,
                max_val: 0.0
            }
        );
        assert_eq!(
            c.conditioners[3].kind,
            ZonosConditionerKind::Fourier {
                input_dim: 1,
                min_val: 0.0,
                max_val: 24_000.0
            }
        );
        assert_eq!(
            c.conditioners[4].kind,
            ZonosConditionerKind::Fourier {
                input_dim: 1,
                min_val: 0.0,
                max_val: 400.0
            }
        );
        assert_eq!(
            c.conditioners[5].kind,
            ZonosConditionerKind::Fourier {
                input_dim: 1,
                min_val: 0.0,
                max_val: 40.0
            }
        );
        assert_eq!(
            c.conditioners[6].kind,
            ZonosConditionerKind::Integer {
                min_val: -1,
                max_val: 126
            }
        );

        // Codebook / head / special ids.
        assert_eq!(c.num_codebooks, 9);
        assert_eq!(c.codebook_vocab, 1026);
        assert_eq!(c.head_vocab, 1025);
        assert_eq!(c.eos_token_id, 1024);
        assert_eq!(c.masked_token_id, 1025);
        // Delay pattern: [1, 2, ..., 9] per zonos/codebook_pattern.py.
        assert_eq!(c.delay_pattern, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        // DAC 44.1 kHz inheritance.
        assert_eq!(c.sample_rate, 44_100);
        // Everything above adds up to a well-formed config.
        c.validate_for_forward()
            .expect("zonos-v0.1-transformer is well-formed");
    }

    #[test]
    fn tiny_config_is_well_formed() {
        ZonosConfig::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny config is well-formed");
    }

    #[test]
    fn config_gqa_ill_formed_is_rejected() {
        let mut c = ZonosConfig::tiny_for_tests();
        c.backbone.num_heads_kv = 3; // 4 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_odd_head_dim_is_rejected() {
        let mut c = ZonosConfig::tiny_for_tests();
        // Deliberate: rotary_emb_dim = head_dim = 5 (odd, RoPE fails)
        c.backbone.num_heads = 2;
        c.backbone.d_model = 10;
        c.backbone.rotary_emb_dim = 5;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_delay_pattern_length_must_equal_num_codebooks() {
        let mut c = ZonosConfig::tiny_for_tests();
        c.delay_pattern.push(4);
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_special_ids_are_range_checked() {
        // eos_token_id must fit within head_vocab.
        let mut c = ZonosConfig::tiny_for_tests();
        c.eos_token_id = c.head_vocab as u32;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));

        // masked_token_id must fit within codebook_vocab.
        let mut c = ZonosConfig::tiny_for_tests();
        c.masked_token_id = c.codebook_vocab as u32;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_head_vocab_may_not_exceed_codebook_vocab() {
        let mut c = ZonosConfig::tiny_for_tests();
        c.head_vocab = c.codebook_vocab + 1;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        let c = ZonosConfig::tiny_for_tests();
        let w1 = ZonosWeights::synthesized(&c, 0x42).expect("build 1");
        let w2 = ZonosWeights::synthesized(&c, 0x42).expect("build 2");
        // Determinism.
        assert_eq!(w1.codebook_embeddings[0], w2.codebook_embeddings[0]);
        assert_eq!(
            w1.blocks[0].qkv_proj, w2.blocks[0].qkv_proj,
            "same seed → same weights"
        );
        assert!(w1.is_synthesized);
        // Shape flow.
        assert_eq!(w1.blocks.len(), c.backbone.n_layer);
        assert_eq!(w1.codebook_embeddings.len(), c.num_codebooks);
        assert_eq!(w1.logit_heads.len(), c.num_codebooks);
        // Prefix conditioner slots exist but are empty until real weights bind.
        assert_eq!(w1.prefix_conditioner_state.len(), c.conditioners.len());
        for slot in &w1.prefix_conditioner_state {
            assert!(
                slot.is_empty(),
                "synthesized() must leave prefix_conditioner_state empty"
            );
        }
        // Fused QKV width matches q_hidden + 2*kv_hidden (GQA fused proj).
        assert_eq!(
            w1.blocks[0].qkv_proj.len(),
            c.backbone.d_model * (c.backbone.q_hidden() + 2 * c.backbone.kv_hidden())
        );
        // Packed SwiGLU fc1 = d_model * 2 * d_intermediate.
        assert_eq!(
            w1.blocks[0].mlp_fc1.len(),
            c.backbone.d_model * c.backbone.mlp_fc1_out()
        );
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let c = ZonosConfig::tiny_for_tests();
        let w_a = ZonosWeights::synthesized(&c, 1).expect("build a");
        let w_b = ZonosWeights::synthesized(&c, 2).expect("build b");
        // Two distinct seeds must produce different Xavier draws.
        assert_ne!(w_a.codebook_embeddings[0], w_b.codebook_embeddings[0]);
    }

    #[test]
    fn zonos_tts_new_accepts_matching_config_and_weights() {
        let c = ZonosConfig::tiny_for_tests();
        let w = ZonosWeights::synthesized(&c, 7).expect("weights");
        let tts = ZonosTts::new(c.clone(), w).expect("zonos tts");
        assert_eq!(tts.config().backbone.d_model, c.backbone.d_model);
        assert!(tts.is_synthesized());
        assert!(tts.dac().is_none());
    }

    #[test]
    fn zonos_tts_new_rejects_block_count_mismatch() {
        let c = ZonosConfig::tiny_for_tests();
        let mut w = ZonosWeights::synthesized(&c, 7).expect("weights");
        w.blocks.pop();
        assert!(matches!(
            ZonosTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn zonos_tts_new_rejects_tensor_size_mismatch() {
        let c = ZonosConfig::tiny_for_tests();
        let mut w = ZonosWeights::synthesized(&c, 7).expect("weights");
        w.blocks[0].qkv_proj.pop();
        assert!(matches!(
            ZonosTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn zonos_tts_new_rejects_conditioner_slot_mismatch() {
        let c = ZonosConfig::tiny_for_tests();
        let mut w = ZonosWeights::synthesized(&c, 7).expect("weights");
        w.prefix_conditioner_state.push(Vec::new());
        assert!(matches!(
            ZonosTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesize_rejects_empty_ids() {
        let c = ZonosConfig::tiny_for_tests();
        let w = ZonosWeights::synthesized(&c, 7).expect("weights");
        let tts = ZonosTts::new(c, w).expect("zonos tts");
        assert!(matches!(
            tts.synthesize(&[]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesize_rejects_negative_id() {
        let c = ZonosConfig::tiny_for_tests();
        let w = ZonosWeights::synthesized(&c, 7).expect("weights");
        let tts = ZonosTts::new(c, w).expect("zonos tts");
        assert!(matches!(
            tts.synthesize(&[-1]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// The primary NotImplemented path names the synthesized-weight
    /// blocker (FR-EX-08 — never a silent zero-fill).
    #[test]
    fn synthesize_on_synthesized_weights_is_loud_not_implemented() {
        let c = ZonosConfig::tiny_for_tests();
        let w = ZonosWeights::synthesized(&c, 7).expect("weights");
        let tts = ZonosTts::new(c, w).expect("zonos tts");
        let err = tts.synthesize(&[0, 1, 2]).unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("synthesized"),
                    "message must name synthesized-weight blocker: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn expected_arch_is_zonos() {
        assert_eq!(EXPECTED_ARCH, "zonos");
    }

    #[test]
    fn zonos_num_codebooks_matches_dia_shape() {
        // Zonos-v0.1 (num_codebooks=9) shares the DAC codebook shape with
        // Dia; the same `DacCodecGguf` binder handles both models. The
        // ADR-level statement that these two models can share the codec
        // slot is enforced by the numeric equality here.
        assert_eq!(ZONOS_NUM_CODEBOOKS, 9);
    }
}
