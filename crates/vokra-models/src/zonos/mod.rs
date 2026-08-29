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
//!   (`uncond_type=learned`, except required eSpeak input). The typed
//!   projection weights are bound only after the fixed 246-name/shape
//!   manifest is authenticated.
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
//! [`crate::dac::Dac`] is the complete token-to-PCM binder — the codec GGUF is what
//! carries the sample rate (`vokra.dac.sample_rate`), and the runtime
//! cross-checks it against [`ZonosConfig::sample_rate`] in
//! [`ZonosTts::with_dac`].
//!
//! # Inspection/runtime boundary
//!
//! - [`ZonosConfig`] and synthesized [`ZonosWeights`] remain explicit
//!   shape fixtures for diagnostics only. A production weight store can only
//!   be produced by [`ZonosCheckpoint::load_weights`] after strict binding.
//! - [`ZonosWeights`] — a backbone + codebook weight store with a
//!   deterministic [`ZonosWeights::synthesized`] fixture (SplitMix64 +
//!   Xavier) so shape / dtype / size flow can be exercised without the
//!   real HF checkpoint. The typed prefix-conditioner tensors are populated
//!   only by [`ZonosCheckpoint::load_weights`] after strict binding.
//! - [`ZonosTts`] keeps the complete DAC slot typed as [`crate::dac::Dac`];
//!   raw text remains outside the runtime because eSpeak is an offline
//!   preparation boundary. The typed packet path is the only native
//!   conditioning entry point.
//!
//! The transformer and conditioning path are source-shaped and Compute
//! dispatched, while end-to-end PCM parity still requires the VAST/Apple
//! evidence wave. No numerical gate is claimed before that evidence exists.
//!
//! # No ONNX (permanent)
//!
//! Zonos ships as safetensors / a Python pipeline; the pipeline is
//! re-implemented natively (whisper.cpp 型, CLAUDE.md 設計判断 4). This
//! module never touches ONNX.

use vokra_core::gguf::GgufFile;
use vokra_core::rng::SplitMix64;
use vokra_core::{BackendKind, Result, VokraError};

mod bound;
mod conditioning;
mod transformer;
pub use bound::{ZonosCheckpoint, ZonosSpeakerProjection};
pub use conditioning::ZonosPrefixConditionerWeights;

#[cfg(test)]
use crate::codec::DacCodecGguf;
use crate::dac::Dac;

/// `vokra.model.arch` a Zonos GGUF must carry. Written by
/// `vokra-convert::models::zonos::ARCH`; the compliance registry
/// (`vokra_core::compliance`) knows `zonos` / `zonos-v0.1` as
/// [`LicenseClass::Permissive`](vokra_core::LicenseClass::Permissive)
/// (Apache 2.0 code + weight). This architecture tag alone does not authorize
/// a runtime: the converter and strict checkpoint binder remain inspection-only
/// until the fixed artifact evidence is reviewed.
pub const EXPECTED_ARCH: &str = "zonos";

/// PCM sample rate Zonos emits. Not written in the upstream `config.json`;
/// inherited from **DAC 44.1 kHz** (`descript/dac_44khz` loaded upstream
/// in `zonos/autoencoder.py`).
pub const ZONOS_SAMPLE_RATE: u32 = 44_100;

/// Number of DAC codebook channels the Zonos-v0.1 decoder emits per step.
/// Wired to `DACAutoencoder.num_codebooks` (upstream constructor) — the
/// same 9 codebook channels as Dia, but Zonos must bind the complete
/// [`crate::dac::Dac`] rather than its lower-level `DacCodecGguf` container.
pub const ZONOS_NUM_CODEBOOKS: usize = 9;
const ZONOS_HOT_OPS: &[crate::compute::HotOp] = &[
    crate::compute::HotOp::Gemm,
    crate::compute::HotOp::Softmax,
    crate::compute::HotOp::LayerNorm,
    crate::compute::HotOp::Silu,
];
const UNKNOWN_GENERATION_TOKEN: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Prefix conditioner descriptor
// ---------------------------------------------------------------------------

/// One typed conditioner in the prefix stack (primary source:
/// `config.prefix_conditioner.conditioners[i]`).
///
/// Zonos-v0.1 prepends 7 typed conditioning tokens before the codebook
/// token stream. Each entry has a **type** (which decides the projection
/// shape at real-conversion time) and, for the numeric conditioners, a
/// **numeric domain** (min/max). Projection weights are held in
/// [`ZonosPrefixConditionerWeights`] only after strict manifest binding.
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
        /// `min_val` of the numeric domain. Emotion uses the source default
        /// `0.0` and is normalized to a probability vector before encoding.
        min_val: f32,
        /// `max_val` of the numeric domain. Emotion uses the source default
        /// range `[0.0, 1.0]` before its normalized control vector is encoded.
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
    /// same [`crate::dac::Dac`] binder works).
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
                        max_val: 1.0,
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
    #[allow(dead_code)] // staged until the authenticated Zonos runtime binder is wired
    pub(crate) fn tiny_for_tests() -> Self {
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

    /// Applies the official multi-codebook delay layout.  Each codebook is
    /// shifted by its configured delay and the exposed leading/trailing
    /// positions are filled with the masked token; callers may feed the
    /// result directly to a causal transformer state.
    pub fn apply_delay_pattern(&self, codes: &[Vec<u32>]) -> Result<Vec<Vec<u32>>> {
        if codes.len() != self.num_codebooks || codes.iter().any(|row| row.is_empty()) {
            return Err(VokraError::InvalidArgument(
                "zonos delay pattern requires one non-empty row per codebook".to_owned(),
            ));
        }
        let frames = codes[0].len();
        if codes.iter().any(|row| row.len() != frames) {
            return Err(VokraError::InvalidArgument(
                "zonos delay pattern rows must have equal length".to_owned(),
            ));
        }
        for row in codes {
            if row
                .iter()
                .any(|&token| token as usize >= self.codebook_vocab)
            {
                return Err(VokraError::InvalidArgument(
                    "zonos delay pattern contains an out-of-range code".to_owned(),
                ));
            }
        }
        let max_delay = self.delay_pattern.iter().copied().max().unwrap_or(0);
        let mut delayed = vec![vec![self.masked_token_id; frames + max_delay]; self.num_codebooks];
        for codebook in 0..self.num_codebooks {
            let delay = self.delay_pattern[codebook];
            for (frame, &token) in codes[codebook].iter().enumerate() {
                delayed[codebook][frame + delay] = token;
            }
        }
        Ok(delayed)
    }

    /// Reverts a delayed codebook matrix after terminal drain.  Only the
    /// positions corresponding to original frames are exposed; leading and
    /// trailing masked positions are never sent to the DAC.
    pub fn revert_delay_pattern(
        &self,
        delayed: &[Vec<u32>],
        original_frames: usize,
    ) -> Result<Vec<Vec<u32>>> {
        if delayed.len() != self.num_codebooks || original_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "zonos revert delay pattern shape is invalid".to_owned(),
            ));
        }
        let max_delay = self.delay_pattern.iter().copied().max().unwrap_or(0);
        let expected_len = original_frames.checked_add(max_delay).ok_or_else(|| {
            VokraError::InvalidArgument("zonos delay pattern length overflow".to_owned())
        })?;
        if delayed.iter().any(|row| row.len() != expected_len) {
            return Err(VokraError::InvalidArgument(
                "zonos delayed rows do not match the terminal drain extent".to_owned(),
            ));
        }
        let mut result = vec![vec![0; original_frames]; self.num_codebooks];
        for codebook in 0..self.num_codebooks {
            let delay = self.delay_pattern[codebook];
            for frame in 0..original_frames {
                let token = delayed[codebook][frame + delay];
                if token as usize >= self.codebook_vocab || token >= self.eos_token_id {
                    return Err(VokraError::InvalidArgument(
                        "zonos strict revert requires ordinary DAC code values".to_owned(),
                    ));
                }
                result[codebook][frame] = token;
            }
        }
        Ok(result)
    }

    /// Validates one greedy AR step and enforces CB0-only EOS.  EOS from any
    /// non-primary codebook is masked, matching the upstream delayed
    /// generation contract rather than silently terminating a partial frame.
    pub fn greedy_step(&self, logits: &[Vec<f32>]) -> Result<Vec<u32>> {
        if logits.len() != self.num_codebooks
            || logits.iter().any(|row| row.len() != self.head_vocab)
        {
            return Err(VokraError::InvalidArgument(
                "zonos greedy step logits shape mismatch".to_owned(),
            ));
        }
        let mut result = Vec::with_capacity(self.num_codebooks);
        for (codebook, row) in logits.iter().enumerate() {
            if row.iter().any(|value| !value.is_finite()) {
                return Err(VokraError::InvalidArgument(
                    "zonos greedy step contains non-finite logits".to_owned(),
                ));
            }
            let mut best = None;
            for (token, &value) in row.iter().enumerate() {
                if codebook != 0 && token == self.eos_token_id as usize {
                    continue;
                }
                if best.is_none_or(|(_, score)| value > score) {
                    best = Some((token, value));
                }
            }
            let Some((token, _)) = best else {
                return Err(VokraError::InvalidArgument(
                    "zonos greedy step has no legal token".to_owned(),
                ));
            };
            result.push(token as u32);
        }
        Ok(result)
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

/// Zonos weight store: typed prefix-conditioner projections, per-codebook
/// input embeddings, backbone blocks, and per-codebook logit heads.
///
/// # Prefix-conditioner state
///
/// Upstream Zonos wraps each conditioner in its own `nn.Module` (eSpeak
/// tokenizer + text embedding, speaker linear projection, Fourier /
/// integer embedding tables, and learned unconditional tokens). The
/// production fields are the typed [`ZonosPrefixConditionerWeights`], which
/// are constructed only by strict manifest binding. The compatibility slots
/// below remain empty for synthetic fixtures and are not consumed by native
/// conditioning.
///
/// # Real-checkpoint binding
///
/// [`Self::synthesized`] builds a deterministic fixture (SplitMix64 +
/// Xavier) against `config` so shape / dtype / size can be exercised
/// without the real HF checkpoint. Real checkpoint binding is exposed by
/// [`ZonosCheckpoint::load_weights`] after strict name/shape validation.
#[derive(Debug, Clone)]
pub struct ZonosWeights {
    /// Compatibility slots preserving conditioner ordering. Production
    /// forward uses [`Self::prefix_conditioner`] and rejects empty binding.
    pub prefix_conditioner_state: Vec<Vec<f32>>,
    /// Fully typed native prefix conditioner. `None` is reserved for the
    /// deterministic synthetic fixture and cannot authorize inference.
    pub prefix_conditioner: Option<ZonosPrefixConditionerWeights>,
    /// Per-codebook input embeddings, `num_codebooks` tables each of
    /// shape `[codebook_vocab, d_model]`.
    pub codebook_embeddings: Vec<Vec<f32>>,
    /// Backbone blocks in order.
    pub blocks: Vec<ZonosBlockWeights>,
    /// Per-codebook logit heads (transposed), `num_codebooks` tables each
    /// of shape `[d_model, head_vocab]`.
    pub logit_heads: Vec<Vec<f32>>,
    /// Final pre-head LayerNorm γ (`backbone.norm_f.weight`).
    pub norm_f_w: Vec<f32>,
    /// Final pre-head LayerNorm β (`backbone.norm_f.bias`).
    pub norm_f_b: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real upstream
    /// checkpoint. Real-checkpoint bindings set this to `false`.
    is_synthesized: bool,
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
    #[allow(dead_code)] // staged until the authenticated Zonos runtime binder is wired
    pub(crate) fn synthesized(config: &ZonosConfig, seed: u64) -> Result<Self> {
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
            prefix_conditioner: None,
            codebook_embeddings,
            blocks,
            logit_heads,
            norm_f_w: vec![1.0; bb.d_model],
            norm_f_b: vec![0.0; bb.d_model],
            is_synthesized: true,
        })
    }

    /// Constructs a production weight store only for the strict checkpoint
    /// binder. Keeping the provenance bit private prevents external callers
    /// from relabelling arbitrary tensors as authenticated weights.
    pub(crate) fn from_bound_parts(
        prefix_conditioner_state: Vec<Vec<f32>>,
        prefix_conditioner: ZonosPrefixConditionerWeights,
        codebook_embeddings: Vec<Vec<f32>>,
        blocks: Vec<ZonosBlockWeights>,
        logit_heads: Vec<Vec<f32>>,
        norm_f_w: Vec<f32>,
        norm_f_b: Vec<f32>,
    ) -> Self {
        Self {
            prefix_conditioner_state,
            prefix_conditioner: Some(prefix_conditioner),
            codebook_embeddings,
            blocks,
            logit_heads,
            norm_f_w,
            norm_f_b,
            is_synthesized: false,
        }
    }
}

/// Versioned, offline-prepared Zonos conditioning packet.
///
/// The runtime deliberately accepts phoneme IDs, not text. eSpeak-NG is a
/// GPL application dependency and is therefore run by an offline preparer;
/// the packet carries its resulting representation and raw controls. Legacy
/// projected-prefix payload fields remain covered by the packet digest for
/// format compatibility but are never exposed to or consumed by production
/// conditioning. The 32-byte digest is computed over packet contents and
/// compared with the digest authenticated by the model manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct ZonosConditioningPacket {
    /// Packet format version (currently `1`).
    pub version: u32,
    /// eSpeak phoneme IDs produced by the offline preparer.
    pub phoneme_ids: Vec<u32>,
    /// Speaker embedding, exactly 128 finite values.
    pub speaker: Vec<f32>,
    /// Emotion controls, exactly eight finite values.
    pub emotion: Vec<f32>,
    /// Maximum frequency control in Hz.
    pub fmax: f32,
    /// Pitch standard deviation control.
    pub pitch_std: f32,
    /// Speaking-rate control.
    pub speaking_rate: f32,
    /// Upstream integer language ID (`-1` means unset).
    pub language_id: i32,
    /// Prompt audio codes in codebook-major order. These are the delayed
    /// prefill codes; an empty prompt is valid for text-only conditioning.
    pub prompt_codes: Vec<Vec<u32>>,
    digest: [u8; 32],
}

impl ZonosConditioningPacket {
    const MAGIC: &'static [u8; 8] = b"ZONOSCP1";
    const VERSION: u32 = 1;
    const MAX_PHONEMES: usize = 1 << 20;
    const MAX_PREFIX_VALUES: usize = 1 << 24;

    /// Parses the strict binary packet and authenticates its precomputed
    /// digest against the checkpoint manifest's expected digest.
    pub fn parse(bytes: &[u8], expected_digest: [u8; 32], d_model: usize) -> Result<Self> {
        let mut cursor = 0usize;
        let take = |cursor: &mut usize, count: usize| -> Result<&[u8]> {
            let end = cursor.checked_add(count).ok_or_else(|| {
                VokraError::InvalidArgument("zonos conditioning packet overflow".to_owned())
            })?;
            let slice = bytes.get(*cursor..end).ok_or_else(|| {
                VokraError::InvalidArgument("zonos conditioning packet truncated".to_owned())
            })?;
            *cursor = end;
            Ok(slice)
        };
        let magic = take(&mut cursor, Self::MAGIC.len())?;
        if magic != Self::MAGIC {
            return Err(VokraError::InvalidArgument(
                "zonos conditioning packet magic mismatch".to_owned(),
            ));
        }
        let u32_at = |cursor: &mut usize| -> Result<u32> {
            Ok(u32::from_le_bytes(take(cursor, 4)?.try_into().unwrap()))
        };
        let f32_at = |cursor: &mut usize| -> Result<f32> {
            Ok(f32::from_le_bytes(take(cursor, 4)?.try_into().unwrap()))
        };
        let version = u32_at(&mut cursor)?;
        if version != Self::VERSION {
            return Err(VokraError::InvalidArgument(format!(
                "zonos conditioning packet version {version} is unsupported"
            )));
        }
        let phoneme_count = u32_at(&mut cursor)? as usize;
        if phoneme_count == 0 || phoneme_count > Self::MAX_PHONEMES {
            return Err(VokraError::InvalidArgument(
                "zonos conditioning packet phoneme count is out of bounds".to_owned(),
            ));
        }
        let mut speaker = Vec::with_capacity(128);
        for _ in 0..128 {
            speaker.push(f32_at(&mut cursor)?);
        }
        let mut emotion = Vec::with_capacity(8);
        for _ in 0..8 {
            emotion.push(f32_at(&mut cursor)?);
        }
        let fmax = f32_at(&mut cursor)?;
        let pitch_std = f32_at(&mut cursor)?;
        let speaking_rate = f32_at(&mut cursor)?;
        let language_id = i32::from_le_bytes(take(&mut cursor, 4)?.try_into().unwrap());
        let codebook_count = u32_at(&mut cursor)? as usize;
        let prompt_frames = u32_at(&mut cursor)? as usize;
        if codebook_count == 0 || codebook_count > 32 || prompt_frames > Self::MAX_PHONEMES {
            return Err(VokraError::InvalidArgument(
                "zonos conditioning prompt-code shape is invalid".to_owned(),
            ));
        }
        let conditional_count = u32_at(&mut cursor)? as usize;
        let unconditional_count = u32_at(&mut cursor)? as usize;
        if d_model == 0
            || conditional_count == 0
            || conditional_count != unconditional_count
            || conditional_count > Self::MAX_PREFIX_VALUES
            || conditional_count % d_model != 0
        {
            return Err(VokraError::InvalidArgument(
                "zonos conditioning prefix shapes are invalid".to_owned(),
            ));
        }
        let digest_offset = cursor;
        let digest: [u8; 32] = take(&mut cursor, 32)?.try_into().unwrap();
        let computed_digest = packet_digest(bytes, digest_offset);
        if digest != computed_digest {
            return Err(VokraError::InvalidArgument(
                "zonos conditioning packet content digest mismatch".to_owned(),
            ));
        }
        if digest != expected_digest {
            return Err(VokraError::InvalidArgument(
                "zonos conditioning packet digest does not match checkpoint manifest".to_owned(),
            ));
        }
        let mut phoneme_ids = Vec::with_capacity(phoneme_count);
        for _ in 0..phoneme_count {
            phoneme_ids.push(u32_at(&mut cursor)?);
        }
        if phoneme_ids.len() < 2
            || phoneme_ids.first() != Some(&2)
            || phoneme_ids.last() != Some(&3)
            || phoneme_ids
                .get(1..phoneme_ids.len() - 1)
                .is_some_and(|interior| {
                    interior
                        .iter()
                        .any(|&id| id == 0 || id == 2 || id == 3 || id >= 189)
                })
        {
            return Err(VokraError::InvalidArgument(
                "zonos phoneme symbols must use PAD=0/UNK=1/BOS=2/EOS=3 framing".to_owned(),
            ));
        }
        let mut prompt_codes = vec![Vec::with_capacity(prompt_frames); codebook_count];
        for row in &mut prompt_codes {
            for _ in 0..prompt_frames {
                let code = u32_at(&mut cursor)?;
                if code as usize >= 1026 {
                    return Err(VokraError::InvalidArgument(
                        "zonos conditioning prompt code is out of range".to_owned(),
                    ));
                }
                row.push(code);
            }
        }
        let read_values = |cursor: &mut usize, count: usize| -> Result<Vec<f32>> {
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(f32_at(cursor)?);
            }
            Ok(values)
        };
        let conditional_prefix = read_values(&mut cursor, conditional_count)?;
        let unconditional_prefix = read_values(&mut cursor, unconditional_count)?;
        if cursor != bytes.len()
            || speaker
                .iter()
                .chain(&emotion)
                .chain(std::slice::from_ref(&fmax))
                .chain(std::slice::from_ref(&pitch_std))
                .chain(std::slice::from_ref(&speaking_rate))
                .chain(&conditional_prefix)
                .chain(&unconditional_prefix)
                .any(|value| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(
                "zonos conditioning packet has trailing bytes or non-finite values".to_owned(),
            ));
        }
        if !(0.0..=24_000.0).contains(&fmax)
            || !(0.0..=400.0).contains(&pitch_std)
            || !(0.0..=40.0).contains(&speaking_rate)
            || !(-1..=126).contains(&language_id)
            || emotion.iter().any(|value| !(0.0..=1.0).contains(value))
            || (emotion.iter().sum::<f32>() - 1.0).abs() > 1.0e-4
        {
            return Err(VokraError::InvalidArgument(
                "zonos conditioning control is outside the audited domain".to_owned(),
            ));
        }
        Ok(Self {
            version,
            phoneme_ids,
            speaker,
            emotion,
            fmax,
            pitch_std,
            speaking_rate,
            language_id,
            prompt_codes,
            digest,
        })
    }

    /// Returns the prehashed packet identity checked during parsing.
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// Xavier-uniform draw of `count` `f32`s in `[-a, +a]` where
/// `a = sqrt(6 / (fan_in + fan_out))`. Deterministic under a fixed `rng`.
#[allow(dead_code)] // staged until the authenticated Zonos runtime binder is wired
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

/// Source-compatible sampling controls for Zonos codebook logits.
#[derive(Debug, Clone, PartialEq)]
pub struct ZonosSamplingParams {
    /// Temperature; exactly zero selects deterministic argmax.
    pub temperature: f32,
    /// Repetition penalty applied to the recent token window.
    pub repetition_penalty: f32,
    /// Number of previous tokens per codebook considered for repetition.
    pub repetition_window: usize,
    /// Nucleus probability cutoff.
    pub top_p: Option<f32>,
    /// Top-k cutoff.
    pub top_k: Option<usize>,
    /// Minimum probability relative to the row maximum.
    pub min_p: Option<f32>,
    /// Source `apply_unified` linear coefficient. Applied only for stochastic
    /// sampling, after temperature and before nucleus/top-k/min-p filters.
    pub unified_linear: f32,
    /// Source `apply_unified` entropy confidence coefficient.
    pub unified_confidence: f32,
    /// Source `apply_unified` quadratic log-probability coefficient.
    pub unified_quadratic: f32,
}

impl Default for ZonosSamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            repetition_penalty: 3.0,
            repetition_window: 2,
            top_p: None,
            top_k: None,
            min_p: Some(0.1),
            unified_linear: 0.0,
            unified_confidence: 0.0,
            unified_quadratic: 0.0,
        }
    }
}

impl ZonosSamplingParams {
    /// Deterministic source-shaped greedy controls.
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            repetition_penalty: 1.0,
            repetition_window: 0,
            min_p: None,
            ..Self::default()
        }
    }

    fn validate(&self, head_vocab: usize) -> Result<()> {
        if !self.temperature.is_finite()
            || self.temperature < 0.0
            || !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.0
            || (self.repetition_window == 0 && self.repetition_penalty != 1.0)
            || self.top_k.is_some_and(|k| k == 0 || k > head_vocab)
            || self
                .top_p
                .is_some_and(|p| !p.is_finite() || p <= 0.0 || p > 1.0)
            || self
                .min_p
                .is_some_and(|p| !p.is_finite() || !(0.0..=1.0).contains(&p))
            || !self.unified_linear.is_finite()
            || !(0.0..=1.0).contains(&self.unified_linear)
            || !self.unified_confidence.is_finite()
            || !self.unified_quadratic.is_finite()
        {
            return Err(VokraError::InvalidArgument(
                "zonos sampling parameters are outside the audited domain".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Zonos TTS engine handle.
///
/// Carries the resolved config, weight store, and an optional DAC codec
/// bind ([`crate::dac::Dac`] — MIT). [`Self::synthesize`] is the primary
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
    dac: Option<Dac>,
    backend: BackendKind,
    #[cfg(test)]
    dac_fixture: Option<DacCodecGguf>,
}

impl ZonosTts {
    /// Binds the fixed 246-tensor Zonos transformer artifact into the typed
    /// native store. The DAC is intentionally a separate required resource;
    /// callers must attach the authenticated 44.1-kHz [`crate::dac::Dac`]
    /// with [`Self::with_dac`] before requesting PCM.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let config = ZonosConfig::zonos_v0_1_transformer();
        let checkpoint = ZonosCheckpoint::from_gguf(file)?;
        let weights = checkpoint.load_weights(file, &config)?;
        Self::new(config, weights)
    }

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
        if let Some(prefix) = &weights.prefix_conditioner {
            prefix.validate(bb.d_model)?;
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
        if weights.norm_f_w.len() != bb.d_model || weights.norm_f_b.len() != bb.d_model {
            return Err(VokraError::InvalidArgument(
                "zonos weights: backbone.norm_f must have d_model gamma and beta".to_owned(),
            ));
        }
        Ok(Self {
            cfg,
            weights,
            dac: None,
            backend: BackendKind::Cpu,
            #[cfg(test)]
            dac_fixture: None,
        })
    }

    /// Injects the complete [`crate::dac::Dac`] — the terminal factorized
    /// RVQ codes → PCM decoder. The lower-level `DacCodecGguf` container is
    /// intentionally not accepted as a Zonos runtime substitute.
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
    pub fn with_dac(mut self, dac: Dac) -> Result<Self> {
        if dac.n_codebooks() < self.cfg.num_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "zonos with_dac: dac has {} codebooks but Zonos emits {} channels",
                dac.n_codebooks(),
                self.cfg.num_codebooks,
            )));
        }
        if dac.sample_rate() != self.cfg.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "zonos with_dac: dac sample_rate {} Hz != Zonos config sample_rate \
                 {} Hz (Zonos-v0.1 is bound to descript/dac_44khz)",
                dac.sample_rate(),
                self.cfg.sample_rate,
            )));
        }
        self.dac = Some(dac);
        Ok(self)
    }

    /// Selects one backend for the complete Zonos transformer + DAC route.
    /// Backend capability is checked at [`Self::forward_conditioned`] and
    /// [`Self::decode_codes`] entry; selection itself never probes or falls
    /// back to another backend.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    #[cfg(test)]
    fn with_dac_fixture(mut self, dac: DacCodecGguf) -> Result<Self> {
        if dac.attrs.n_codebooks < self.cfg.num_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "zonos with_dac fixture: dac has {} codebooks but Zonos emits {} channels",
                dac.attrs.n_codebooks, self.cfg.num_codebooks,
            )));
        }
        if dac.sample_rate != self.cfg.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "zonos with_dac: dac sample_rate {} Hz != Zonos config sample_rate {} Hz \
                 (Zonos-v0.1 is bound to descript/dac_44khz)",
                dac.sample_rate, self.cfg.sample_rate,
            )));
        }
        self.dac_fixture = Some(dac);
        Ok(self)
    }

    #[cfg(test)]
    fn dac_is_bound(&self) -> bool {
        self.dac.is_some() || self.dac_fixture.is_some()
    }

    #[cfg(not(test))]
    fn dac_is_bound(&self) -> bool {
        self.dac.is_some()
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &ZonosConfig {
        &self.cfg
    }

    /// The bound DAC codec, if any.
    #[must_use]
    pub fn dac(&self) -> Option<&Dac> {
        self.dac.as_ref()
    }

    /// Reports whether this handle contains the deterministic diagnostic
    /// fixture rather than a strict upstream checkpoint binding.
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Synthesizes PCM given a phoneme-id sequence.
    ///
    /// `phoneme_ids` is an eSpeak-NG phoneme id sequence (upstream Zonos
    /// consumes the same ids the `EspeakPhonemeConditioner` would). This
    /// legacy entry point intentionally cannot provide the authenticated
    /// speaker/emotion/scalar controls required by the production packet.
    ///
    /// This is the primary text → PCM entry point. **Real weights required**:
    /// synthesized-weight builds cannot produce meaningful audio (they'd
    /// be noise or a hallucinated "silence"), so this returns
    /// [`VokraError::NotImplemented`] naming the blocker. Callers verify
    /// the shape flow through [`ZonosTts::new`] +
    /// [`ZonosWeights::synthesized`] today. Production callers must use the
    /// authenticated [`Self::synthesize_with_conditioning_packet`] entry
    /// point because this legacy API lacks the required controls.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on empty `phoneme_ids` or a
    ///   negative id (upstream eSpeak ids are always ≥ 0).
    /// - [`VokraError::NotImplemented`] because raw phoneme-only input cannot
    ///   authenticate the complete conditioning contract.
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
                 (Apache 2.0, Zyphra/Zonos-v0.1-transformer) before invoking synthesize.",
            ));
        }
        if !self.dac_is_bound() {
            return Err(VokraError::NotImplemented(
                "zonos synthesize: no DAC codec has been bound — call \
                 `.with_dac(crate::dac::Dac::from_gguf(&dac_gguf)?)?` first. Zonos's \
                 decoder emits 9 DAC codebook channels per step which the DAC 44.1 kHz \
                 codec reduces to PCM; without it there is nothing honest to return \
                 (FR-EX-08).",
            ));
        }
        Err(VokraError::NotImplemented(
            "zonos synthesize: raw phoneme-only input cannot authenticate speaker, emotion, \
             scalar controls, and language; use synthesize_with_conditioning_packet",
        ))
    }

    /// Runs the source-authenticated typed transformer over a prepared
    /// conditioning packet and returns one logits vector per codebook.
    ///
    /// This returns the current guided logits for the prepared packet. The
    /// packet generation entry point adds delayed autoregressive sampling and
    /// the separately authenticated DAC decode; both use the same bound
    /// transformer and Compute backend.
    pub fn forward_conditioned(
        &self,
        packet: &ZonosConditioningPacket,
        guidance_scale: f32,
    ) -> Result<Vec<Vec<f32>>> {
        let compute = crate::compute::Compute::for_backend(self.backend, ZONOS_HOT_OPS)?;
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "zonos forward_conditioned: synthesized weights are diagnostic only",
            ));
        }
        if !guidance_scale.is_finite() {
            return Err(VokraError::InvalidArgument(
                "zonos guidance_scale must be finite".to_owned(),
            ));
        }
        let d = self.cfg.backbone.d_model;
        if self.weights.prefix_conditioner.is_none() {
            return Err(VokraError::NotImplemented(
                "zonos forward_conditioned: authenticated native seven-conditioner weights are not bound; packet projected-prefix fields are diagnostic-only",
            ));
        }
        let delayed = if packet.prompt_codes.is_empty() {
            vec![Vec::new(); self.cfg.num_codebooks]
        } else {
            if packet.prompt_codes.len() != self.cfg.num_codebooks
                || packet.prompt_codes.iter().any(Vec::is_empty)
            {
                return Err(VokraError::InvalidArgument(
                    "zonos prompt codes must have one non-empty row per codebook".to_owned(),
                ));
            }
            self.cfg.apply_delay_pattern(&packet.prompt_codes)?
        };
        self.forward_conditioned_delayed(packet, &delayed, &compute, guidance_scale, d)
    }

    fn forward_conditioned_delayed(
        &self,
        packet: &ZonosConditioningPacket,
        delayed: &[Vec<u32>],
        compute: &crate::compute::Compute,
        guidance_scale: f32,
        d: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let (mut conditional_input, mut unconditional_input) = conditioning::build_prefix(
            packet,
            self.weights
                .prefix_conditioner
                .as_ref()
                .ok_or(VokraError::NotImplemented(
                    "zonos prefix conditioner weights are not bound",
                ))?,
            compute,
            d,
        )?;
        if delayed.len() != self.cfg.num_codebooks {
            return Err(VokraError::InvalidArgument(
                "zonos delayed codebook count mismatch".to_owned(),
            ));
        }
        let delayed_frames = delayed.first().map_or(0, Vec::len);
        if delayed.iter().any(|row| row.len() != delayed_frames) {
            return Err(VokraError::InvalidArgument(
                "zonos delayed codebook rows have different lengths".to_owned(),
            ));
        }
        for frame in 0..delayed_frames {
            let mut embedding = vec![0.0; d];
            for (codebook, row) in delayed.iter().enumerate() {
                let token = row[frame] as usize;
                if token >= self.cfg.codebook_vocab {
                    return Err(VokraError::InvalidArgument(
                        "zonos delayed code contains an out-of-range token".to_owned(),
                    ));
                }
                let table = &self.weights.codebook_embeddings[codebook];
                for index in 0..d {
                    embedding[index] += table[token * d + index];
                }
            }
            conditional_input.extend_from_slice(&embedding);
            unconditional_input.extend_from_slice(&embedding);
        }
        let frames = conditional_input.len() / d;
        let conditional = transformer::forward_incremental(
            &self.cfg,
            &self.weights,
            &conditional_input,
            frames,
            compute,
        )?;
        let unconditional = transformer::forward_incremental(
            &self.cfg,
            &self.weights,
            &unconditional_input,
            frames,
            compute,
        )?;
        let mut guided = Vec::with_capacity(self.cfg.num_codebooks);
        for codebook in 0..self.cfg.num_codebooks {
            let mut row = Vec::with_capacity(self.cfg.head_vocab);
            for index in 0..self.cfg.head_vocab {
                let u = unconditional[codebook][index];
                let c = conditional[codebook][index];
                row.push(u + guidance_scale * (c - u));
            }
            guided.push(row);
        }
        Ok(guided)
    }

    /// Decodes generated Zonos codebooks through the complete bound
    /// [`crate::dac::Dac`].  The codebook-major API is converted to the DAC's
    /// frame-major wire format without selecting a backend implicitly.
    pub fn decode_codes(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let Some(dac) = self.dac.as_ref() else {
            return Err(VokraError::NotImplemented(
                "zonos decode_codes: a complete 44.1-kHz crate::dac::Dac is required",
            ));
        };
        if dac.backend() != self.backend {
            return Err(VokraError::InvalidArgument(
                "zonos transformer and DAC backends must match".to_owned(),
            ));
        }
        if codes.len() != self.cfg.num_codebooks || codes.iter().any(Vec::is_empty) {
            return Err(VokraError::InvalidArgument(
                "zonos decode_codes requires one non-empty row per codebook".to_owned(),
            ));
        }
        let frames = codes[0].len();
        if codes.iter().any(|row| row.len() != frames) {
            return Err(VokraError::InvalidArgument(
                "zonos decode_codes rows must have equal length".to_owned(),
            ));
        }
        let mut frame_major = Vec::with_capacity(frames * self.cfg.num_codebooks);
        for frame in 0..frames {
            for row in codes {
                let token = row[frame];
                if token as usize >= 1024 {
                    return Err(VokraError::InvalidArgument(
                        "zonos decode_codes contains an out-of-range emitted code".to_owned(),
                    ));
                }
                frame_major.push(token);
            }
        }
        dac.decode_codes(&frame_major)
    }

    /// Greedily generates delayed nine-codebook frames, then decodes the
    /// generated (not prompt) frames through the bound DAC. CB0 is the only
    /// termination surface; after CB0 EOS the upstream nine-column terminal
    /// drain is materialized before [`ZonosConfig::revert_delay_pattern`].
    /// Each causal pass uses the layer-wise Compute KV-step implementation;
    /// delay is applied only to the codebook matrix and never re-applied to
    /// already delayed rows.
    pub fn generate_greedy(
        &self,
        packet: &ZonosConditioningPacket,
        max_steps: usize,
        guidance_scale: f32,
    ) -> Result<Vec<f32>> {
        self.generate_with_sampling(
            packet,
            max_steps,
            guidance_scale,
            &ZonosSamplingParams::greedy(),
            &[],
        )
    }

    /// Generates with source-shaped sampling filters. Stochastic operation
    /// requires one positive exponential draw per vocabulary candidate, per
    /// codebook, at every sampled frame (the exact source
    /// `multinomial(...).exponential_()` contract); draws are caller-owned so
    /// the native route has no hidden RNG. The flat draw order is sampled
    /// frame, codebook, then vocabulary index; temperature-zero generation
    /// consumes no draws.
    pub fn generate_with_sampling(
        &self,
        packet: &ZonosConditioningPacket,
        max_steps: usize,
        guidance_scale: f32,
        sampling: &ZonosSamplingParams,
        exponential_draws: &[f32],
    ) -> Result<Vec<f32>> {
        let codes = self.generate_codes_with_sampling(
            packet,
            max_steps,
            guidance_scale,
            sampling,
            exponential_draws,
        )?;
        self.decode_codes(&codes)
    }

    /// Generates the exact codebook-major frame matrix before DAC decoding.
    ///
    /// This is the validation seam used to compare source-generated codes
    /// independently of PCM tolerances. It has the same authenticated packet,
    /// backend, cache, delay, and caller-owned-draw requirements as
    /// [`Self::generate_with_sampling`].
    pub fn generate_codes_with_sampling(
        &self,
        packet: &ZonosConditioningPacket,
        max_steps: usize,
        guidance_scale: f32,
        sampling: &ZonosSamplingParams,
        exponential_draws: &[f32],
    ) -> Result<Vec<Vec<u32>>> {
        if max_steps == 0 {
            return Err(VokraError::InvalidArgument(
                "zonos generate_greedy requires max_steps > 0".to_owned(),
            ));
        }
        sampling.validate(self.cfg.head_vocab)?;
        if sampling.temperature == 0.0 && !exponential_draws.is_empty() {
            return Err(VokraError::InvalidArgument(
                "zonos greedy sampling consumes no exponential draws".to_owned(),
            ));
        }
        let prompt_frames = packet.prompt_codes.first().map_or(0, Vec::len);
        let max_delay = self.cfg.delay_pattern.iter().copied().max().unwrap_or(0);
        let mut delayed = initialize_generation_delay(&self.cfg, &packet.prompt_codes, max_steps)?;
        let compute = crate::compute::Compute::for_backend(self.backend, ZONOS_HOT_OPS)?;
        let d = self.cfg.backbone.d_model;
        let mut offset = prompt_frames + 1;
        let prefix = self
            .weights
            .prefix_conditioner
            .as_ref()
            .ok_or(VokraError::NotImplemented(
                "zonos prefix conditioner weights are not bound",
            ))?;
        let (conditional_prefix, unconditional_prefix) =
            conditioning::build_prefix(packet, prefix, &compute, d)?;
        let mut conditional_cache = transformer::KvCache::new(&self.cfg)?;
        let mut unconditional_cache = transformer::KvCache::new(&self.cfg)?;
        let mut conditional_logits = None;
        let mut unconditional_logits = None;
        for row in conditional_prefix.chunks_exact(d) {
            conditional_logits =
                Some(conditional_cache.step(&self.cfg, &self.weights, row, &compute)?);
        }
        for row in unconditional_prefix.chunks_exact(d) {
            unconditional_logits =
                Some(unconditional_cache.step(&self.cfg, &self.weights, row, &compute)?);
        }
        for frame in 0..offset {
            let embedding = embed_delayed_frame(&self.cfg, &self.weights, &delayed, frame, d)?;
            conditional_logits =
                Some(conditional_cache.step(&self.cfg, &self.weights, &embedding, &compute)?);
            unconditional_logits =
                Some(unconditional_cache.step(&self.cfg, &self.weights, &embedding, &compute)?);
        }
        let mut next_logits = guided_logits(
            conditional_logits
                .as_ref()
                .ok_or(VokraError::InvalidArgument(
                    "zonos conditioner produced no logits".to_owned(),
                ))?,
            unconditional_logits
                .as_ref()
                .ok_or(VokraError::InvalidArgument(
                    "zonos unconditional produced no logits".to_owned(),
                ))?,
            self.cfg.num_codebooks,
            self.cfg.head_vocab,
            guidance_scale,
        )?;
        let history = vec![Vec::new(); self.cfg.num_codebooks];
        let mut draw_cursor = 0usize;
        // Upstream samples the first frame from the prefill logits, writes it
        // before entering the decode loop, and only applies CB0 EOS stopping
        // inside that loop. Keeping this boundary explicit is important for
        // the one-frame/initial-EOS edge case.
        let initial = sample_tokens(
            &next_logits,
            &history,
            sampling,
            exponential_draws,
            &mut draw_cursor,
            self.cfg.eos_token_id,
            false,
        )?;
        if offset >= delayed[0].len() {
            return Err(VokraError::InvalidArgument(
                "zonos delayed generation exceeded reserved sequence".to_owned(),
            ));
        }
        scatter_unknown_tokens(&mut delayed, offset, &initial)?;
        offset += 1;
        let mut remaining_steps = delayed[0].len().checked_sub(offset).ok_or_else(|| {
            VokraError::InvalidArgument("zonos delayed generation offset is invalid".to_owned())
        })?;
        let mut stopping = false;
        while remaining_steps > 0 {
            if offset >= delayed[0].len() {
                return Err(VokraError::InvalidArgument(
                    "zonos delayed generation exceeded reserved sequence".to_owned(),
                ));
            }
            // The first loop input is the frame written by the prefill
            // sample. Decode it exactly once before sampling the next frame;
            // never reuse the prefill logits for this second sample.
            let embedding = embed_delayed_frame(&self.cfg, &self.weights, &delayed, offset - 1, d)?;
            conditional_logits =
                Some(conditional_cache.step(&self.cfg, &self.weights, &embedding, &compute)?);
            unconditional_logits =
                Some(unconditional_cache.step(&self.cfg, &self.weights, &embedding, &compute)?);
            next_logits = guided_logits(
                conditional_logits.as_ref().ok_or_else(|| {
                    VokraError::InvalidArgument("zonos conditioner produced no logits".to_owned())
                })?,
                unconditional_logits.as_ref().ok_or_else(|| {
                    VokraError::InvalidArgument("zonos unconditional produced no logits".to_owned())
                })?,
                self.cfg.num_codebooks,
                self.cfg.head_vocab,
                guidance_scale,
            )?;
            let histories: Vec<Vec<u32>> =
                delayed.iter().map(|row| row[..offset].to_vec()).collect();
            let next = sample_tokens(
                &next_logits,
                &histories,
                sampling,
                exponential_draws,
                &mut draw_cursor,
                self.cfg.eos_token_id,
                true,
            )?;
            if next[0] == self.cfg.eos_token_id {
                remaining_steps = remaining_steps.min(self.cfg.num_codebooks);
                stopping = true;
            }
            let stop_index = self
                .cfg
                .num_codebooks
                .saturating_sub(remaining_steps)
                .min(self.cfg.num_codebooks - 1);
            let written = apply_stopping_frame(
                &next,
                stopping,
                stop_index,
                self.cfg.masked_token_id,
                self.cfg.eos_token_id,
            )?;
            scatter_unknown_tokens(&mut delayed, offset, &written)?;
            remaining_steps -= 1;
            offset += 1;
        }
        if sampling.temperature > 0.0 && draw_cursor != exponential_draws.len() {
            return Err(VokraError::InvalidArgument(
                "zonos stochastic draw count has unused values".to_owned(),
            ));
        }
        let output_extent = offset.checked_sub(max_delay).ok_or_else(|| {
            VokraError::InvalidArgument("zonos delayed output extent underflow".to_owned())
        })?;
        let original_frames = prompt_frames.checked_add(max_steps).ok_or_else(|| {
            VokraError::InvalidArgument("zonos delayed output length overflow".to_owned())
        })?;
        let required_delayed = original_frames.checked_add(max_delay).ok_or_else(|| {
            VokraError::InvalidArgument("zonos delayed generation length overflow".to_owned())
        })?;
        for row in &mut delayed {
            row.truncate(required_delayed);
        }
        let mut generated =
            revert_generation_delay(&self.cfg, &delayed, original_frames, output_extent)?;
        for row in &mut generated {
            row.drain(..prompt_frames);
        }
        Ok(generated)
    }

    /// Production packet entry point. Raw text is intentionally not accepted;
    /// callers must supply the versioned, prehashed offline packet and the
    /// strictly bound model/DAC pair.
    pub fn synthesize_with_conditioning_packet(
        &self,
        packet: &ZonosConditioningPacket,
        max_steps: usize,
        guidance_scale: f32,
    ) -> Result<Vec<f32>> {
        self.generate_greedy(packet, max_steps, guidance_scale)
    }
}

fn apply_stopping_frame(
    sampled: &[u32],
    stopping: bool,
    stop_index: usize,
    masked_token_id: u32,
    eos_token_id: u32,
) -> Result<Vec<u32>> {
    if sampled.is_empty() || stop_index >= sampled.len() {
        return Err(VokraError::InvalidArgument(
            "zonos stopping frame has an invalid codebook shape".to_owned(),
        ));
    }
    if !stopping {
        return Ok(sampled.to_vec());
    }
    let mut written = sampled.to_vec();
    for token in written.iter_mut().take(stop_index) {
        *token = masked_token_id;
    }
    written[stop_index] = eos_token_id;
    Ok(written)
}

/// Builds the source delayed matrix from the complete prompt-plus-unknown
/// matrix in one pass.  Applying delay only to the prompt would lose the
/// trailing fixed masks and makes masked-scatter generation observably wrong.
fn initialize_generation_delay(
    config: &ZonosConfig,
    prompt_codes: &[Vec<u32>],
    max_steps: usize,
) -> Result<Vec<Vec<u32>>> {
    if max_steps == 0 {
        return Err(VokraError::InvalidArgument(
            "zonos delayed generation requires max_steps > 0".to_owned(),
        ));
    }
    let prompt_frames = prompt_codes.first().map_or(0, Vec::len);
    if !prompt_codes.is_empty()
        && (prompt_codes.len() != config.num_codebooks
            || prompt_codes.iter().any(|row| row.len() != prompt_frames)
            || prompt_frames == 0)
    {
        return Err(VokraError::InvalidArgument(
            "zonos prompt codes must have one equal non-empty row per codebook".to_owned(),
        ));
    }
    for row in prompt_codes {
        if row
            .iter()
            .any(|&token| token as usize >= config.codebook_vocab)
        {
            return Err(VokraError::InvalidArgument(
                "zonos prompt code is outside the codebook vocabulary".to_owned(),
            ));
        }
    }
    let source_frames = prompt_frames.checked_add(max_steps).ok_or_else(|| {
        VokraError::InvalidArgument("zonos delayed generation length overflow".to_owned())
    })?;
    let max_delay = config.delay_pattern.iter().copied().max().unwrap_or(0);
    let delayed_frames = source_frames.checked_add(max_delay).ok_or_else(|| {
        VokraError::InvalidArgument("zonos delayed generation length overflow".to_owned())
    })?;
    let mut delayed = vec![vec![config.masked_token_id; delayed_frames]; config.num_codebooks];
    for (codebook, row) in delayed.iter_mut().enumerate() {
        let delay = config.delay_pattern[codebook];
        for frame in 0..source_frames {
            row[frame + delay] = if frame < prompt_frames {
                prompt_codes[codebook][frame]
            } else {
                UNKNOWN_GENERATION_TOKEN
            };
        }
    }
    Ok(delayed)
}

/// Implements the source `masked_scatter`: only unknown generation slots are
/// replaced. Prompt and fixed delay masks are never overwritten by samples.
fn scatter_unknown_tokens(delayed: &mut [Vec<u32>], offset: usize, sampled: &[u32]) -> Result<()> {
    if delayed.is_empty()
        || delayed.len() != sampled.len()
        || delayed.iter().any(|row| offset >= row.len())
    {
        return Err(VokraError::InvalidArgument(
            "zonos delayed masked-scatter shape mismatch".to_owned(),
        ));
    }
    for (row, &token) in delayed.iter_mut().zip(sampled) {
        if row[offset] == UNKNOWN_GENERATION_TOKEN {
            row[offset] = token;
        }
    }
    Ok(())
}

/// Generation-only delay reversion. The upstream reverter lets EOS, mask,
/// and still-unknown terminal slots pass through before mapping every special
/// value to DAC code zero. The public `revert_delay_pattern` remains strict
/// and accepts ordinary emitted codes only.
fn revert_generation_delay(
    config: &ZonosConfig,
    delayed: &[Vec<u32>],
    original_frames: usize,
    generated_extent: usize,
) -> Result<Vec<Vec<u32>>> {
    if delayed.len() != config.num_codebooks
        || original_frames == 0
        || generated_extent > original_frames
    {
        return Err(VokraError::InvalidArgument(
            "zonos generation revert shape is invalid".to_owned(),
        ));
    }
    let max_delay = config.delay_pattern.iter().copied().max().unwrap_or(0);
    let expected_len = original_frames.checked_add(max_delay).ok_or_else(|| {
        VokraError::InvalidArgument("zonos generation revert length overflow".to_owned())
    })?;
    if delayed.iter().any(|row| row.len() != expected_len) {
        return Err(VokraError::InvalidArgument(
            "zonos generation delayed rows have invalid length".to_owned(),
        ));
    }
    let mut result = vec![vec![0; generated_extent]; config.num_codebooks];
    for (codebook, output) in result.iter_mut().enumerate() {
        let delay = config.delay_pattern[codebook];
        for (frame, token) in output.iter_mut().enumerate() {
            let delayed_token = delayed[codebook][frame + delay];
            *token = if delayed_token == UNKNOWN_GENERATION_TOKEN
                || delayed_token == config.eos_token_id
                || delayed_token == config.masked_token_id
            {
                0
            } else if delayed_token as usize >= config.codebook_vocab {
                return Err(VokraError::InvalidArgument(
                    "zonos generation contains an out-of-range code".to_owned(),
                ));
            } else {
                delayed_token
            };
        }
    }
    Ok(result)
}

fn embed_delayed_frame(
    config: &ZonosConfig,
    weights: &ZonosWeights,
    delayed: &[Vec<u32>],
    frame: usize,
    d_model: usize,
) -> Result<Vec<f32>> {
    if delayed.len() != config.num_codebooks
        || d_model != config.backbone.d_model
        || delayed.iter().any(|row| frame >= row.len())
        || weights.codebook_embeddings.len() != config.num_codebooks
    {
        return Err(VokraError::InvalidArgument(
            "zonos delayed embedding shape mismatch".to_owned(),
        ));
    }
    let mut embedding = vec![0.0; d_model];
    for (codebook, row) in delayed.iter().enumerate() {
        let token = if row[frame] == UNKNOWN_GENERATION_TOKEN {
            config.masked_token_id as usize
        } else {
            row[frame] as usize
        };
        if token >= config.codebook_vocab {
            return Err(VokraError::InvalidArgument(
                "zonos delayed embedding token is outside codebook vocabulary".to_owned(),
            ));
        }
        let table = &weights.codebook_embeddings[codebook];
        let start = token.checked_mul(d_model).ok_or_else(|| {
            VokraError::InvalidArgument("zonos delayed embedding offset overflow".to_owned())
        })?;
        let end = start.checked_add(d_model).ok_or_else(|| {
            VokraError::InvalidArgument("zonos delayed embedding range overflow".to_owned())
        })?;
        let values = table.get(start..end).ok_or_else(|| {
            VokraError::InvalidArgument("zonos delayed embedding table shape mismatch".to_owned())
        })?;
        for (out, value) in embedding.iter_mut().zip(values) {
            *out += value;
        }
    }
    if embedding.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "zonos delayed embedding is non-finite".to_owned(),
        ));
    }
    Ok(embedding)
}

#[cfg(test)]
fn fill_terminal_drain(
    delayed: &mut [Vec<u32>],
    start: usize,
    masked_token_id: u32,
    eos: Option<(u32, usize)>,
) -> Result<()> {
    if delayed.is_empty() || delayed.iter().any(|row| row.len() != delayed[0].len()) {
        return Err(VokraError::InvalidArgument(
            "zonos terminal drain rows have invalid shape".to_owned(),
        ));
    }
    let codebooks = delayed.len();
    let count = eos.map_or(codebooks, |(_, remaining)| remaining.min(codebooks));
    for step in 0..count {
        let column = start.checked_add(step).ok_or_else(|| {
            VokraError::InvalidArgument("zonos terminal drain index overflow".to_owned())
        })?;
        if column >= delayed[0].len() {
            return Err(VokraError::InvalidArgument(
                "zonos delayed terminal drain exceeded reserved sequence".to_owned(),
            ));
        }
        let active = eos
            .map(|(_, remaining)| (codebooks - remaining.min(codebooks) + step).min(codebooks - 1));
        for (codebook, row) in delayed.iter_mut().enumerate() {
            row[column] = match active {
                Some(active) if codebook < active => masked_token_id,
                Some(active) if codebook == active => eos.map_or(masked_token_id, |(id, _)| id),
                Some(_) => row[column],
                None => masked_token_id,
            };
        }
    }
    Ok(())
}

fn guided_logits(
    conditional: &[Vec<f32>],
    unconditional: &[Vec<f32>],
    codebooks: usize,
    head_vocab: usize,
    guidance_scale: f32,
) -> Result<Vec<Vec<f32>>> {
    if !guidance_scale.is_finite()
        || conditional.len() != codebooks
        || unconditional.len() != codebooks
        || conditional
            .iter()
            .chain(unconditional)
            .any(|row| row.len() != head_vocab || row.iter().any(|value| !value.is_finite()))
    {
        return Err(VokraError::InvalidArgument(
            "zonos guided logits shape or finiteness mismatch".to_owned(),
        ));
    }
    let guided: Vec<Vec<f32>> = (0..codebooks)
        .map(|codebook| {
            (0..head_vocab)
                .map(|index| {
                    let u = unconditional[codebook][index];
                    let c = conditional[codebook][index];
                    u + guidance_scale * (c - u)
                })
                .collect()
        })
        .collect();
    if guided.iter().flatten().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "zonos guided logits are non-finite".to_owned(),
        ));
    }
    Ok(guided)
}

fn sample_tokens(
    logits: &[Vec<f32>],
    histories: &[Vec<u32>],
    sampling: &ZonosSamplingParams,
    exponential_draws: &[f32],
    draw_cursor: &mut usize,
    eos_token_id: u32,
    suppress_non_primary_eos: bool,
) -> Result<Vec<u32>> {
    if logits.len() != histories.len() {
        return Err(VokraError::InvalidArgument(
            "zonos sampling codebook/history count mismatch".to_owned(),
        ));
    }
    let mut result = Vec::with_capacity(logits.len());
    for (codebook, row) in logits.iter().enumerate() {
        sampling.validate(row.len())?;
        if row.is_empty() || row.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "zonos sampling logits must be finite and non-empty".to_owned(),
            ));
        }
        let mut filtered = row.clone();
        for &token in histories[codebook]
            .iter()
            .rev()
            .take(sampling.repetition_window)
        {
            let index = (token as usize).min(row.len() - 1);
            if let Some(value) = filtered.get_mut(index) {
                if *value >= 0.0 {
                    *value /= sampling.repetition_penalty;
                } else {
                    *value *= sampling.repetition_penalty;
                }
            }
        }
        if suppress_non_primary_eos && codebook != 0 {
            let eos = eos_token_id as usize;
            if eos < filtered.len() {
                filtered[eos] = f32::NEG_INFINITY;
            }
        }
        if filtered
            .iter()
            .any(|value| !value.is_finite() && *value != f32::NEG_INFINITY)
        {
            return Err(VokraError::InvalidArgument(
                "zonos sampling repetition filter produced non-finite logits".to_owned(),
            ));
        }
        if sampling.temperature == 0.0 {
            let mut best = None;
            for (index, &value) in filtered.iter().enumerate() {
                if value.is_finite() && best.is_none_or(|(_, best_value)| value > best_value) {
                    best = Some((index, value));
                }
            }
            let (index, _) = best.ok_or_else(|| {
                VokraError::InvalidArgument("zonos sampling filters removed every token".to_owned())
            })?;
            result.push(u32::try_from(index).map_err(|_| {
                VokraError::InvalidArgument("zonos sampled token index overflow".to_owned())
            })?);
            continue;
        }
        for value in &mut filtered {
            *value /= sampling.temperature;
        }
        if filtered
            .iter()
            .any(|value| !value.is_finite() && *value != f32::NEG_INFINITY)
        {
            return Err(VokraError::InvalidArgument(
                "zonos sampling temperature produced non-finite logits".to_owned(),
            ));
        }
        let max = filtered
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return Err(VokraError::InvalidArgument(
                "zonos sampling filters removed every token".to_owned(),
            ));
        }
        let mut probabilities: Vec<f32> = filtered
            .iter()
            .map(|value| {
                if value.is_finite() {
                    (*value - max).exp()
                } else {
                    0.0
                }
            })
            .collect();
        let total = probabilities.iter().sum::<f32>();
        if !total.is_finite() || total <= 0.0 {
            return Err(VokraError::InvalidArgument(
                "zonos sampling probability mass is empty".to_owned(),
            ));
        }
        for probability in &mut probabilities {
            *probability /= total;
        }
        if sampling.unified_linear > 0.0 {
            let entropy = probabilities
                .iter()
                .filter(|&&probability| probability > 0.0)
                .map(|&probability| -probability * probability.max(1.0e-20).ln())
                .sum::<f32>();
            let transformed: Vec<f32> = probabilities
                .iter()
                .map(|&probability| {
                    let log_probability = probability.max(1.0e-20).ln();
                    log_probability
                        * (sampling.unified_linear + entropy * sampling.unified_confidence)
                        - log_probability * log_probability * sampling.unified_quadratic
                })
                .collect();
            let max_transformed = transformed
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            if !max_transformed.is_finite() {
                return Err(VokraError::InvalidArgument(
                    "zonos unified sampling produced no finite logits".to_owned(),
                ));
            }
            probabilities = transformed
                .into_iter()
                .map(|value| (value - max_transformed).exp())
                .collect();
            let unified_mass = probabilities.iter().sum::<f32>();
            if !unified_mass.is_finite() || unified_mass <= 0.0 {
                return Err(VokraError::InvalidArgument(
                    "zonos unified sampling probability mass is empty".to_owned(),
                ));
            }
            for probability in &mut probabilities {
                *probability /= unified_mass;
            }
        }
        if let Some(top_p) = sampling.top_p {
            let mut order: Vec<usize> = (0..probabilities.len()).collect();
            order.sort_by(|&a, &b| {
                probabilities[b]
                    .partial_cmp(&probabilities[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut cumulative = 0.0;
            for &index in &order {
                let probability = probabilities[index];
                cumulative += probability;
                // Match `cumsum(sorted) - sorted > p`: retain the first
                // token that crosses the threshold.
                if cumulative - probability > top_p {
                    probabilities[index] = 0.0;
                }
            }
        }
        if let Some(top_k) = sampling.top_k {
            let mut order: Vec<usize> = (0..probabilities.len()).collect();
            order.sort_by(|&a, &b| {
                probabilities[b]
                    .partial_cmp(&probabilities[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let pivot = probabilities[order[top_k - 1]];
            for probability in &mut probabilities {
                if *probability < pivot {
                    *probability = 0.0;
                }
            }
        }
        if let Some(min_p) = sampling.min_p {
            let max_probability = probabilities.iter().copied().fold(0.0, f32::max);
            let threshold = max_probability * min_p;
            for probability in &mut probabilities {
                if *probability < threshold {
                    *probability = 0.0;
                }
            }
        }
        let mass = probabilities.iter().sum::<f32>();
        if !mass.is_finite() || mass <= 0.0 {
            return Err(VokraError::InvalidArgument(
                "zonos sampling filters removed every token".to_owned(),
            ));
        }
        for probability in &mut probabilities {
            *probability /= mass;
        }
        let mut best = None;
        for (index, &probability) in probabilities.iter().enumerate() {
            let draw = exponential_draws
                .get(*draw_cursor)
                .copied()
                .ok_or_else(|| {
                    VokraError::InvalidArgument(
                        "zonos stochastic sampling requires one draw per vocabulary candidate"
                            .to_owned(),
                    )
                })?;
            if !draw.is_finite() || draw <= 0.0 {
                return Err(VokraError::InvalidArgument(
                    "zonos exponential draws must be finite and positive".to_owned(),
                ));
            }
            *draw_cursor += 1;
            let score = probability / draw;
            if score.is_finite() && best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((index, score));
            }
        }
        let (index, _) = best.ok_or_else(|| {
            VokraError::InvalidArgument("zonos sampling probability mass is empty".to_owned())
        })?;
        result.push(u32::try_from(index).map_err(|_| {
            VokraError::InvalidArgument("zonos sampled token index overflow".to_owned())
        })?);
    }
    Ok(result)
}

#[cfg(test)]
fn greedy_tokens(logits: &[Vec<f32>], codebooks: usize, head_vocab: usize) -> Result<Vec<u32>> {
    if logits.len() != codebooks || logits.iter().any(|row| row.len() != head_vocab) {
        return Err(VokraError::InvalidArgument(
            "zonos greedy logits shape mismatch".to_owned(),
        ));
    }
    let mut tokens = Vec::with_capacity(codebooks);
    for row in logits {
        let mut best = None;
        for (index, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(VokraError::InvalidArgument(
                    "zonos greedy logits contain non-finite values".to_owned(),
                ));
            }
            if best.is_none_or(|(_, best_value)| value > best_value) {
                best = Some((index, value));
            }
        }
        let (index, _) = best.ok_or_else(|| {
            VokraError::InvalidArgument("zonos greedy logits row is empty".to_owned())
        })?;
        tokens.push(u32::try_from(index).map_err(|_| {
            VokraError::InvalidArgument("zonos greedy token index overflow".to_owned())
        })?);
    }
    Ok(tokens)
}

#[cfg(test)]
fn transformer_logits(
    cfg: &ZonosConfig,
    weights: &ZonosWeights,
    input: &[f32],
    frames: usize,
    compute: &crate::compute::Compute,
) -> Result<Vec<Vec<f32>>> {
    let bb = &cfg.backbone;
    let d = bb.d_model;
    if frames == 0 || input.len() != frames * d {
        return Err(VokraError::InvalidArgument(
            "zonos transformer input shape mismatch".to_owned(),
        ));
    }
    let mut hidden = input.to_vec();
    for block in &weights.blocks {
        let normed = layer_norm_rows(
            &hidden,
            &block.norm_1_w,
            &block.norm_1_b,
            frames,
            d,
            bb.norm_epsilon,
            compute,
        )?;
        let qh = bb.q_hidden();
        let kvh = bb.kv_hidden();
        let qkv_width = qh + 2 * kvh;
        let mut q = vec![0.0; frames * qh];
        let mut k = vec![0.0; frames * kvh];
        let mut v = vec![0.0; frames * kvh];
        let mut packed = vec![0.0; frames * qkv_width];
        compute.gemm_f32(
            frames,
            qkv_width,
            d,
            &normed,
            &block.qkv_proj,
            None,
            &mut packed,
        )?;
        for frame in 0..frames {
            q[frame * qh..(frame + 1) * qh]
                .copy_from_slice(&packed[frame * qkv_width..frame * qkv_width + qh]);
            k[frame * kvh..(frame + 1) * kvh]
                .copy_from_slice(&packed[frame * qkv_width + qh..frame * qkv_width + qh + kvh]);
            v[frame * kvh..(frame + 1) * kvh]
                .copy_from_slice(&packed[frame * qkv_width + qh + kvh..(frame + 1) * qkv_width]);
        }
        apply_interleaved_rope(&mut q, frames, bb.num_heads, bb.head_dim(), 10_000.0);
        apply_interleaved_rope(&mut k, frames, bb.num_heads_kv, bb.head_dim(), 10_000.0);
        let mut attended = vec![0.0; frames * qh];
        let groups = bb.num_heads / bb.num_heads_kv;
        let scale = (bb.head_dim() as f32).sqrt().recip();
        for frame in 0..frames {
            for head in 0..bb.num_heads {
                let kv_head = head / groups;
                let mut scores = vec![f32::NEG_INFINITY; frame + 1];
                for prior in 0..=frame {
                    let mut score = 0.0;
                    for lane in 0..bb.head_dim() {
                        score += q[frame * qh + head * bb.head_dim() + lane]
                            * k[prior * kvh + kv_head * bb.head_dim() + lane];
                    }
                    scores[prior] = score * scale;
                }
                let mut probabilities = vec![0.0; frame + 1];
                compute.softmax_f32(&scores, &mut probabilities, 1, frame + 1)?;
                for prior in 0..=frame {
                    let probability = probabilities[prior];
                    for lane in 0..bb.head_dim() {
                        attended[frame * qh + head * bb.head_dim() + lane] +=
                            probability * v[prior * kvh + kv_head * bb.head_dim() + lane];
                    }
                }
            }
        }
        let mut attention_out = vec![0.0; frames * d];
        compute.gemm_f32(
            frames,
            d,
            qh,
            &attended,
            &block.o_proj,
            None,
            &mut attention_out,
        )?;
        for (value, update) in hidden.iter_mut().zip(attention_out) {
            *value += update;
        }
        let normed = layer_norm_rows(
            &hidden,
            &block.norm_2_w,
            &block.norm_2_b,
            frames,
            d,
            bb.norm_epsilon,
            compute,
        )?;
        let fc1_width = bb.mlp_fc1_out();
        let mut ffn = vec![0.0; frames * d];
        for frame in 0..frames {
            let row = &normed[frame * d..(frame + 1) * d];
            let mut projected = vec![0.0; fc1_width];
            compute.gemm_f32(1, fc1_width, d, row, &block.mlp_fc1, None, &mut projected)?;
            let mut gate = vec![0.0; bb.d_intermediate];
            gate.copy_from_slice(&projected[bb.d_intermediate..]);
            let mut silu_gate = vec![0.0; bb.d_intermediate];
            compute.silu_f32(&gate, &mut silu_gate)?;
            let mut activated = vec![0.0; bb.d_intermediate];
            for intermediate in 0..bb.d_intermediate {
                activated[intermediate] = projected[intermediate] * silu_gate[intermediate];
            }
            compute.gemm_f32(
                1,
                d,
                bb.d_intermediate,
                &activated,
                &block.mlp_fc2,
                None,
                &mut ffn[frame * d..(frame + 1) * d],
            )?;
        }
        for (value, update) in hidden.iter_mut().zip(ffn) {
            *value += update;
        }
    }
    let final_hidden = layer_norm_rows(
        &hidden,
        &weights.norm_f_w,
        &weights.norm_f_b,
        frames,
        d,
        bb.norm_epsilon,
        compute,
    )?;
    let last = &final_hidden[(frames - 1) * d..frames * d];
    let mut logits = Vec::with_capacity(cfg.num_codebooks);
    for head in &weights.logit_heads {
        let mut row = vec![0.0; cfg.head_vocab];
        compute.gemm_f32(1, cfg.head_vocab, d, last, head, None, &mut row)?;
        logits.push(row);
    }
    Ok(logits)
}

fn layer_norm_rows(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    rows: usize,
    width: usize,
    eps: f32,
    compute: &crate::compute::Compute,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0; x.len()];
    compute.layer_norm_f32(x, &mut output, rows, width, weight, bias, eps)?;
    Ok(output)
}

#[cfg(test)]
fn apply_interleaved_rope(x: &mut [f32], rows: usize, heads: usize, head_dim: usize, base: f32) {
    let width = heads * head_dim;
    for row in 0..rows {
        for head in 0..heads {
            for pair in (0..head_dim).step_by(2) {
                let angle = row as f32 / base.powf(pair as f32 / head_dim as f32);
                let (sin, cos) = angle.sin_cos();
                let offset = row * width + head * head_dim + pair;
                let real = x[offset];
                let imag = x[offset + 1];
                x[offset] = real * cos - imag * sin;
                x[offset + 1] = real * sin + imag * cos;
            }
        }
    }
}

fn packet_digest(bytes: &[u8], digest_offset: usize) -> [u8; 32] {
    let mut canonical = Vec::with_capacity(bytes.len().saturating_sub(32));
    canonical.extend_from_slice(&bytes[..digest_offset]);
    canonical.extend_from_slice(bytes.get(digest_offset + 32..).unwrap_or_default());
    sha256(&canonical)
}

/// Small dependency-free SHA-256 used solely to authenticate the bounded
/// offline conditioning packet.  It follows FIPS 180-4 and never hashes model
/// tensor values.
fn sha256(input: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len: u64 = (input.len() as u64).wrapping_mul(8);
    let padded_len = (input.len() + 9).div_ceil(64) * 64;
    let mut padded = vec![0u8; padded_len];
    padded[..input.len()].copy_from_slice(input);
    padded[input.len()] = 0x80;
    padded[padded_len - 8..].copy_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut schedule = [0u32; 64];
        for (index, word) in schedule[..16].iter_mut().enumerate() {
            *word = u32::from_be_bytes(chunk[index * 4..index * 4 + 4].try_into().unwrap());
        }
        for index in 16..64 {
            let x = schedule[index - 15];
            let y = schedule[index - 2];
            let small_sigma0 = x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3);
            let small_sigma1 = y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(small_sigma0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(small_sigma1);
        }
        let mut working: [u32; 8] = state;
        for index in 0..64 {
            let a = working[0];
            let e = working[4];
            let big_sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let big_sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & working[5]) ^ ((!e) & working[6]);
            let majority = (a & working[1]) ^ (a & working[2]) ^ (working[1] & working[2]);
            let temp1 = working[7]
                .wrapping_add(big_sigma1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let temp2 = big_sigma0.wrapping_add(majority);
            working[7] = working[6];
            working[6] = working[5];
            working[5] = working[4];
            working[4] = working[3].wrapping_add(temp1);
            working[3] = working[2];
            working[2] = working[1];
            working[1] = working[0];
            working[0] = temp1.wrapping_add(temp2);
        }
        for index in 0..8 {
            state[index] = state[index].wrapping_add(working[index]);
        }
    }
    let mut output = [0u8; 32];
    for (index, word) in state.into_iter().enumerate() {
        output[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    output
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
                max_val: 1.0
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
    fn greedy_tokens_selects_each_codebook_head_independently() {
        let logits = vec![vec![0.0, 2.0, 1.0], vec![4.0, 1.0, 3.0]];
        assert_eq!(greedy_tokens(&logits, 2, 3).unwrap(), vec![1, 0]);
        assert!(greedy_tokens(&[vec![f32::NAN, 0.0]], 1, 2).is_err());
        assert!(greedy_tokens(&[vec![0.0]], 2, 1).is_err());
    }

    #[test]
    fn sampling_filters_and_explicit_draws_are_deterministic() {
        let sampling = ZonosSamplingParams {
            temperature: 1.0,
            top_k: Some(2),
            min_p: None,
            ..ZonosSamplingParams::default()
        };
        let logits = vec![vec![4.0, 3.0, 1.0]];
        let mut history = vec![vec![0]];
        let mut cursor = 0;
        assert_eq!(
            sample_tokens(
                &logits,
                &history,
                &sampling,
                &[1.0, 1.0, 1.0],
                &mut cursor,
                8,
                true,
            )
            .unwrap(),
            vec![1]
        );
        assert_eq!(cursor, 3);
        history[0] = vec![0, 0];
        let mut cursor = 0;
        assert!(sample_tokens(&logits, &history, &sampling, &[], &mut cursor, 8, true).is_err());

        let greedy = ZonosSamplingParams::greedy();
        let mut cursor = 0;
        assert_eq!(
            sample_tokens(&logits, &[Vec::new()], &greedy, &[], &mut cursor, 8, false).unwrap(),
            vec![0]
        );
    }

    #[test]
    fn unified_sampling_runs_before_probability_filters_and_requires_draws() {
        let sampling = ZonosSamplingParams {
            temperature: 1.0,
            min_p: None,
            unified_linear: 1.0,
            unified_confidence: 0.5,
            unified_quadratic: 0.1,
            ..ZonosSamplingParams::default()
        };
        let logits = vec![vec![2.0, 1.0, 0.0]];
        let mut cursor = 0;
        let token = sample_tokens(
            &logits,
            &[Vec::new()],
            &sampling,
            &[1.0, 1.0, 1.0],
            &mut cursor,
            8,
            false,
        )
        .unwrap();
        assert_eq!(token.len(), 1);
        assert_eq!(cursor, 3);
        let mut cursor = 0;
        assert!(
            sample_tokens(
                &logits,
                &[Vec::new()],
                &sampling,
                &[],
                &mut cursor,
                8,
                false
            )
            .is_err()
        );
    }

    #[test]
    fn sampling_matches_source_crossing_and_tie_rules() {
        let mut top_p = ZonosSamplingParams::greedy();
        top_p.temperature = 1.0;
        top_p.top_p = Some(0.8);
        top_p.min_p = None;
        let mut cursor = 0;
        // The first token crossing the nucleus boundary remains eligible.
        assert_eq!(
            sample_tokens(
                &[vec![2.0, 1.0, 0.0]],
                &[Vec::new()],
                &top_p,
                &[1.0, 1.0, 1.0],
                &mut cursor,
                8,
                false,
            )
            .unwrap(),
            vec![0]
        );

        let mut top_k = ZonosSamplingParams::greedy();
        top_k.temperature = 1.0;
        top_k.top_k = Some(2);
        let mut cursor = 0;
        // Both tied kth-probability candidates remain eligible; the
        // candidate-specific draw can select the second tied token.
        assert_eq!(
            sample_tokens(
                &[vec![2.0, 1.0, 1.0]],
                &[Vec::new()],
                &top_k,
                &[100.0, 1.0, 0.01],
                &mut cursor,
                8,
                false,
            )
            .unwrap(),
            vec![2]
        );

        let mut stochastic = ZonosSamplingParams::greedy();
        stochastic.temperature = 1.0;
        let mut cursor = 0;
        // Source multinomial draws one independent exponential variate per
        // vocabulary candidate; this can select a non-argmax token.
        assert_eq!(
            sample_tokens(
                &[vec![2.0, 1.0, 0.0]],
                &[Vec::new()],
                &stochastic,
                &[100.0, 0.01, 100.0],
                &mut cursor,
                8,
                false,
            )
            .unwrap(),
            vec![1]
        );
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
        // Zonos-v0.1 emits nine codebooks, matching the channel count
        // required by the complete DAC binder.
        assert_eq!(ZONOS_NUM_CODEBOOKS, 9);
    }

    // -----------------------------------------------------------------------
    // Gap-fill tests (sota-phase1 audit, 2026-07-24).
    //
    // These 10 tests close the untested `ZonosTts::with_dac` pub API (both
    // error branches + success path — the audit called this "0 of 3
    // covered"), the four `ZonosTts::new` codebook_embeddings / logit_heads
    // count-and-size branches (a converter regression would trip exactly
    // these), the two `ZonosTts::synthesize` NotImplemented branches
    // (no-DAC + real-weights-forward-not-landed), and the
    // `validate_for_forward` zero-size hparam branch. All feasible with
    // in-memory fixtures (no external checkpoint), deterministic, zero-dep.
    // -----------------------------------------------------------------------

    /// Builds a minimal [`DacCodecGguf`] from pub fields. [`ZonosTts::with_dac`]
    /// only inspects `attrs.n_codebooks` and `sample_rate`, so empty
    /// `tables` / `out_projs` are fine — the real decode chain is never
    /// reached from this scaffold's tests (and the audit's synthesize final
    /// arm is reached before decode runs).
    fn stub_dac(n_codebooks: usize, sample_rate: u32) -> DacCodecGguf {
        DacCodecGguf {
            attrs: vokra_ops::DacRvqAttrs {
                n_codebooks,
                codebook_size: 1,
                codebook_dim: 1,
                d_model: 1,
            },
            tables: Vec::new(),
            out_projs: Vec::new(),
            sample_rate,
            hop_length: 1,
        }
    }

    /// Pins the zero-size hparam arm of [`ZonosConfig::validate_for_forward`]
    /// (line 474). `is_well_formed` only inspects `backbone` fields, so a
    /// well-formed backbone paired with `num_codebooks == 0` reaches the
    /// zero-size guard rather than short-circuiting earlier. Also pin the
    /// `delay_pattern.len() == 0` case matching so the audit's line-474
    /// branch — not the line-489 delay-pattern-length branch — is what
    /// fires here (FR-EX-08 — the error message must name the zero-size
    /// hparam).
    #[test]
    fn config_zero_size_hparam_is_rejected() {
        let mut c = ZonosConfig::tiny_for_tests();
        c.num_codebooks = 0;
        // Match the length so we don't trip the delay_pattern.len() branch
        // (line 489) before the zero-size branch (line 474) fires.
        c.delay_pattern = Vec::new();
        match c.validate_for_forward() {
            Err(VokraError::InvalidArgument(msg)) => assert!(
                msg.contains("zero-size hparam"),
                "must name the zero-size hparam branch, got: {msg}"
            ),
            other => panic!("expected InvalidArgument(zero-size hparam), got {other:?}"),
        }
    }

    /// Pins line 730 of [`ZonosTts::new`]:
    /// `codebook_embeddings.len() != num_codebooks` — the count-side
    /// converter regression on the codebook embedding tables.
    #[test]
    fn zonos_tts_new_rejects_codebook_embeddings_count_mismatch() {
        let c = ZonosConfig::tiny_for_tests();
        let mut w = ZonosWeights::synthesized(&c, 7).expect("weights");
        w.codebook_embeddings.pop();
        match ZonosTts::new(c, w) {
            Err(VokraError::InvalidArgument(msg)) => assert!(
                msg.contains("codebook_embeddings.len()") && msg.contains("num_codebooks"),
                "must name codebook_embeddings count mismatch, got: {msg}"
            ),
            other => panic!("expected InvalidArgument(count mismatch), got {other:?}"),
        }
    }

    /// Pins line 738 of [`ZonosTts::new`]:
    /// `codebook_embeddings[i].len() != codebook_vocab * d_model` — the
    /// per-tensor size-side converter regression on a codebook embedding.
    #[test]
    fn zonos_tts_new_rejects_codebook_embeddings_size_mismatch() {
        let c = ZonosConfig::tiny_for_tests();
        let mut w = ZonosWeights::synthesized(&c, 7).expect("weights");
        // Count stays correct; one row is short by a single f32.
        w.codebook_embeddings[0].pop();
        match ZonosTts::new(c, w) {
            Err(VokraError::InvalidArgument(msg)) => assert!(
                msg.contains("codebook_embeddings[0].len()"),
                "must name the offending codebook_embeddings[i] size, got: {msg}"
            ),
            other => panic!("expected InvalidArgument(size mismatch), got {other:?}"),
        }
    }

    /// Pins line 779 of [`ZonosTts::new`]:
    /// `logit_heads.len() != num_codebooks` — the count-side converter
    /// regression on the per-codebook logit heads.
    #[test]
    fn zonos_tts_new_rejects_logit_heads_count_mismatch() {
        let c = ZonosConfig::tiny_for_tests();
        let mut w = ZonosWeights::synthesized(&c, 7).expect("weights");
        w.logit_heads.pop();
        match ZonosTts::new(c, w) {
            Err(VokraError::InvalidArgument(msg)) => assert!(
                msg.contains("logit_heads.len()") && msg.contains("num_codebooks"),
                "must name logit_heads count mismatch, got: {msg}"
            ),
            other => panic!("expected InvalidArgument(count mismatch), got {other:?}"),
        }
    }

    /// Pins line 787 of [`ZonosTts::new`]:
    /// `logit_heads[i].len() != d_model * head_vocab` — the per-tensor
    /// size-side converter regression on a logit head.
    #[test]
    fn zonos_tts_new_rejects_logit_heads_size_mismatch() {
        let c = ZonosConfig::tiny_for_tests();
        let mut w = ZonosWeights::synthesized(&c, 7).expect("weights");
        // Count stays correct; head 0 is short by one f32.
        w.logit_heads[0].pop();
        match ZonosTts::new(c, w) {
            Err(VokraError::InvalidArgument(msg)) => assert!(
                msg.contains("logit_heads[0].len()"),
                "must name the offending logit_heads[i] size, got: {msg}"
            ),
            other => panic!("expected InvalidArgument(size mismatch), got {other:?}"),
        }
    }

    /// Pins the happy path of [`ZonosTts::with_dac`]: a codec with
    /// `n_codebooks >= cfg.num_codebooks` and a matching sample rate binds
    /// successfully and becomes observable via [`ZonosTts::dac`]. The audit
    /// flagged the DAC-bind happy path as the primary observable slot that
    /// went untested (`dac()` returning `Some(...)` was never exercised).
    #[test]
    fn with_dac_happy_path_binds_dac() {
        let c = ZonosConfig::tiny_for_tests();
        let w = ZonosWeights::synthesized(&c, 7).expect("weights");
        let tts = ZonosTts::new(c.clone(), w).expect("zonos tts");
        assert!(tts.dac().is_none(), "sanity: no DAC before with_dac");
        let dac = stub_dac(c.num_codebooks, c.sample_rate);
        let tts = tts
            .with_dac_fixture(dac)
            .expect("with_dac fixture happy path");
        let bound = tts.dac_fixture.as_ref().expect("fixture DAC must be bound");
        assert_eq!(bound.attrs.n_codebooks, c.num_codebooks);
        assert_eq!(bound.sample_rate, c.sample_rate);
    }

    /// Pins line 819 of [`ZonosTts::with_dac`]:
    /// `dac.attrs.n_codebooks < cfg.num_codebooks` — the channel-misroute
    /// guard. The module docstring (line 810-811) explicitly warns that a
    /// codebook shortfall "would misroute channel indices at decode time";
    /// FR-EX-08 requires this to fail loud rather than silently truncate.
    #[test]
    fn with_dac_rejects_codebook_shortfall() {
        let c = ZonosConfig::tiny_for_tests();
        let short = c.num_codebooks - 1;
        let w = ZonosWeights::synthesized(&c, 7).expect("weights");
        let tts = ZonosTts::new(c.clone(), w).expect("zonos tts");
        let dac = stub_dac(short, c.sample_rate);
        match tts.with_dac_fixture(dac) {
            Err(VokraError::InvalidArgument(msg)) => assert!(
                msg.contains("codebooks") && msg.contains("channels"),
                "must name codebook / channel mismatch, got: {msg}"
            ),
            other => panic!("expected InvalidArgument(codebook shortfall), got {other:?}"),
        }
    }

    /// Pins line 824 of [`ZonosTts::with_dac`]:
    /// `dac.sample_rate != cfg.sample_rate` — the 44.1 kHz DAC binding
    /// guard. Zonos-v0.1 is explicitly bound to descript/dac_44khz
    /// upstream; a DAC with a different sample rate is a load-time bug
    /// (FR-EX-08 — never a silent resample).
    #[test]
    fn with_dac_rejects_sample_rate_mismatch() {
        let c = ZonosConfig::tiny_for_tests();
        let w = ZonosWeights::synthesized(&c, 7).expect("weights");
        let tts = ZonosTts::new(c.clone(), w).expect("zonos tts");
        // Pair matching codebooks with a 24 kHz DAC to isolate this branch
        // from the codebook-shortfall guard.
        let dac = stub_dac(c.num_codebooks, 24_000);
        match tts.with_dac_fixture(dac) {
            Err(VokraError::InvalidArgument(msg)) => assert!(
                msg.contains("sample_rate") && msg.contains("dac_44khz"),
                "must name sample_rate + dac_44khz binding, got: {msg}"
            ),
            other => panic!("expected InvalidArgument(sample_rate mismatch), got {other:?}"),
        }
    }

    /// Pins line 904 of [`ZonosTts::synthesize`]: the no-DAC-bound arm. It
    /// is unreachable via [`ZonosWeights::synthesized`] (which short-circuits
    /// at the synthesized-weight guard) but reachable inside this module by
    /// flipping the private `is_synthesized` flag, which is the shape a
    /// real-checkpoint bind path will take. Message must name the DAC blocker + `with_dac`
    /// call so callers know how to unblock — FR-EX-08, never a silent
    /// zero-fill.
    #[test]
    fn synthesize_without_dac_is_loud_not_implemented() {
        let c = ZonosConfig::tiny_for_tests();
        let mut w = ZonosWeights::synthesized(&c, 7).expect("weights");
        // Pretend a real checkpoint so we skip the synthesized-weight arm.
        w.is_synthesized = false;
        let tts = ZonosTts::new(c, w).expect("zonos tts");
        assert!(tts.dac().is_none(), "sanity: no DAC bound");
        match tts.synthesize(&[0, 1, 2]).unwrap_err() {
            VokraError::NotImplemented(msg) => assert!(
                msg.contains("DAC") && msg.contains("with_dac"),
                "message must name DAC blocker + with_dac call, got: {msg}"
            ),
            other => panic!("expected NotImplemented(no DAC), got {other:?}"),
        }
    }

    /// Pins the raw-input refusal in [`ZonosTts::synthesize`]. Reachable by
    /// flipping
    /// `is_synthesized = false` **and** binding a DAC via
    /// [`ZonosTts::with_dac`]. The message must direct callers to the
    /// authenticated packet API so raw controls cannot be silently omitted.
    #[test]
    fn synthesize_real_weights_with_dac_refuses_raw_controls() {
        let c = ZonosConfig::tiny_for_tests();
        let mut w = ZonosWeights::synthesized(&c, 7).expect("weights");
        w.is_synthesized = false;
        let tts = ZonosTts::new(c.clone(), w).expect("zonos tts");
        let dac = stub_dac(c.num_codebooks, c.sample_rate);
        let tts = tts
            .with_dac_fixture(dac)
            .expect("with_dac fixture happy path");
        match tts.synthesize(&[0, 1, 2]).unwrap_err() {
            VokraError::NotImplemented(msg) => assert!(
                msg.contains("raw phoneme-only") && msg.contains("conditioning_packet"),
                "message must direct callers to authenticated packet conditioning, got: {msg}"
            ),
            other => panic!("expected NotImplemented(raw controls), got {other:?}"),
        }
    }

    #[test]
    fn conditioning_packet_is_versioned_and_digest_bound() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(ZonosConditioningPacket::MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend(
            std::iter::repeat(0.0f32.to_le_bytes())
                .flatten()
                .take(128 * 4),
        );
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend(
            std::iter::repeat(0.0f32.to_le_bytes())
                .flatten()
                .take(7 * 4),
        );
        for value in [24_000.0_f32, 400.0, 1.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&(-1i32).to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&16u32.to_le_bytes());
        let digest_offset = bytes.len();
        bytes.extend([0u8; 32]);
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&42u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend(
            std::iter::repeat(0.0f32.to_le_bytes())
                .flatten()
                .take(16 * 4),
        );
        bytes.extend(
            std::iter::repeat(1.0f32.to_le_bytes())
                .flatten()
                .take(16 * 4),
        );
        let digest = packet_digest(&bytes, digest_offset);
        bytes[digest_offset..digest_offset + 32].copy_from_slice(&digest);
        let packet = ZonosConditioningPacket::parse(&bytes, digest, 16).unwrap();
        assert_eq!(packet.phoneme_ids, vec![2, 42, 3]);
        assert!(ZonosConditioningPacket::parse(&bytes, [8u8; 32], 16).is_err());
        bytes[digest_offset + 100] ^= 1;
        assert!(ZonosConditioningPacket::parse(&bytes, digest, 16).is_err());
    }

    #[test]
    fn delay_and_greedy_step_enforce_codebook_contract() {
        let config = ZonosConfig::tiny_for_tests();
        let delayed = config
            .apply_delay_pattern(&[vec![1, 2], vec![3, 4], vec![5, 6]])
            .unwrap();
        assert_eq!(
            delayed[0],
            vec![
                config.masked_token_id,
                1,
                2,
                config.masked_token_id,
                config.masked_token_id
            ]
        );
        assert_eq!(
            delayed[2],
            vec![
                config.masked_token_id,
                config.masked_token_id,
                config.masked_token_id,
                5,
                6
            ]
        );
        assert_eq!(
            config.revert_delay_pattern(&delayed, 2).unwrap(),
            vec![vec![1, 2], vec![3, 4], vec![5, 6]]
        );
        let mut early_stop = delayed.clone();
        early_stop[0][2] = UNKNOWN_GENERATION_TOKEN;
        assert!(config.revert_delay_pattern(&early_stop, 2).is_err());
        let mut terminal = delayed;
        terminal[0][2] = config.eos_token_id;
        terminal[1][2] = config.masked_token_id;
        terminal[2][3] = UNKNOWN_GENERATION_TOKEN;
        assert!(config.revert_delay_pattern(&terminal, 2).is_err());
        let sanitized = revert_generation_delay(&config, &terminal, 2, 2).unwrap();
        assert!(
            sanitized
                .iter()
                .flatten()
                .all(|&token| token < config.eos_token_id)
        );
        assert_eq!(sanitized, vec![vec![1, 0], vec![0, 4], vec![0, 6]]);
        let mut logits = vec![vec![0.0; config.head_vocab]; config.num_codebooks];
        logits[0][config.eos_token_id as usize] = 10.0;
        logits[1][config.eos_token_id as usize] = 10.0;
        let step = config.greedy_step(&logits).unwrap();
        assert_eq!(step[0], config.eos_token_id);
        assert_ne!(step[1], config.eos_token_id);
    }

    #[test]
    fn terminal_drain_writes_all_nine_columns_in_delayed_space() {
        let mut delayed = vec![vec![UNKNOWN_GENERATION_TOKEN; 12]; 9];
        fill_terminal_drain(&mut delayed, 1, 1025, Some((1024, 9))).unwrap();
        for column in 1..10 {
            let active = column - 1;
            for (codebook, row) in delayed.iter().enumerate() {
                if codebook == active {
                    assert_eq!(row[column], 1024);
                } else if codebook > active {
                    assert_eq!(row[column], UNKNOWN_GENERATION_TOKEN);
                } else {
                    assert_eq!(row[column], 1025);
                }
            }
        }
        assert_eq!(delayed[0][0], UNKNOWN_GENERATION_TOKEN);
    }

    #[test]
    fn stopping_frame_masks_prefix_but_keeps_later_sampled_codebooks() {
        // The first prefill sample is written without invoking this stopping
        // rewrite; CB0 EOS there therefore remains an ordinary first frame.
        assert_eq!(
            apply_stopping_frame(&[1024, 11, 12], false, 0, 1025, 1024).unwrap(),
            vec![1024, 11, 12]
        );
        assert_eq!(
            apply_stopping_frame(&[10, 11, 12, 13, 14], true, 2, 1025, 1024).unwrap(),
            vec![1025, 1025, 1024, 13, 14]
        );
        assert_eq!(
            apply_stopping_frame(&[10, 11, 12], false, 0, 1025, 1024).unwrap(),
            vec![10, 11, 12]
        );
    }

    #[test]
    fn generation_delay_layout_has_fixed_masks_and_unknown_slots() {
        let config = ZonosConfig::tiny_for_tests();
        let no_prompt = initialize_generation_delay(&config, &[], 2).unwrap();
        assert_eq!(
            no_prompt,
            vec![
                vec![9, UNKNOWN_GENERATION_TOKEN, UNKNOWN_GENERATION_TOKEN, 9, 9],
                vec![9, 9, UNKNOWN_GENERATION_TOKEN, UNKNOWN_GENERATION_TOKEN, 9],
                vec![9, 9, 9, UNKNOWN_GENERATION_TOKEN, UNKNOWN_GENERATION_TOKEN],
            ]
        );
        let prompt =
            initialize_generation_delay(&config, &[vec![1, 2], vec![3, 4], vec![5, 6]], 2).unwrap();
        assert_eq!(
            prompt[0],
            vec![
                9,
                1,
                2,
                UNKNOWN_GENERATION_TOKEN,
                UNKNOWN_GENERATION_TOKEN,
                9,
                9
            ]
        );
        assert_eq!(
            prompt[1],
            vec![
                9,
                9,
                3,
                4,
                UNKNOWN_GENERATION_TOKEN,
                UNKNOWN_GENERATION_TOKEN,
                9
            ]
        );
        assert_eq!(
            prompt[2],
            vec![
                9,
                9,
                9,
                5,
                6,
                UNKNOWN_GENERATION_TOKEN,
                UNKNOWN_GENERATION_TOKEN
            ]
        );
        let mut scattered = prompt;
        scatter_unknown_tokens(&mut scattered, 4, &[7, 8, 6]).unwrap();
        assert_eq!(scattered[0][4], 7, "unknown slot must accept a sample");
        assert_eq!(scattered[1][4], 8, "unknown slot must accept a sample");
        assert_eq!(scattered[2][4], 6, "prompt value must not be overwritten");
    }
}
