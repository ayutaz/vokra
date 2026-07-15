//! CSM configuration resolved from the `vokra.csm.*` GGUF chunk group
//! (M4-05-T04 chunk design / T06 config type).
//!
//! Every field is **read from the GGUF** — nothing is hard-coded (CLAUDE.md
//! ハルシネーション厳禁; FR-LD-02 / FR-MD-02). The converter
//! (`vokra-convert::models::csm`) derives the numeric hparams from the
//! upstream checkpoint tensor shapes plus the `SesameAILabs/csm` `models.py`
//! constants recorded in ADR M4-05 §D2 / §D9; the runtime only consumes them
//! here.
//!
//! # `0`-placeholder posture (cosyvoice2 precedent)
//!
//! A key that is **absent** reads as `0` (or the documented model-card
//! default for ε / RoPE base); a key that is present with the wrong type is
//! a loud [`VokraError::InvalidArgument`] (FR-EX-08). The forward paths
//! call [`CsmConfig::validate_for_forward`] before running so a
//! shape-only converter GGUF fails loudly at first use — never a silent
//! zero-shape forward.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

// --- `vokra.csm.*` metadata keys (ADR M4-05 §D9) ---------------------------
//
// Duplicated verbatim in `vokra-convert::models::csm` (the two crates only
// share `vokra-core`; a round-trip test on the converter side catches
// drift — the cosyvoice2 / kokoro pattern).

pub(crate) const KEY_SAMPLE_RATE: &str = "vokra.csm.sample_rate";
pub(crate) const KEY_FRAME_RATE_MHZ: &str = "vokra.csm.frame_rate_mhz";

pub(crate) const KEY_BB_N_LAYER: &str = "vokra.csm.arch.backbone.n_layer";
pub(crate) const KEY_BB_D_MODEL: &str = "vokra.csm.arch.backbone.d_model";
pub(crate) const KEY_BB_N_HEAD_Q: &str = "vokra.csm.arch.backbone.n_head_q";
pub(crate) const KEY_BB_N_HEAD_KV: &str = "vokra.csm.arch.backbone.n_head_kv";
pub(crate) const KEY_BB_FFN_DIM: &str = "vokra.csm.arch.backbone.ffn_dim";

pub(crate) const KEY_DT_N_LAYER: &str = "vokra.csm.arch.depth.n_layer";
pub(crate) const KEY_DT_D_MODEL: &str = "vokra.csm.arch.depth.d_model";
pub(crate) const KEY_DT_N_HEAD_Q: &str = "vokra.csm.arch.depth.n_head_q";
pub(crate) const KEY_DT_N_HEAD_KV: &str = "vokra.csm.arch.depth.n_head_kv";
pub(crate) const KEY_DT_FFN_DIM: &str = "vokra.csm.arch.depth.ffn_dim";

pub(crate) const KEY_RMS_NORM_EPS: &str = "vokra.csm.arch.rms_norm_eps";
pub(crate) const KEY_ROPE_BASE: &str = "vokra.csm.arch.rope_base";
pub(crate) const KEY_N_CTX: &str = "vokra.csm.arch.n_ctx";

pub(crate) const KEY_ROPE_SCALE_FACTOR: &str = "vokra.csm.rope.scale_factor";
pub(crate) const KEY_ROPE_LOW_FREQ_FACTOR: &str = "vokra.csm.rope.low_freq_factor";
pub(crate) const KEY_ROPE_HIGH_FREQ_FACTOR: &str = "vokra.csm.rope.high_freq_factor";
pub(crate) const KEY_ROPE_OLD_CONTEXT_LEN: &str = "vokra.csm.rope.old_context_len";

pub(crate) const KEY_AUDIO_N_CODEBOOKS: &str = "vokra.csm.audio.n_codebooks";
pub(crate) const KEY_AUDIO_VOCAB_SIZE: &str = "vokra.csm.audio.vocab_size";
pub(crate) const KEY_TEXT_VOCAB_SIZE: &str = "vokra.csm.text.vocab_size";

/// Safety-net RMSNorm ε used **only** when the GGUF omits the key. The
/// upstream `SesameAILabs/csm` `models.py` ships `norm_eps=1e-5` on both
/// flavors (`llama3_2_1B` / `llama3_2_100M`) — ADR M4-05 §D2.
pub const DEFAULT_CSM_RMS_NORM_EPS: f32 = 1e-5;

/// Safety-net RoPE base θ used **only** when the GGUF omits the key.
/// `models.py` ships `rope_base=500_000` on both flavors (ADR M4-05 §D2).
pub const DEFAULT_CSM_ROPE_BASE: f32 = 500_000.0;

/// One transformer stack's hparams (backbone or depth transformer — the two
/// stacks share the Llama-3.2 block recipe and differ only in dims; ADR
/// M4-05 §D2).
#[derive(Debug, Clone, PartialEq)]
pub struct CsmTransformerConfig {
    /// Transformer block count.
    pub n_layer: usize,
    /// Hidden width (`d`).
    pub d_model: usize,
    /// Query attention heads.
    pub n_head_q: usize,
    /// Key/value attention heads (GQA — `n_head_q % n_head_kv == 0`).
    pub n_head_kv: usize,
    /// SwiGLU FFN inner width.
    pub ffn_dim: usize,
}

impl CsmTransformerConfig {
    /// Per-query-head width (`d_model / n_head_q`); `0` when `n_head_q == 0`
    /// (shape-only converter sentinel) so shape checks never panic.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.d_model.checked_div(self.n_head_q).unwrap_or(0)
    }

    /// KV width `n_head_kv * head_dim` (GQA broadcast); `0` on any zero
    /// component.
    #[must_use]
    pub fn kv_hidden_dim(&self) -> usize {
        self.n_head_kv.saturating_mul(self.head_dim())
    }

    /// GQA algebraic constraint: `n_head_q % n_head_kv == 0` and
    /// `d_model % n_head_q == 0`, all non-zero.
    #[must_use]
    pub fn is_gqa_well_formed(&self) -> bool {
        self.n_head_q != 0
            && self.n_head_kv != 0
            && self.n_head_q % self.n_head_kv == 0
            && self.d_model % self.n_head_q == 0
    }
}

/// Llama-3 scaled-RoPE parameters (torchtune `Llama3ScaledRoPE`, ADR M4-05
/// §D3). Present iff `vokra.csm.rope.scale_factor > 0`; absent (or `0`)
/// selects the plain unscaled RoPE.
#[derive(Debug, Clone, PartialEq)]
pub struct CsmRopeScaling {
    /// Low-frequency division factor (`scale_factor=32` for CSM-1B —
    /// `models.py`).
    pub scale_factor: f32,
    /// torchtune default `1.0` (converter writes it; ADR §D3).
    pub low_freq_factor: f32,
    /// torchtune default `4.0`.
    pub high_freq_factor: f32,
    /// torchtune default `8192`.
    pub old_context_len: usize,
}

/// The resolved CSM hparam snapshot (`vokra.csm.*` chunk group).
#[derive(Debug, Clone, PartialEq)]
pub struct CsmConfig {
    /// Output PCM sample rate (Hz). Mimi native = 24 000 (ADR §D2).
    pub sample_rate: u32,
    /// Audio frame rate in milli-Hz (12.5 Hz → `12_500`) — integer anchoring,
    /// no F32 drift (ADR §D9).
    pub frame_rate_mhz: u32,
    /// Backbone stack (`llama3_2_1B` flavor for the real checkpoint).
    pub backbone: CsmTransformerConfig,
    /// Depth transformer stack (`llama3_2_100M` flavor — upstream呼称
    /// `decoder`; ADR §D1-(e)).
    pub depth: CsmTransformerConfig,
    /// RMSNorm ε (both stacks — `models.py` ships one value).
    pub rms_norm_eps: f32,
    /// RoPE base θ (both stacks).
    pub rope_base: f32,
    /// Max backbone sequence length (frame positions). `0` placeholder is
    /// rejected by [`Self::validate_for_forward`].
    pub n_ctx: usize,
    /// Llama-3 scaled RoPE (None = plain RoPE).
    pub rope_scaling: Option<CsmRopeScaling>,
    /// Mimi RVQ codebooks per frame (32 for CSM-1B).
    pub n_codebooks: usize,
    /// Per-codebook audio token vocab (shape-driven from the checkpoint —
    /// ADR §D2 "T29 待ち"; includes any special ids above the Mimi bins).
    pub audio_vocab_size: usize,
    /// Text token vocab (Llama-3.2 tokenizer table size, shape-driven).
    pub text_vocab_size: usize,
}

impl CsmConfig {
    /// Reads the CSM hparams from a CSM GGUF. Missing numeric keys read as
    /// `0` placeholders (except ε / RoPE base which take the documented
    /// `models.py` defaults); wrong-typed keys are loud errors (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if any present key has the wrong
    /// metadata type.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let backbone = CsmTransformerConfig {
            n_layer: read_u32_or_zero(file, KEY_BB_N_LAYER)? as usize,
            d_model: read_u32_or_zero(file, KEY_BB_D_MODEL)? as usize,
            n_head_q: read_u32_or_zero(file, KEY_BB_N_HEAD_Q)? as usize,
            n_head_kv: read_u32_or_zero(file, KEY_BB_N_HEAD_KV)? as usize,
            ffn_dim: read_u32_or_zero(file, KEY_BB_FFN_DIM)? as usize,
        };
        let depth = CsmTransformerConfig {
            n_layer: read_u32_or_zero(file, KEY_DT_N_LAYER)? as usize,
            d_model: read_u32_or_zero(file, KEY_DT_D_MODEL)? as usize,
            n_head_q: read_u32_or_zero(file, KEY_DT_N_HEAD_Q)? as usize,
            n_head_kv: read_u32_or_zero(file, KEY_DT_N_HEAD_KV)? as usize,
            ffn_dim: read_u32_or_zero(file, KEY_DT_FFN_DIM)? as usize,
        };
        let scale_factor = read_f32_or(file, KEY_ROPE_SCALE_FACTOR, 0.0)?;
        let rope_scaling = if scale_factor > 0.0 {
            Some(CsmRopeScaling {
                scale_factor,
                low_freq_factor: read_f32_or(file, KEY_ROPE_LOW_FREQ_FACTOR, 1.0)?,
                high_freq_factor: read_f32_or(file, KEY_ROPE_HIGH_FREQ_FACTOR, 4.0)?,
                old_context_len: read_u32_or_zero(file, KEY_ROPE_OLD_CONTEXT_LEN)? as usize,
            })
        } else {
            None
        };
        Ok(Self {
            sample_rate: read_u32_or_zero(file, KEY_SAMPLE_RATE)?,
            frame_rate_mhz: read_u32_or_zero(file, KEY_FRAME_RATE_MHZ)?,
            backbone,
            depth,
            rms_norm_eps: read_f32_or(file, KEY_RMS_NORM_EPS, DEFAULT_CSM_RMS_NORM_EPS)?,
            rope_base: read_f32_or(file, KEY_ROPE_BASE, DEFAULT_CSM_ROPE_BASE)?,
            n_ctx: read_u32_or_zero(file, KEY_N_CTX)? as usize,
            rope_scaling,
            n_codebooks: read_u32_or_zero(file, KEY_AUDIO_N_CODEBOOKS)? as usize,
            audio_vocab_size: read_u32_or_zero(file, KEY_AUDIO_VOCAB_SIZE)? as usize,
            text_vocab_size: read_u32_or_zero(file, KEY_TEXT_VOCAB_SIZE)? as usize,
        })
    }

    /// Rejects `0`-placeholder / GQA-ill-formed configs before any forward
    /// runs (FR-EX-08 — the shape-only converter path fails loudly here, not
    /// deep inside a GEMM).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        for (stack, cfg) in [("backbone", &self.backbone), ("depth", &self.depth)] {
            if !cfg.is_gqa_well_formed() {
                return Err(VokraError::InvalidArgument(format!(
                    "csm config: {stack} not GQA well-formed (n_layer={}, d_model={}, \
                     n_head_q={}, n_head_kv={}, ffn_dim={})",
                    cfg.n_layer, cfg.d_model, cfg.n_head_q, cfg.n_head_kv, cfg.ffn_dim
                )));
            }
            if cfg.n_layer == 0 || cfg.ffn_dim == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "csm config: {stack} carries a 0-placeholder (n_layer={}, ffn_dim={}) — \
                     re-convert with real hparams (T04) or use CsmConfig::tiny_for_tests",
                    cfg.n_layer, cfg.ffn_dim
                )));
            }
            if cfg.head_dim() % 2 != 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "csm config: {stack} head_dim {} must be even (RoPE pairs)",
                    cfg.head_dim()
                )));
            }
        }
        if self.n_ctx == 0 {
            return Err(VokraError::InvalidArgument(
                "csm config: n_ctx = 0 placeholder — no forward can bound its KV reserve".into(),
            ));
        }
        if self.n_codebooks == 0 || self.audio_vocab_size == 0 || self.text_vocab_size == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "csm config: zero-size vocab hparam (n_codebooks={}, audio_vocab={}, \
                 text_vocab={})",
                self.n_codebooks, self.audio_vocab_size, self.text_vocab_size
            )));
        }
        if let Some(s) = &self.rope_scaling {
            if s.old_context_len == 0 || s.high_freq_factor <= s.low_freq_factor {
                return Err(VokraError::InvalidArgument(format!(
                    "csm config: rope scaling ill-formed (old_context_len={}, low={}, high={})",
                    s.old_context_len, s.low_freq_factor, s.high_freq_factor
                )));
            }
        }
        Ok(())
    }

    /// PCM samples per audio frame: `sample_rate / frame_rate`. The division
    /// must be exact (`sample_rate * 1000 % frame_rate_mhz == 0`) — Mimi's
    /// 24 000 / 12.5 = 1920 is; anything else is a converter bug surfaced
    /// loudly (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a zero or non-exact rate pair.
    pub fn frame_hop_samples(&self) -> Result<usize> {
        if self.sample_rate == 0 || self.frame_rate_mhz == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "csm config: sample_rate={} / frame_rate_mhz={} — both must be > 0",
                self.sample_rate, self.frame_rate_mhz
            )));
        }
        let num = self.sample_rate as u64 * 1000;
        let den = self.frame_rate_mhz as u64;
        if num % den != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "csm config: sample_rate {} not an integer multiple of frame rate \
                 ({} mHz) — frame hop would not be sample-exact",
                self.sample_rate, self.frame_rate_mhz
            )));
        }
        Ok((num / den) as usize)
    }

    /// A miniature, GQA-well-formed config for synthesized-weight tests and
    /// the host-only CLI smoke fixture. Dims are deliberately tiny so the
    /// whole frame loop runs in milliseconds; the *shape relationships*
    /// (GQA split, even head_dim, 12.5 Hz-exact hop) mirror the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            sample_rate: 24_000,
            frame_rate_mhz: 12_500,
            backbone: CsmTransformerConfig {
                n_layer: 2,
                d_model: 16,
                n_head_q: 4,
                n_head_kv: 2,
                ffn_dim: 32,
            },
            depth: CsmTransformerConfig {
                n_layer: 2,
                d_model: 8,
                n_head_q: 2,
                n_head_kv: 1,
                ffn_dim: 16,
            },
            rms_norm_eps: DEFAULT_CSM_RMS_NORM_EPS,
            rope_base: DEFAULT_CSM_ROPE_BASE,
            n_ctx: 64,
            rope_scaling: Some(CsmRopeScaling {
                scale_factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                old_context_len: 8192,
            }),
            n_codebooks: 4,
            audio_vocab_size: 11,
            text_vocab_size: 13,
        }
    }
}

fn read_u32_or_zero(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(v)) => Ok(*v),
        None => Ok(0),
        Some(other) => Err(VokraError::InvalidArgument(format!(
            "csm config: `{key}` is not a UINT32 (got {:?})",
            other.value_type()
        ))),
    }
}

fn read_f32_or(file: &GgufFile, key: &str, default: f32) -> Result<f32> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(v)) => Ok(*v),
        None => Ok(default),
        Some(other) => Err(VokraError::InvalidArgument(format!(
            "csm config: `{key}` is not a FLOAT32 (got {:?})",
            other.value_type()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    fn gguf_with(entries: &[(&str, GgufMetadataValue)]) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "csm");
        for (k, v) in entries {
            b.add_metadata(k, v.clone());
        }
        // GGUF requires at least the header; a tensor is not needed for
        // metadata-only parsing.
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    #[test]
    fn missing_keys_read_as_placeholders_and_documented_defaults() {
        let file = gguf_with(&[]);
        let cfg = CsmConfig::from_gguf(&file).expect("from_gguf");
        assert_eq!(cfg.backbone.n_layer, 0);
        assert_eq!(cfg.n_ctx, 0);
        assert_eq!(cfg.rms_norm_eps, DEFAULT_CSM_RMS_NORM_EPS);
        assert_eq!(cfg.rope_base, DEFAULT_CSM_ROPE_BASE);
        assert!(
            cfg.rope_scaling.is_none(),
            "scale_factor absent → no scaling"
        );
        // And the placeholder config refuses to run (FR-EX-08).
        assert!(matches!(
            cfg.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn wrong_typed_key_is_loud() {
        let file = gguf_with(&[(KEY_BB_N_LAYER, GgufMetadataValue::String("x".into()))]);
        assert!(matches!(
            CsmConfig::from_gguf(&file),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn round_trip_of_fully_populated_chunk_group() {
        let t = CsmConfig::tiny_for_tests();
        let file = gguf_with(&[
            (KEY_SAMPLE_RATE, GgufMetadataValue::U32(t.sample_rate)),
            (KEY_FRAME_RATE_MHZ, GgufMetadataValue::U32(t.frame_rate_mhz)),
            (
                KEY_BB_N_LAYER,
                GgufMetadataValue::U32(t.backbone.n_layer as u32),
            ),
            (
                KEY_BB_D_MODEL,
                GgufMetadataValue::U32(t.backbone.d_model as u32),
            ),
            (
                KEY_BB_N_HEAD_Q,
                GgufMetadataValue::U32(t.backbone.n_head_q as u32),
            ),
            (
                KEY_BB_N_HEAD_KV,
                GgufMetadataValue::U32(t.backbone.n_head_kv as u32),
            ),
            (
                KEY_BB_FFN_DIM,
                GgufMetadataValue::U32(t.backbone.ffn_dim as u32),
            ),
            (
                KEY_DT_N_LAYER,
                GgufMetadataValue::U32(t.depth.n_layer as u32),
            ),
            (
                KEY_DT_D_MODEL,
                GgufMetadataValue::U32(t.depth.d_model as u32),
            ),
            (
                KEY_DT_N_HEAD_Q,
                GgufMetadataValue::U32(t.depth.n_head_q as u32),
            ),
            (
                KEY_DT_N_HEAD_KV,
                GgufMetadataValue::U32(t.depth.n_head_kv as u32),
            ),
            (
                KEY_DT_FFN_DIM,
                GgufMetadataValue::U32(t.depth.ffn_dim as u32),
            ),
            (KEY_RMS_NORM_EPS, GgufMetadataValue::F32(t.rms_norm_eps)),
            (KEY_ROPE_BASE, GgufMetadataValue::F32(t.rope_base)),
            (KEY_N_CTX, GgufMetadataValue::U32(t.n_ctx as u32)),
            (KEY_ROPE_SCALE_FACTOR, GgufMetadataValue::F32(32.0)),
            (KEY_ROPE_LOW_FREQ_FACTOR, GgufMetadataValue::F32(1.0)),
            (KEY_ROPE_HIGH_FREQ_FACTOR, GgufMetadataValue::F32(4.0)),
            (KEY_ROPE_OLD_CONTEXT_LEN, GgufMetadataValue::U32(8192)),
            (
                KEY_AUDIO_N_CODEBOOKS,
                GgufMetadataValue::U32(t.n_codebooks as u32),
            ),
            (
                KEY_AUDIO_VOCAB_SIZE,
                GgufMetadataValue::U32(t.audio_vocab_size as u32),
            ),
            (
                KEY_TEXT_VOCAB_SIZE,
                GgufMetadataValue::U32(t.text_vocab_size as u32),
            ),
        ]);
        let cfg = CsmConfig::from_gguf(&file).expect("from_gguf");
        assert_eq!(cfg, t);
        cfg.validate_for_forward()
            .expect("tiny config is well-formed");
        assert_eq!(
            cfg.frame_hop_samples().expect("hop"),
            1920,
            "24 kHz / 12.5 Hz"
        );
    }

    #[test]
    fn non_exact_frame_hop_is_rejected() {
        let mut cfg = CsmConfig::tiny_for_tests();
        cfg.frame_rate_mhz = 12_700; // 24 000 000 / 12 700 is not integral
        assert!(matches!(
            cfg.frame_hop_samples(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn gqa_ill_formed_is_rejected() {
        let mut cfg = CsmConfig::tiny_for_tests();
        cfg.backbone.n_head_kv = 3; // 4 % 3 != 0
        assert!(matches!(
            cfg.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn odd_head_dim_is_rejected() {
        let mut cfg = CsmConfig::tiny_for_tests();
        // d_model 12 / n_head_q 4 = head_dim 3 (odd) — RoPE pairs need even.
        cfg.backbone.d_model = 12;
        assert!(matches!(
            cfg.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
