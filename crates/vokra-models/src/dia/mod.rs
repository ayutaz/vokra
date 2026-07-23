//! Dia-1.6B — nari-labs' text-to-dialog TTS (SoTA plan Phase 1-4, 2026-07-24).
//!
//! # What Dia is (primary source)
//!
//! Dia is a 1.6B-parameter text-to-dialog TTS model published by nari-labs
//! (Apache 2.0 code + weight). Architecture per
//! `huggingface.co/nari-labs/Dia-1.6B/config.json` (fetched verbatim into
//! this module — CLAUDE.md「ハルシネーション厳禁」):
//!
//! - **Text encoder** (`model.encoder`): `n_layer=12`, `n_embd=1024`,
//!   `n_head=16`, `head_dim=128`, `n_hidden=4096` (SwiGLU FFN inner width),
//!   source vocab = `256` (byte-level input).
//! - **Decoder** (`model.decoder`): `n_layer=18`, `n_embd=2048`,
//!   GQA `n_head_q=16`, `n_head_kv=4`, `head_dim=128`,
//!   `n_hidden=8192`; cross-attention `n_head_q=16` / `head_dim=128`.
//! - **Delay pattern** (`data.delay_pattern`): `[0, 8, 9, 10, 11, 12, 13, 14, 15]`
//!   — one delay per audio channel (9 channels total, `data.channels=9`);
//!   channel 0 is unshifted, channels 1..8 are staggered.
//! - **Special ids** (`data`): `audio_bos_value=1026`, `audio_eos_value=1024`,
//!   `audio_pad_value=1025` in a target vocab of `1028`
//!   (`model.tgt_vocab_size`); source-side `text_pad_value=0`.
//! - **RoPE**: `rope_max_timescale=10000`, `rope_min_timescale=1`.
//! - **RMSNorm ε**: `1e-5` (`model.normalization_layer_epsilon`).
//!
//! # Terminal codec (upstream primary source)
//!
//! Dia decodes to PCM via **DAC 44.1 kHz** (`descript-audio-codec` — Descript's
//! open MIT codec) fetched by upstream `dia/model.py::_load_dac_model` through
//! `dac.utils.download()`. The `data.channels=9` shape lines up 1:1 with DAC's
//! 9-codebook RVQ frames at 44.1 kHz. In Vokra, DAC lives in
//! `vokra-ops::dac_rvq` + `vokra-models::codec::DacCodecGguf` — a caller with a
//! DAC GGUF injects it via [`DiaTts::with_dac`]. Until then
//! [`DiaTts::synthesize`] returns [`VokraError::NotImplemented`] naming the
//! blocker (FR-EX-08 — never a silent zero-fill).
//!
//! # What lands in this Phase 1-4 slice
//!
//! - [`DiaConfig`] — every hparam transcribed from the primary source (no
//!   hardcoded fabrication; sample-rate is inherited from DAC 44.1 kHz per
//!   upstream `_load_dac_model`, documented on the field).
//! - [`DiaWeights`] — a text-encoder + decoder weight store with a
//!   deterministic [`DiaWeights::synthesized`] fixture (SplitMix64 + Xavier)
//!   so shape / dtype / size flow can be exercised without the real HF
//!   checkpoint.
//! - [`DiaTts`] — engine handle carrying config + weights + optional DAC bind.
//!   [`DiaTts::synthesize`] returns [`VokraError::NotImplemented`] until real
//!   weights are bound (the real forward — encoder embed → per-layer prenorm
//!   attn/FFN → decoder channel-embed sum → delayed AR sampling per channel →
//!   DAC decode → PCM — is a follow-up wave gated on the real-checkpoint
//!   tensor manifest).
//!
//! Real-checkpoint parity is deferred exactly like CosyVoice2 T02 / CSM T29:
//! this scaffold sets the seam so the follow-up lands drop-in.

use vokra_core::rng::SplitMix64;
use vokra_core::{Result, VokraError};

use crate::codec::DacCodecGguf;

/// `vokra.model.arch` a Dia GGUF must carry. Written by
/// `vokra-convert::models::dia::ARCH`; the compliance registry
/// (`vokra_core::compliance`) knows `dia` / `dia-1.6b` as `Permissive`
/// (Apache 2.0 code + weight), so a stock Dia GGUF passes the M2-13 gate
/// without a research flag.
pub const EXPECTED_ARCH: &str = "dia";

/// PCM sample rate Dia emits. Not written in the upstream `config.json`;
/// inherited from **DAC 44.1 kHz** (the codec `_load_dac_model` fetches via
/// `dac.utils.download()`).
pub const DIA_SAMPLE_RATE: u32 = 44_100;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Text-encoder hparams (primary source: `config.json` `model.encoder`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiaEncoderConfig {
    /// `model.encoder.n_layer` — 12 for Dia-1.6B.
    pub n_layer: usize,
    /// `model.encoder.n_embd` — hidden width, 1024.
    pub n_embd: usize,
    /// `model.encoder.n_head` — attention heads, 16 (MHA, `n_head_kv=n_head`).
    pub n_head: usize,
    /// `model.encoder.head_dim` — 128 (so `n_embd = n_head * head_dim`).
    pub head_dim: usize,
    /// `model.encoder.n_hidden` — SwiGLU FFN inner width, 4096.
    pub n_hidden: usize,
}

/// Decoder hparams (primary source: `config.json` `model.decoder`). Uses GQA
/// (`gqa_query_heads` > `kv_heads`) plus cross-attention to the text encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiaDecoderConfig {
    /// `model.decoder.n_layer` — 18 for Dia-1.6B.
    pub n_layer: usize,
    /// `model.decoder.n_embd` — hidden width, 2048.
    pub n_embd: usize,
    /// `model.decoder.gqa_query_heads` — Q-heads, 16.
    pub gqa_query_heads: usize,
    /// `model.decoder.kv_heads` — KV-heads (GQA broadcast), 4.
    pub kv_heads: usize,
    /// `model.decoder.gqa_head_dim` — 128 (Q head width).
    pub gqa_head_dim: usize,
    /// `model.decoder.cross_query_heads` — cross-attn Q-heads, 16.
    pub cross_query_heads: usize,
    /// `model.decoder.cross_head_dim` — cross-attn Q head width, 128.
    pub cross_head_dim: usize,
    /// `model.decoder.n_hidden` — SwiGLU FFN inner width, 8192.
    pub n_hidden: usize,
}

impl DiaEncoderConfig {
    /// All fields non-zero and `head_dim` even (RoPE pairs).
    ///
    /// Note: Dia's encoder deliberately does **not** obey the standard
    /// `n_embd == n_head * head_dim` invariant. Primary source has
    /// `n_embd=1024` with `n_head=16, head_dim=128` → attention Q/K/V/O
    /// projections span the residual stream (`n_embd`) and the attention
    /// hidden ([`Self::attn_hidden`] = `n_head * head_dim = 2048`).
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        self.n_layer != 0
            && self.n_embd != 0
            && self.n_head != 0
            && self.head_dim != 0
            && self.n_hidden != 0
    }

    /// Attention hidden width, `n_head * head_dim`. Q/K/V project residual
    /// `n_embd` to this width; O projects back.
    #[must_use]
    pub fn attn_hidden(&self) -> usize {
        self.n_head * self.head_dim
    }
}

impl DiaDecoderConfig {
    /// GQA constraints: `gqa_query_heads % kv_heads == 0`,
    /// `n_embd == gqa_query_heads * gqa_head_dim`, cross-attn axes
    /// non-zero.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        self.n_layer != 0
            && self.kv_heads != 0
            && self.gqa_query_heads != 0
            && self.gqa_head_dim != 0
            && self.gqa_query_heads % self.kv_heads == 0
            && self.n_embd == self.gqa_query_heads * self.gqa_head_dim
            && self.cross_query_heads != 0
            && self.cross_head_dim != 0
            && self.n_hidden != 0
    }

    /// KV hidden width, `kv_heads * gqa_head_dim` (GQA broadcast).
    #[must_use]
    pub fn kv_hidden_dim(&self) -> usize {
        self.kv_heads * self.gqa_head_dim
    }
}

/// Resolved Dia hparam snapshot — every field is transcribed from the
/// upstream `config.json` (module docstring) or from the DAC codec Dia
/// depends on (`sample_rate`).
#[derive(Debug, Clone, PartialEq)]
pub struct DiaConfig {
    /// Text-encoder hparams.
    pub encoder: DiaEncoderConfig,
    /// Decoder hparams.
    pub decoder: DiaDecoderConfig,
    /// `model.src_vocab_size` — byte-level source vocab (256).
    pub src_vocab_size: usize,
    /// `model.tgt_vocab_size` — per-channel audio target vocab (1028).
    pub tgt_vocab_size: usize,
    /// `data.channels` — number of audio codebook channels the decoder
    /// generates each step (9 for Dia; matches DAC 44.1 kHz's 9 quantizers).
    pub channels: usize,
    /// `data.delay_pattern` — one delay (in steps) per channel; length ==
    /// `channels`. Primary source: `[0, 8, 9, 10, 11, 12, 13, 14, 15]`.
    pub delay_pattern: Vec<usize>,
    /// `data.text_length` — max text-side sequence length (1024).
    pub text_length: usize,
    /// `data.audio_length` — max audio-side sequence length (3072).
    pub audio_length: usize,
    /// `data.text_pad_value` — source-side pad id (0).
    pub text_pad_value: u32,
    /// `data.audio_bos_value` — decoder BOS id (1026).
    pub audio_bos_value: u32,
    /// `data.audio_eos_value` — decoder EOS id (1024).
    pub audio_eos_value: u32,
    /// `data.audio_pad_value` — decoder pad id (1025).
    pub audio_pad_value: u32,
    /// `model.normalization_layer_epsilon` — RMSNorm ε (1e-5).
    pub norm_eps: f32,
    /// `model.rope_max_timescale` — RoPE max timescale (10000).
    pub rope_max_timescale: f32,
    /// `model.rope_min_timescale` — RoPE min timescale (1).
    pub rope_min_timescale: f32,
    /// PCM sample rate Dia emits — 44_100 (inherited from DAC 44.1 kHz;
    /// **not** written in the upstream `config.json`, taken from the codec
    /// `_load_dac_model` fetches).
    pub sample_rate: u32,
}

impl DiaConfig {
    /// Primary-source Dia-1.6B config (every value transcribed from
    /// `huggingface.co/nari-labs/Dia-1.6B/config.json`).
    #[must_use]
    pub fn dia_1_6b() -> Self {
        Self {
            encoder: DiaEncoderConfig {
                n_layer: 12,
                n_embd: 1024,
                n_head: 16,
                head_dim: 128,
                n_hidden: 4096,
            },
            decoder: DiaDecoderConfig {
                n_layer: 18,
                n_embd: 2048,
                gqa_query_heads: 16,
                kv_heads: 4,
                gqa_head_dim: 128,
                cross_query_heads: 16,
                cross_head_dim: 128,
                n_hidden: 8192,
            },
            src_vocab_size: 256,
            tgt_vocab_size: 1028,
            channels: 9,
            delay_pattern: vec![0, 8, 9, 10, 11, 12, 13, 14, 15],
            text_length: 1024,
            audio_length: 3072,
            text_pad_value: 0,
            audio_bos_value: 1026,
            audio_eos_value: 1024,
            audio_pad_value: 1025,
            norm_eps: 1e-5,
            rope_max_timescale: 10_000.0,
            rope_min_timescale: 1.0,
            sample_rate: DIA_SAMPLE_RATE,
        }
    }

    /// Miniature well-formed config for shape / stability tests. Dims are
    /// tiny so synthesized-weight builds fit in KB; the *shape relationships*
    /// (GQA split, even head_dim, channels == delay_pattern.len()) mirror
    /// the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            encoder: DiaEncoderConfig {
                n_layer: 2,
                n_embd: 16,
                n_head: 4,
                head_dim: 4,
                n_hidden: 32,
            },
            decoder: DiaDecoderConfig {
                n_layer: 2,
                n_embd: 16,
                gqa_query_heads: 4,
                kv_heads: 2,
                gqa_head_dim: 4,
                cross_query_heads: 4,
                cross_head_dim: 4,
                n_hidden: 32,
            },
            src_vocab_size: 8,
            tgt_vocab_size: 12,
            channels: 3,
            delay_pattern: vec![0, 1, 2],
            text_length: 32,
            audio_length: 32,
            text_pad_value: 0,
            audio_bos_value: 10,
            audio_eos_value: 8,
            audio_pad_value: 9,
            norm_eps: 1e-5,
            rope_max_timescale: 10_000.0,
            rope_min_timescale: 1.0,
            sample_rate: DIA_SAMPLE_RATE,
        }
    }

    /// Rejects `0`-placeholder / GQA-ill-formed configs before any forward
    /// runs (FR-EX-08 — a shape-only converter path fails loudly here, not
    /// deep inside a GEMM).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        if !self.encoder.is_well_formed() {
            return Err(VokraError::InvalidArgument(format!(
                "dia config: encoder ill-formed (n_layer={}, n_embd={}, n_head={}, \
                 head_dim={}, n_hidden={}) — expected all fields > 0",
                self.encoder.n_layer,
                self.encoder.n_embd,
                self.encoder.n_head,
                self.encoder.head_dim,
                self.encoder.n_hidden,
            )));
        }
        if !self.decoder.is_well_formed() {
            return Err(VokraError::InvalidArgument(format!(
                "dia config: decoder ill-formed (n_layer={}, n_embd={}, gqa_query_heads={}, \
                 kv_heads={}, gqa_head_dim={}, cross_query_heads={}, cross_head_dim={}, \
                 n_hidden={}) — expected GQA well-formed (query % kv == 0, \
                 n_embd == query * head_dim)",
                self.decoder.n_layer,
                self.decoder.n_embd,
                self.decoder.gqa_query_heads,
                self.decoder.kv_heads,
                self.decoder.gqa_head_dim,
                self.decoder.cross_query_heads,
                self.decoder.cross_head_dim,
                self.decoder.n_hidden,
            )));
        }
        if self.encoder.head_dim % 2 != 0 || self.decoder.gqa_head_dim % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "dia config: RoPE requires even head_dim (encoder.head_dim={}, \
                 decoder.gqa_head_dim={})",
                self.encoder.head_dim, self.decoder.gqa_head_dim,
            )));
        }
        if self.src_vocab_size == 0
            || self.tgt_vocab_size == 0
            || self.channels == 0
            || self.text_length == 0
            || self.audio_length == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "dia config: zero-size hparam (src_vocab={}, tgt_vocab={}, channels={}, \
                 text_length={}, audio_length={})",
                self.src_vocab_size,
                self.tgt_vocab_size,
                self.channels,
                self.text_length,
                self.audio_length,
            )));
        }
        if self.delay_pattern.len() != self.channels {
            return Err(VokraError::InvalidArgument(format!(
                "dia config: delay_pattern.len() {} != channels {}",
                self.delay_pattern.len(),
                self.channels,
            )));
        }
        // The special ids must fit within the target vocab; upstream places
        // them at the top of `tgt_vocab_size` (EOS=1024, PAD=1025, BOS=1026
        // in a 1028-wide vocab).
        for (name, id) in [
            ("audio_bos_value", self.audio_bos_value),
            ("audio_eos_value", self.audio_eos_value),
            ("audio_pad_value", self.audio_pad_value),
        ] {
            if (id as usize) >= self.tgt_vocab_size {
                return Err(VokraError::InvalidArgument(format!(
                    "dia config: {name}={id} does not fit in tgt_vocab_size={}",
                    self.tgt_vocab_size,
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Per-block encoder weights (pre-norm MHA + SwiGLU FFN).
///
/// Field names track the upstream Dia block shape: `norm_1` before attention,
/// `attn.{q,k,v,o}_proj`, `norm_2` before FFN, `ffn.{gate,up,down}_proj` for
/// the SwiGLU stage.
///
/// Dia's encoder deliberately projects the residual stream (`n_embd=1024`)
/// to a **larger** attention hidden ([`DiaEncoderConfig::attn_hidden`] =
/// `n_head * head_dim = 2048`), so the QKV/O shapes are asymmetric — see
/// each field's docstring.
#[derive(Debug, Clone)]
pub struct DiaEncoderBlockWeights {
    /// Pre-attention RMSNorm γ, shape `[n_embd]`.
    pub norm_1: Vec<f32>,
    /// Q projection weight (transposed), shape `[n_embd, attn_hidden]`.
    pub q_proj: Vec<f32>,
    /// K projection weight (transposed), shape `[n_embd, attn_hidden]`.
    pub k_proj: Vec<f32>,
    /// V projection weight (transposed), shape `[n_embd, attn_hidden]`.
    pub v_proj: Vec<f32>,
    /// Output projection weight (transposed), shape `[attn_hidden, n_embd]`.
    pub o_proj: Vec<f32>,
    /// Pre-FFN RMSNorm γ, shape `[n_embd]`.
    pub norm_2: Vec<f32>,
    /// SwiGLU gate proj (transposed), shape `[n_embd, n_hidden]`.
    pub gate_proj: Vec<f32>,
    /// SwiGLU up proj (transposed), shape `[n_embd, n_hidden]`.
    pub up_proj: Vec<f32>,
    /// SwiGLU down proj (transposed), shape `[n_hidden, n_embd]`.
    pub down_proj: Vec<f32>,
}

/// Per-block decoder weights (pre-norm self-attention with GQA +
/// pre-norm cross-attention + pre-norm SwiGLU FFN).
#[derive(Debug, Clone)]
pub struct DiaDecoderBlockWeights {
    // --- self-attention (GQA) ---
    /// Pre-self-attn RMSNorm γ, shape `[n_embd]`.
    pub sa_norm: Vec<f32>,
    /// Q projection (transposed), shape `[n_embd, n_embd]`.
    pub sa_q_proj: Vec<f32>,
    /// K projection (transposed), shape `[n_embd, kv_hidden]`
    /// (`kv_hidden = kv_heads * gqa_head_dim`).
    pub sa_k_proj: Vec<f32>,
    /// V projection (transposed), shape `[n_embd, kv_hidden]`.
    pub sa_v_proj: Vec<f32>,
    /// Output projection (transposed), shape `[n_embd, n_embd]`.
    pub sa_o_proj: Vec<f32>,
    // --- cross-attention ---
    /// Pre-cross-attn RMSNorm γ, shape `[n_embd]`.
    pub xa_norm: Vec<f32>,
    /// Q projection (transposed), shape
    /// `[n_embd, cross_query_heads * cross_head_dim]`.
    pub xa_q_proj: Vec<f32>,
    /// K projection (transposed), shape
    /// `[enc_n_embd, cross_query_heads * cross_head_dim]`.
    pub xa_k_proj: Vec<f32>,
    /// V projection (transposed), shape
    /// `[enc_n_embd, cross_query_heads * cross_head_dim]`.
    pub xa_v_proj: Vec<f32>,
    /// Output projection (transposed), shape
    /// `[cross_query_heads * cross_head_dim, n_embd]`.
    pub xa_o_proj: Vec<f32>,
    // --- SwiGLU FFN ---
    /// Pre-FFN RMSNorm γ, shape `[n_embd]`.
    pub ffn_norm: Vec<f32>,
    /// Gate proj (transposed), shape `[n_embd, n_hidden]`.
    pub gate_proj: Vec<f32>,
    /// Up proj (transposed), shape `[n_embd, n_hidden]`.
    pub up_proj: Vec<f32>,
    /// Down proj (transposed), shape `[n_hidden, n_embd]`.
    pub down_proj: Vec<f32>,
}

/// Dia weight store: text encoder + decoder + per-channel logits heads.
///
/// [`Self::synthesized`] builds a deterministic fixture (SplitMix64 + Xavier)
/// against `config` so shape / dtype / size can be exercised without the
/// real HF checkpoint. Real-checkpoint binding is a follow-up
/// (T29-equivalent — tensor-name manifest fetch from the upstream release).
#[derive(Debug, Clone)]
pub struct DiaWeights {
    /// Text-encoder input embedding, shape `[src_vocab_size, enc_n_embd]`.
    pub text_embedding: Vec<f32>,
    /// Encoder blocks in order.
    pub encoder_blocks: Vec<DiaEncoderBlockWeights>,
    /// Final encoder RMSNorm γ, shape `[enc_n_embd]`.
    pub encoder_norm: Vec<f32>,
    /// Per-channel decoder input embeddings, `channels` tables each of
    /// shape `[tgt_vocab_size, dec_n_embd]`.
    pub channel_embeddings: Vec<Vec<f32>>,
    /// Decoder blocks in order.
    pub decoder_blocks: Vec<DiaDecoderBlockWeights>,
    /// Final decoder RMSNorm γ, shape `[dec_n_embd]`.
    pub decoder_norm: Vec<f32>,
    /// Per-channel logit heads (transposed), `channels` tables each of
    /// shape `[dec_n_embd, tgt_vocab_size]`.
    pub logit_heads: Vec<Vec<f32>>,
    /// `true` when built by [`Self::synthesized`] — never a real
    /// upstream checkpoint. Real-checkpoint bindings set this to `false`.
    pub is_synthesized: bool,
}

impl DiaWeights {
    /// Builds a deterministic synthesized fixture from `config` and `seed`.
    ///
    /// Draws are Xavier-uniform ± `sqrt(6 / (fan_in + fan_out))` via a
    /// [`SplitMix64`] stream — reproducible, allocation-only, zero-dep.
    /// Every RMSNorm γ starts at `1.0`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward` fails.
    pub fn synthesized(config: &DiaConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        let enc = &config.encoder;
        let dec = &config.decoder;
        let enc_attn_hidden = enc.attn_hidden();
        let dec_kv_hidden = dec.kv_hidden_dim();
        let xa_qhidden = dec.cross_query_heads * dec.cross_head_dim;

        let text_embedding = xavier(
            &mut rng,
            config.src_vocab_size * enc.n_embd,
            config.src_vocab_size,
            enc.n_embd,
        );
        let mut encoder_blocks = Vec::with_capacity(enc.n_layer);
        for _ in 0..enc.n_layer {
            encoder_blocks.push(DiaEncoderBlockWeights {
                norm_1: vec![1.0; enc.n_embd],
                q_proj: xavier(
                    &mut rng,
                    enc.n_embd * enc_attn_hidden,
                    enc.n_embd,
                    enc_attn_hidden,
                ),
                k_proj: xavier(
                    &mut rng,
                    enc.n_embd * enc_attn_hidden,
                    enc.n_embd,
                    enc_attn_hidden,
                ),
                v_proj: xavier(
                    &mut rng,
                    enc.n_embd * enc_attn_hidden,
                    enc.n_embd,
                    enc_attn_hidden,
                ),
                o_proj: xavier(
                    &mut rng,
                    enc_attn_hidden * enc.n_embd,
                    enc_attn_hidden,
                    enc.n_embd,
                ),
                norm_2: vec![1.0; enc.n_embd],
                gate_proj: xavier(
                    &mut rng,
                    enc.n_embd * enc.n_hidden,
                    enc.n_embd,
                    enc.n_hidden,
                ),
                up_proj: xavier(
                    &mut rng,
                    enc.n_embd * enc.n_hidden,
                    enc.n_embd,
                    enc.n_hidden,
                ),
                down_proj: xavier(
                    &mut rng,
                    enc.n_hidden * enc.n_embd,
                    enc.n_hidden,
                    enc.n_embd,
                ),
            });
        }
        let encoder_norm = vec![1.0; enc.n_embd];

        let mut channel_embeddings = Vec::with_capacity(config.channels);
        for _ in 0..config.channels {
            channel_embeddings.push(xavier(
                &mut rng,
                config.tgt_vocab_size * dec.n_embd,
                config.tgt_vocab_size,
                dec.n_embd,
            ));
        }
        let mut decoder_blocks = Vec::with_capacity(dec.n_layer);
        for _ in 0..dec.n_layer {
            decoder_blocks.push(DiaDecoderBlockWeights {
                sa_norm: vec![1.0; dec.n_embd],
                sa_q_proj: xavier(&mut rng, dec.n_embd * dec.n_embd, dec.n_embd, dec.n_embd),
                sa_k_proj: xavier(
                    &mut rng,
                    dec.n_embd * dec_kv_hidden,
                    dec.n_embd,
                    dec_kv_hidden,
                ),
                sa_v_proj: xavier(
                    &mut rng,
                    dec.n_embd * dec_kv_hidden,
                    dec.n_embd,
                    dec_kv_hidden,
                ),
                sa_o_proj: xavier(&mut rng, dec.n_embd * dec.n_embd, dec.n_embd, dec.n_embd),
                xa_norm: vec![1.0; dec.n_embd],
                xa_q_proj: xavier(&mut rng, dec.n_embd * xa_qhidden, dec.n_embd, xa_qhidden),
                xa_k_proj: xavier(&mut rng, enc.n_embd * xa_qhidden, enc.n_embd, xa_qhidden),
                xa_v_proj: xavier(&mut rng, enc.n_embd * xa_qhidden, enc.n_embd, xa_qhidden),
                xa_o_proj: xavier(&mut rng, xa_qhidden * dec.n_embd, xa_qhidden, dec.n_embd),
                ffn_norm: vec![1.0; dec.n_embd],
                gate_proj: xavier(
                    &mut rng,
                    dec.n_embd * dec.n_hidden,
                    dec.n_embd,
                    dec.n_hidden,
                ),
                up_proj: xavier(
                    &mut rng,
                    dec.n_embd * dec.n_hidden,
                    dec.n_embd,
                    dec.n_hidden,
                ),
                down_proj: xavier(
                    &mut rng,
                    dec.n_hidden * dec.n_embd,
                    dec.n_hidden,
                    dec.n_embd,
                ),
            });
        }
        let decoder_norm = vec![1.0; dec.n_embd];
        let mut logit_heads = Vec::with_capacity(config.channels);
        for _ in 0..config.channels {
            logit_heads.push(xavier(
                &mut rng,
                dec.n_embd * config.tgt_vocab_size,
                dec.n_embd,
                config.tgt_vocab_size,
            ));
        }

        Ok(Self {
            text_embedding,
            encoder_blocks,
            encoder_norm,
            channel_embeddings,
            decoder_blocks,
            decoder_norm,
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

/// Dia TTS engine handle.
///
/// Carries the resolved config, weight store, and an optional DAC codec
/// bind ([`DacCodecGguf`] — MIT). [`Self::synthesize`] is the primary text →
/// PCM entry point; until real weights are bound (see the module docstring)
/// it returns [`VokraError::NotImplemented`] with a message naming the
/// blocker (FR-EX-08 — never a silent zero-fill fallback).
#[derive(Debug, Clone)]
pub struct DiaTts {
    cfg: DiaConfig,
    weights: DiaWeights,
    /// Optional DAC codec bind. Injected via [`Self::with_dac`]; the real
    /// synth path consumes the RVQ decode + DAC neural chain to produce
    /// 44.1 kHz PCM.
    dac: Option<DacCodecGguf>,
}

impl DiaTts {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// weight-store shapes against `cfg` (`n_layer` counts, channel table
    /// counts, per-tensor sizes) so a mismatched pair fails loudly here
    /// rather than deep inside a forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape mismatch.
    pub fn new(cfg: DiaConfig, weights: DiaWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let enc = &cfg.encoder;
        let dec = &cfg.decoder;
        // Encoder shape checks.
        if weights.text_embedding.len() != cfg.src_vocab_size * enc.n_embd {
            return Err(VokraError::InvalidArgument(format!(
                "dia weights: text_embedding.len()={} != src_vocab_size*enc_n_embd={}",
                weights.text_embedding.len(),
                cfg.src_vocab_size * enc.n_embd,
            )));
        }
        if weights.encoder_blocks.len() != enc.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "dia weights: encoder_blocks.len()={} != encoder.n_layer={}",
                weights.encoder_blocks.len(),
                enc.n_layer,
            )));
        }
        if weights.encoder_norm.len() != enc.n_embd {
            return Err(VokraError::InvalidArgument(format!(
                "dia weights: encoder_norm.len()={} != encoder.n_embd={}",
                weights.encoder_norm.len(),
                enc.n_embd,
            )));
        }
        let enc_attn_hidden = enc.attn_hidden();
        for (i, blk) in weights.encoder_blocks.iter().enumerate() {
            let expected_qkv = enc.n_embd * enc_attn_hidden;
            let expected_o = enc_attn_hidden * enc.n_embd;
            let expected_gate_up = enc.n_embd * enc.n_hidden;
            for (name, len, expected) in [
                ("norm_1", blk.norm_1.len(), enc.n_embd),
                ("q_proj", blk.q_proj.len(), expected_qkv),
                ("k_proj", blk.k_proj.len(), expected_qkv),
                ("v_proj", blk.v_proj.len(), expected_qkv),
                ("o_proj", blk.o_proj.len(), expected_o),
                ("norm_2", blk.norm_2.len(), enc.n_embd),
                ("gate_proj", blk.gate_proj.len(), expected_gate_up),
                ("up_proj", blk.up_proj.len(), expected_gate_up),
                ("down_proj", blk.down_proj.len(), enc.n_hidden * enc.n_embd),
            ] {
                if len != expected {
                    return Err(VokraError::InvalidArgument(format!(
                        "dia weights: encoder block {i} `{name}` len={len} != {expected}",
                    )));
                }
            }
        }
        // Decoder shape checks.
        if weights.channel_embeddings.len() != cfg.channels {
            return Err(VokraError::InvalidArgument(format!(
                "dia weights: channel_embeddings.len()={} != channels={}",
                weights.channel_embeddings.len(),
                cfg.channels,
            )));
        }
        for (i, tbl) in weights.channel_embeddings.iter().enumerate() {
            let expected = cfg.tgt_vocab_size * dec.n_embd;
            if tbl.len() != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "dia weights: channel_embeddings[{i}].len()={} != {expected}",
                    tbl.len(),
                )));
            }
        }
        if weights.decoder_blocks.len() != dec.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "dia weights: decoder_blocks.len()={} != decoder.n_layer={}",
                weights.decoder_blocks.len(),
                dec.n_layer,
            )));
        }
        if weights.decoder_norm.len() != dec.n_embd {
            return Err(VokraError::InvalidArgument(format!(
                "dia weights: decoder_norm.len()={} != decoder.n_embd={}",
                weights.decoder_norm.len(),
                dec.n_embd,
            )));
        }
        let kv_hidden = dec.kv_hidden_dim();
        let xa_qhidden = dec.cross_query_heads * dec.cross_head_dim;
        for (i, blk) in weights.decoder_blocks.iter().enumerate() {
            for (name, len, expected) in [
                ("sa_norm", blk.sa_norm.len(), dec.n_embd),
                ("sa_q_proj", blk.sa_q_proj.len(), dec.n_embd * dec.n_embd),
                ("sa_k_proj", blk.sa_k_proj.len(), dec.n_embd * kv_hidden),
                ("sa_v_proj", blk.sa_v_proj.len(), dec.n_embd * kv_hidden),
                ("sa_o_proj", blk.sa_o_proj.len(), dec.n_embd * dec.n_embd),
                ("xa_norm", blk.xa_norm.len(), dec.n_embd),
                ("xa_q_proj", blk.xa_q_proj.len(), dec.n_embd * xa_qhidden),
                ("xa_k_proj", blk.xa_k_proj.len(), enc.n_embd * xa_qhidden),
                ("xa_v_proj", blk.xa_v_proj.len(), enc.n_embd * xa_qhidden),
                ("xa_o_proj", blk.xa_o_proj.len(), xa_qhidden * dec.n_embd),
                ("ffn_norm", blk.ffn_norm.len(), dec.n_embd),
                ("gate_proj", blk.gate_proj.len(), dec.n_embd * dec.n_hidden),
                ("up_proj", blk.up_proj.len(), dec.n_embd * dec.n_hidden),
                ("down_proj", blk.down_proj.len(), dec.n_hidden * dec.n_embd),
            ] {
                if len != expected {
                    return Err(VokraError::InvalidArgument(format!(
                        "dia weights: decoder block {i} `{name}` len={len} != {expected}",
                    )));
                }
            }
        }
        if weights.logit_heads.len() != cfg.channels {
            return Err(VokraError::InvalidArgument(format!(
                "dia weights: logit_heads.len()={} != channels={}",
                weights.logit_heads.len(),
                cfg.channels,
            )));
        }
        for (i, tbl) in weights.logit_heads.iter().enumerate() {
            let expected = dec.n_embd * cfg.tgt_vocab_size;
            if tbl.len() != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "dia weights: logit_heads[{i}].len()={} != {expected}",
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

    /// Injects a [`DacCodecGguf`] — the terminal RVQ codes → PCM decoder.
    ///
    /// Dia's decoder outputs `channels` (9) RVQ codes per step; the DAC codec
    /// reduces them to a 44.1 kHz PCM waveform. Without a DAC bind
    /// [`Self::synthesize`] cannot honestly return audio (FR-EX-08).
    ///
    /// Cross-checks that the DAC codec has at least as many codebooks as
    /// Dia emits channels — a mismatch would misroute channel indices at
    /// decode time.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a codebook / sample-rate mismatch.
    pub fn with_dac(mut self, dac: DacCodecGguf) -> Result<Self> {
        if dac.attrs.n_codebooks < self.cfg.channels {
            return Err(VokraError::InvalidArgument(format!(
                "dia with_dac: dac has {} codebooks but Dia emits {} channels",
                dac.attrs.n_codebooks, self.cfg.channels,
            )));
        }
        if dac.sample_rate != self.cfg.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "dia with_dac: dac sample_rate {} Hz != Dia config sample_rate {} Hz",
                dac.sample_rate, self.cfg.sample_rate,
            )));
        }
        self.dac = Some(dac);
        Ok(self)
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &DiaConfig {
        &self.cfg
    }

    /// The bound DAC codec, if any.
    #[must_use]
    pub fn dac(&self) -> Option<&DacCodecGguf> {
        self.dac.as_ref()
    }

    /// True iff the weight store was built by [`DiaWeights::synthesized`]
    /// (never a real upstream checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Synthesizes PCM from a byte-level source token sequence.
    ///
    /// `text_ids` is a slice of source-side ids in `[0, src_vocab_size)`
    /// (byte-level for Dia: `src_vocab_size == 256`); the caller performs
    /// UTF-8 → byte-id mapping.
    ///
    /// This is the primary text → PCM entry point. **Real weights required**:
    /// synthesized-weight builds cannot produce meaningful audio (they'd be
    /// noise or a hallucinated "silence"), so this returns
    /// [`VokraError::NotImplemented`] naming the blocker. Callers verify the
    /// shape flow through [`DiaTts::new`] + [`DiaWeights::synthesized`]
    /// today; a follow-up wave binds the real HF checkpoint tensor names and
    /// wires the forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on `text_ids` length or an id ≥
    ///   `src_vocab_size`.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn synthesize(&self, text_ids: &[i64]) -> Result<Vec<f32>> {
        if text_ids.is_empty() {
            return Err(VokraError::InvalidArgument(
                "dia synthesize: text_ids is empty".to_owned(),
            ));
        }
        if text_ids.len() > self.cfg.text_length {
            return Err(VokraError::InvalidArgument(format!(
                "dia synthesize: text_ids.len()={} > text_length cap {}",
                text_ids.len(),
                self.cfg.text_length,
            )));
        }
        let vocab = self.cfg.src_vocab_size as i64;
        for (i, id) in text_ids.iter().enumerate() {
            if *id < 0 || *id >= vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "dia synthesize: text_ids[{i}]={id} out of [0, {vocab})",
                )));
            }
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "dia synthesize: this engine holds synthesized weights (deterministic \
                 fixture from DiaWeights::synthesized) — synthesized-weight PCM would \
                 be noise, not speech. Bind real Dia-1.6B weights (Apache 2.0, \
                 nari-labs/Dia-1.6B) before invoking synthesize. The shape flow \
                 (config validation, weight-store construction) is exercised through \
                 DiaTts::new; the real-checkpoint tensor-name manifest lands in a \
                 follow-up wave (T29-equivalent).",
            ));
        }
        if self.dac.is_none() {
            return Err(VokraError::NotImplemented(
                "dia synthesize: no DAC codec has been bound — call `.with_dac(\
                 DacCodecGguf::from_gguf(&dac_gguf)?)?` first. Dia's decoder emits \
                 9 RVQ codebook channels per step which the DAC 44.1 kHz codec \
                 reduces to PCM; without it there is nothing honest to return \
                 (FR-EX-08).",
            ));
        }
        Err(VokraError::NotImplemented(
            "dia synthesize: real weights are bound and a DAC codec is present, but \
             the encoder + delayed-AR decoder forward path has not landed yet. \
             Follow-up wave: transcribe the upstream tensor manifest and wire the \
             pre-norm MHA + GQA + cross-attn + SwiGLU forward through the \
             `Compute` seam (CosyVoice2 T07/T08 pattern).",
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
    /// (`huggingface.co/nari-labs/Dia-1.6B/config.json`) verbatim.
    #[test]
    fn dia_1_6b_matches_primary_source_config_json() {
        let c = DiaConfig::dia_1_6b();
        // model.encoder
        assert_eq!(c.encoder.n_layer, 12);
        assert_eq!(c.encoder.n_embd, 1024);
        assert_eq!(c.encoder.n_head, 16);
        assert_eq!(c.encoder.head_dim, 128);
        assert_eq!(c.encoder.n_hidden, 4096);
        // model.decoder
        assert_eq!(c.decoder.n_layer, 18);
        assert_eq!(c.decoder.n_embd, 2048);
        assert_eq!(c.decoder.gqa_query_heads, 16);
        assert_eq!(c.decoder.kv_heads, 4);
        assert_eq!(c.decoder.gqa_head_dim, 128);
        assert_eq!(c.decoder.cross_query_heads, 16);
        assert_eq!(c.decoder.cross_head_dim, 128);
        assert_eq!(c.decoder.n_hidden, 8192);
        // vocab / data
        assert_eq!(c.src_vocab_size, 256);
        assert_eq!(c.tgt_vocab_size, 1028);
        assert_eq!(c.channels, 9);
        assert_eq!(c.delay_pattern, vec![0, 8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(c.text_length, 1024);
        assert_eq!(c.audio_length, 3072);
        assert_eq!(c.text_pad_value, 0);
        assert_eq!(c.audio_bos_value, 1026);
        assert_eq!(c.audio_eos_value, 1024);
        assert_eq!(c.audio_pad_value, 1025);
        assert_eq!(c.norm_eps, 1e-5);
        assert_eq!(c.rope_max_timescale, 10_000.0);
        assert_eq!(c.rope_min_timescale, 1.0);
        // DAC 44.1 kHz inheritance.
        assert_eq!(c.sample_rate, 44_100);
        // Everything above adds up to a well-formed config.
        c.validate_for_forward().expect("dia-1.6b is well-formed");
    }

    #[test]
    fn tiny_config_is_well_formed() {
        DiaConfig::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny config is well-formed");
    }

    #[test]
    fn config_gqa_ill_formed_is_rejected() {
        let mut c = DiaConfig::tiny_for_tests();
        c.decoder.kv_heads = 3; // 4 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_odd_head_dim_is_rejected() {
        let mut c = DiaConfig::tiny_for_tests();
        c.decoder.gqa_head_dim = 5; // odd → RoPE fails
        c.decoder.n_embd = c.decoder.gqa_query_heads * 5;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_delay_pattern_length_must_equal_channels() {
        let mut c = DiaConfig::tiny_for_tests();
        c.delay_pattern.push(3);
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_special_id_out_of_range_is_rejected() {
        let mut c = DiaConfig::tiny_for_tests();
        c.audio_bos_value = c.tgt_vocab_size as u32; // >= tgt_vocab_size
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        let c = DiaConfig::tiny_for_tests();
        let w1 = DiaWeights::synthesized(&c, 0x42).expect("build 1");
        let w2 = DiaWeights::synthesized(&c, 0x42).expect("build 2");
        // Determinism.
        assert_eq!(w1.text_embedding, w2.text_embedding);
        assert_eq!(
            w1.encoder_blocks[0].q_proj, w2.encoder_blocks[0].q_proj,
            "same seed → same weights"
        );
        assert!(w1.is_synthesized);
        // Shape flow.
        assert_eq!(w1.text_embedding.len(), c.src_vocab_size * c.encoder.n_embd);
        assert_eq!(w1.encoder_blocks.len(), c.encoder.n_layer);
        assert_eq!(w1.decoder_blocks.len(), c.decoder.n_layer);
        assert_eq!(w1.channel_embeddings.len(), c.channels);
        assert_eq!(w1.logit_heads.len(), c.channels);
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let c = DiaConfig::tiny_for_tests();
        let w_a = DiaWeights::synthesized(&c, 1).expect("build a");
        let w_b = DiaWeights::synthesized(&c, 2).expect("build b");
        // Two distinct seeds must produce different Xavier draws (probability
        // of collision on the first row is vanishing).
        assert_ne!(w_a.text_embedding, w_b.text_embedding);
    }

    #[test]
    fn dia_tts_new_accepts_matching_config_and_weights() {
        let c = DiaConfig::tiny_for_tests();
        let w = DiaWeights::synthesized(&c, 7).expect("weights");
        let tts = DiaTts::new(c.clone(), w).expect("dia tts");
        assert_eq!(tts.config().encoder.n_embd, c.encoder.n_embd);
        assert!(tts.is_synthesized());
        assert!(tts.dac().is_none());
    }

    #[test]
    fn dia_tts_new_rejects_layer_count_mismatch() {
        let c = DiaConfig::tiny_for_tests();
        let mut w = DiaWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks.pop();
        assert!(matches!(
            DiaTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn dia_tts_new_rejects_tensor_size_mismatch() {
        let c = DiaConfig::tiny_for_tests();
        let mut w = DiaWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks[0].q_proj.pop();
        assert!(matches!(
            DiaTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesize_rejects_empty_ids() {
        let c = DiaConfig::tiny_for_tests();
        let w = DiaWeights::synthesized(&c, 7).expect("weights");
        let tts = DiaTts::new(c, w).expect("dia tts");
        assert!(matches!(
            tts.synthesize(&[]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesize_rejects_out_of_range_id() {
        let c = DiaConfig::tiny_for_tests();
        let vocab = c.src_vocab_size as i64;
        let w = DiaWeights::synthesized(&c, 7).expect("weights");
        let tts = DiaTts::new(c, w).expect("dia tts");
        assert!(matches!(
            tts.synthesize(&[vocab]),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            tts.synthesize(&[-1]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesize_rejects_too_long_ids() {
        let c = DiaConfig::tiny_for_tests();
        let text_length = c.text_length;
        let w = DiaWeights::synthesized(&c, 7).expect("weights");
        let tts = DiaTts::new(c, w).expect("dia tts");
        let too_long = vec![0i64; text_length + 1];
        assert!(matches!(
            tts.synthesize(&too_long),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// The primary NotImplemented path names the synthesized-weight blocker
    /// (FR-EX-08 — never a silent zero-fill).
    #[test]
    fn synthesize_on_synthesized_weights_is_loud_not_implemented() {
        let c = DiaConfig::tiny_for_tests();
        let w = DiaWeights::synthesized(&c, 7).expect("weights");
        let tts = DiaTts::new(c, w).expect("dia tts");
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
    fn expected_arch_is_dia() {
        assert_eq!(EXPECTED_ARCH, "dia");
    }
}
